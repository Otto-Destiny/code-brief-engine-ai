from __future__ import annotations

import ast
import configparser
import json
import re
from collections import defaultdict
from pathlib import PurePosixPath

from app.config import Settings
from app.domain_models import (
    EvidencePacket,
    FetchedFile,
    FileCandidate,
    FileSnippet,
    RepoTreeEntry,
    RepositoryAnalysis,
    RepositorySnapshot,
    TechnologySignal,
)
from app.repo_rules import (
    BINARY_EXTENSIONS,
    CONFIG_NAMES,
    DOC_BUILD_FILE_NAMES,
    DOC_DIR_HINTS,
    ENTRYPOINT_NAMES,
    EXCLUDED_DIRECTORIES,
    FRAMEWORK_DEPENDENCY_MAP,
    LOCKFILES,
    MANIFEST_FILE_NAMES,
    NON_STACK_LANGUAGE_IGNORES,
    PACKAGE_NAME_IGNORES,
    README_NAMES,
    SOURCE_DIR_HINTS,
    TEST_DIR_HINTS,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


class RepositoryAnalyzer:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def plan(self, snapshot: RepositorySnapshot) -> RepositoryAnalysis:
        skipped_counts: dict[str, int] = defaultdict(int)
        skipped_examples: dict[str, list[str]] = defaultdict(list)
        candidates: list[FileCandidate] = []

        for entry in snapshot.tree_entries:
            if entry.type != "blob":
                continue
            skip_reason = self._skip_reason(entry)
            if skip_reason is not None:
                skipped_counts[skip_reason] += 1
                if len(skipped_examples[skip_reason]) < 3:
                    skipped_examples[skip_reason].append(entry.path)
                continue
            candidates.append(self._build_candidate(entry))

        candidates.sort(key=lambda item: (-item.score, item.depth, item.path))
        selected_candidates = self._select_candidates(candidates)

        warnings: list[str] = []
        if snapshot.tree_truncated:
            warnings.append(
                "Repository tree was very large, so the index was intentionally bounded before evidence selection."
            )

        return RepositoryAnalysis(
            directory_outline=self._build_directory_outline(snapshot),
            candidates=selected_candidates,
            skipped_counts=dict(skipped_counts),
            skipped_examples=dict(skipped_examples),
            tree_warnings=warnings,
        )

    def build_manifest_highlights(self, fetched_files: list[FetchedFile]) -> list[str]:
        highlights: list[str] = []
        for file in fetched_files:
            if file.category not in {"manifest", "infra"}:
                continue
            summary = self._extract_manifest_excerpt(file.path, file.raw_text)
            if not summary:
                continue
            lines = [line for line in summary.splitlines() if line.strip()]
            if not lines:
                continue
            preview = " | ".join(lines[:2])
            highlights.append(f"{file.path}: {preview}")
        return highlights[:5]

    def extract_technologies(
        self,
        snapshot: RepositorySnapshot,
        fetched_files: list[FetchedFile],
    ) -> list[TechnologySignal]:
        collector: dict[str, dict[str, object]] = {}

        def add_signal(name: str | None, confidence: int, evidence: str) -> None:
            if not name:
                return
            record = collector.setdefault(name, {"confidence": 0, "evidence": set()})
            record["confidence"] = int(record["confidence"]) + confidence
            record["evidence"].add(evidence)

        languages = sorted(snapshot.metadata.languages.items(), key=lambda item: item[1], reverse=True)
        language_rank = 0
        for language, _ in languages:
            if language in NON_STACK_LANGUAGE_IGNORES:
                continue
            add_signal(language, max(3, 6 - language_rank), "repo_languages")
            language_rank += 1
            if language_rank >= 3:
                break

        for entry in snapshot.tree_entries:
            path_lower = entry.path.lower()
            if path_lower.endswith(".tf") or path_lower.endswith(".tfvars"):
                add_signal("Terraform", 3, entry.path)
            if path_lower.startswith(".github/workflows/"):
                add_signal("GitHub Actions", 2, entry.path)
            if "dockerfile" in path_lower or "docker-compose" in path_lower:
                add_signal("Docker", 3, entry.path)
            if "k8s/" in path_lower or "kubernetes/" in path_lower or path_lower.endswith("kustomization.yaml"):
                add_signal("Kubernetes", 3, entry.path)
            if path_lower.endswith("chart.yaml") or "/charts/" in path_lower:
                add_signal("Helm", 3, entry.path)

        for file in fetched_files:
            allow_raw_dependencies = self._manifest_context_for_path(file.path) == "runtime"
            dependency_weight = self._dependency_signal_weight(file.path, file.category)
            for dependency in self._extract_manifest_dependency_names(file.path, file.raw_text):
                tech = self._technology_label_from_dependency(dependency, allow_raw=allow_raw_dependencies)
                if tech:
                    add_signal(tech, dependency_weight, file.path)

            import_weight = 3 if self._looks_like_source(file.path) else 1
            for import_name in self._extract_import_like_names(file.raw_text):
                tech = self._technology_label_from_dependency(import_name, allow_raw=False)
                if tech:
                    add_signal(tech, import_weight, file.path)

            if file.path.lower().endswith("dockerfile"):
                add_signal("Docker", 4, file.path)

        signals = [
            TechnologySignal(
                name=name,
                confidence=int(record["confidence"]),
                evidence=sorted(str(item) for item in record["evidence"]),
            )
            for name, record in collector.items()
        ]
        signals.sort(key=lambda item: (-item.confidence, item.name))
        return signals[:8]

    def build_evidence_packet(
        self,
        snapshot: RepositorySnapshot,
        analysis: RepositoryAnalysis,
        fetched_files: list[FetchedFile],
        manifest_highlights: list[str],
        technology_signals: list[TechnologySignal],
        *,
        reduced: bool = False,
    ) -> EvidencePacket:
        max_chars = self._settings.retry_file_excerpt_chars if reduced else self._settings.max_file_excerpt_chars
        max_lines = self._settings.retry_file_excerpt_lines if reduced else self._settings.max_file_excerpt_lines

        snippets: list[FileSnippet] = []
        truncation_notes = list(analysis.tree_warnings)
        ordered_files = sorted(fetched_files, key=lambda item: (-item.score, item.path))
        if reduced:
            ordered_files = ordered_files[: min(5, len(ordered_files))]

        for fetched in ordered_files:
            excerpt, truncated = self._prepare_snippet_content(
                fetched,
                max_chars=max_chars,
                max_lines=max_lines,
            )
            if truncated:
                truncation_notes.append(f"{fetched.path}: excerpt truncated for context budget.")
            snippets.append(
                FileSnippet(
                    path=fetched.path,
                    category=fetched.category,
                    score=fetched.score,
                    reasons=fetched.reasons,
                    content=excerpt,
                    size_bytes=fetched.size_bytes,
                    content_chars=len(excerpt),
                    content_lines=len(excerpt.splitlines()) if excerpt else 0,
                    truncated=truncated,
                )
            )

        packet_text = self._compose_packet_text(snapshot, analysis, manifest_highlights, technology_signals, snippets)
        while len(packet_text) > self._settings.evidence_char_budget and snippets:
            dropped = snippets.pop()
            truncation_notes.append(f"{dropped.path}: dropped to fit the final evidence budget.")
            packet_text = self._compose_packet_text(snapshot, analysis, manifest_highlights, technology_signals, snippets)

        return EvidencePacket(
            text=packet_text,
            estimated_input_tokens=self._estimate_tokens(packet_text),
            selected_files=snippets,
            manifest_highlights=manifest_highlights,
            candidate_technologies=technology_signals,
            directory_outline=analysis.directory_outline,
            total_chars=len(packet_text),
            truncation_notes=truncation_notes,
        )

    def _compose_packet_text(
        self,
        snapshot: RepositorySnapshot,
        analysis: RepositoryAnalysis,
        manifest_highlights: list[str],
        technology_signals: list[TechnologySignal],
        snippets: list[FileSnippet],
    ) -> str:
        languages = self._format_languages(snapshot.metadata.languages)
        technologies = "\n".join(
            f"- {signal.name} [{signal.confidence}] via {', '.join(signal.evidence[:3])}"
            for signal in technology_signals
        ) or "- No deterministic technology signals were strong enough to list."
        manifest_section = "\n".join(f"- {item}" for item in manifest_highlights) or "- No manifest highlights extracted."
        snippet_sections: list[str] = []
        for snippet in snippets:
            reasons = ", ".join(snippet.reasons[:4])
            snippet_sections.append(
                "\n".join(
                    [
                        f"### {snippet.path}",
                        f"category={snippet.category}; reasons={reasons}",
                        snippet.content or "No text extracted.",
                    ]
                )
            )

        snippets_text = "\n\n".join(snippet_sections) if snippet_sections else "No text file excerpts were selected."
        topics = ", ".join(snapshot.metadata.topics) if snapshot.metadata.topics else "None"
        description = snapshot.metadata.description or "No repository description provided."

        return "\n".join(
            [
                "Repository overview",
                f"- Name: {snapshot.metadata.full_name}",
                f"- Description: {description}",
                f"- Topics: {topics}",
                f"- Languages: {languages}",
                f"- Tree source: {snapshot.tree_source}",
                "",
                "Directory outline",
                analysis.directory_outline or "- No repository tree data available.",
                "",
                "Candidate technologies",
                technologies,
                "",
                "Manifest highlights",
                manifest_section,
                "",
                "Selected evidence files",
                snippets_text,
            ]
        )

    def _build_directory_outline(self, snapshot: RepositorySnapshot) -> str:
        top_level_counts: dict[str, int] = defaultdict(int)
        second_level_counts: dict[str, int] = defaultdict(int)
        top_level_files: list[str] = []

        for entry in snapshot.tree_entries:
            if entry.type != "blob":
                continue
            parts = entry.path.split("/")
            if len(parts) == 1:
                top_level_files.append(parts[0])
                continue
            top_level_counts[parts[0]] += 1
            if len(parts) >= 2:
                second_level_counts[f"{parts[0]}/{parts[1]}"] += 1

        prioritized_top = sorted(top_level_counts.items(), key=lambda item: (-self._directory_priority(item[0]), -item[1], item[0]))
        prioritized_second = sorted(
            second_level_counts.items(),
            key=lambda item: (-self._directory_priority(item[0]), -item[1], item[0]),
        )

        lines: list[str] = []
        for name, count in prioritized_top[:6]:
            lines.append(f"- {name}/ ({count} files)")
        for name, count in prioritized_second[:4]:
            if name.count("/") == 1 and name.split("/")[0] in {item[0] for item in prioritized_top[:3]}:
                lines.append(f"- {name}/ ({count} files)")
        if top_level_files:
            listed = ", ".join(sorted(top_level_files)[:8])
            lines.append(f"- Top-level files: {listed}")
        if snapshot.tree_truncated:
            lines.append("- Note: the repository was large, so the outline is intentionally bounded.")
        return "\n".join(lines)

    def _skip_reason(self, entry: RepoTreeEntry) -> str | None:
        path = PurePosixPath(entry.path)
        basename = path.name.lower()
        suffix = path.suffix.lower()
        path_parts = {part.lower() for part in path.parts[:-1]}
        if path_parts & EXCLUDED_DIRECTORIES:
            return "excluded_directory"
        if basename in LOCKFILES:
            return "lockfile"
        if suffix in BINARY_EXTENSIONS:
            return "binary_or_media"
        if entry.size and entry.size > self._settings.max_file_bytes_to_fetch:
            return "oversized_file"
        return None

    def _build_candidate(self, entry: RepoTreeEntry) -> FileCandidate:
        category = self._categorize(entry.path)
        reasons: list[str] = []
        score = 0.0
        path = entry.path
        basename = PurePosixPath(path).name.lower()
        depth = max(len(PurePosixPath(path).parts) - 1, 0)
        is_test = self._is_test_path(path)
        is_docs_build = self._is_docs_build_file(path)
        is_infra = self._is_infra_path(path)

        if self._is_readme(path):
            score += 110
            reasons.append("readme")
        if basename in MANIFEST_FILE_NAMES and not is_infra:
            score += self._manifest_score(path)
            reasons.append("manifest")
        if path.lower().startswith("docs/") or basename in {"mkdocs.yml", "readthedocs.yml"}:
            score += 12 if basename in MANIFEST_FILE_NAMES else 56
            reasons.append("documentation")
        if path.lower().startswith(".github/workflows/"):
            score += 50
            reasons.append("ci")
        if is_infra:
            score += 52
            reasons.append("runtime-config")
        stem = PurePosixPath(path).stem.lower()
        if stem in ENTRYPOINT_NAMES and not is_test and not is_docs_build:
            score += 45
            reasons.append("entrypoint")
        if self._looks_like_source(path) and basename not in MANIFEST_FILE_NAMES:
            score += 28
            reasons.append("source")
        if depth == 0:
            score += 18
            reasons.append("top-level")
        score += max(0, 12 - (depth * 2))
        if is_test:
            score -= 40
        if is_docs_build:
            score -= 30
        if entry.size:
            score -= min(entry.size / 8192, 18)
        return FileCandidate(
            path=path,
            category=category,
            score=round(score, 2),
            reasons=reasons or ["text-file"],
            size_bytes=entry.size,
            depth=depth,
        )

    def _select_candidates(self, candidates: list[FileCandidate]) -> list[FileCandidate]:
        if not candidates:
            return []

        quotas = {
            "readme": 1,
            "manifest": 2,
            "infra": 1,
            "docs": 1,
            "source": 3,
            "other": 1,
        }
        selected: list[FileCandidate] = []
        selected_paths: set[str] = set()
        category_counts: dict[str, int] = defaultdict(int)
        top_level_counts: dict[str, int] = defaultdict(int)
        parent_counts: dict[str, int] = defaultdict(int)

        for candidate in candidates:
            if not self._can_select_candidate(candidate, category_counts, top_level_counts, parent_counts, quotas):
                continue
            selected.append(candidate)
            selected_paths.add(candidate.path)
            category_counts[candidate.category] += 1
            top_level_counts[self._top_level_key(candidate.path)] += 1
            parent_counts[self._parent_key(candidate.path)] += 1
            if len(selected) >= self._settings.max_selected_files:
                return selected
            if self._should_stop_early(selected, category_counts):
                return selected

        for candidate in candidates:
            if candidate.path in selected_paths:
                continue
            if self._would_overconcentrate(candidate, top_level_counts, parent_counts):
                continue
            selected.append(candidate)
            selected_paths.add(candidate.path)
            top_level_counts[self._top_level_key(candidate.path)] += 1
            parent_counts[self._parent_key(candidate.path)] += 1
            if len(selected) >= self._settings.max_selected_files:
                break
        return selected

    def _can_select_candidate(
        self,
        candidate: FileCandidate,
        category_counts: dict[str, int],
        top_level_counts: dict[str, int],
        parent_counts: dict[str, int],
        quotas: dict[str, int],
    ) -> bool:
        limit = quotas.get(candidate.category, 1)
        if category_counts[candidate.category] >= limit:
            return False
        if self._would_overconcentrate(candidate, top_level_counts, parent_counts):
            return False
        return True

    def _would_overconcentrate(
        self,
        candidate: FileCandidate,
        top_level_counts: dict[str, int],
        parent_counts: dict[str, int],
    ) -> bool:
        top_level = self._top_level_key(candidate.path)
        parent = self._parent_key(candidate.path)
        if candidate.category == "source" and top_level_counts[top_level] >= 2:
            return True
        if candidate.category == "source" and parent_counts[parent] >= 2:
            return True
        return top_level_counts[top_level] >= 3

    @staticmethod
    def _should_stop_early(selected: list[FileCandidate], category_counts: dict[str, int]) -> bool:
        total = len(selected)
        has_readme = category_counts["readme"] > 0
        has_manifest = category_counts["manifest"] > 0
        has_source = category_counts["source"] > 0
        has_docs = category_counts["docs"] > 0
        has_infra = category_counts["infra"] > 0
        has_context = has_readme or has_docs or category_counts["other"] > 0

        if total >= 5 and has_manifest and has_source and (has_readme or has_infra):
            return True
        if total >= 6 and has_source and has_context and (has_manifest or has_infra):
            return True
        return False

    def _categorize(self, path: str) -> str:
        lower_path = path.lower()
        basename = PurePosixPath(path).name.lower()
        if self._is_readme(path):
            return "readme"
        if self._is_infra_path(path):
            return "infra"
        if basename in MANIFEST_FILE_NAMES:
            return "manifest"
        if any(part in DOC_DIR_HINTS for part in lower_path.split("/")):
            return "docs"
        if any(part in SOURCE_DIR_HINTS for part in lower_path.split("/")) or (
            PurePosixPath(path).stem.lower() in ENTRYPOINT_NAMES
            and not self._is_test_path(path)
            and not self._is_docs_build_file(path)
        ):
            return "source"
        return "other"

    def _looks_like_source(self, path: str) -> bool:
        parts = [part.lower() for part in PurePosixPath(path).parts]
        if any(part in TEST_DIR_HINTS for part in parts):
            return False
        if any(part in SOURCE_DIR_HINTS for part in parts):
            return True
        return PurePosixPath(path).suffix.lower() in {".go", ".java", ".js", ".jsx", ".kt", ".py", ".rb", ".rs", ".ts", ".tsx"}

    def _is_readme(self, path: str) -> bool:
        basename = PurePosixPath(path).name.lower()
        return basename in README_NAMES or basename.startswith("readme.")

    def _is_test_path(self, path: str) -> bool:
        return any(part.lower() in TEST_DIR_HINTS for part in PurePosixPath(path).parts)

    def _is_docs_build_file(self, path: str) -> bool:
        parts = [part.lower() for part in PurePosixPath(path).parts]
        basename = PurePosixPath(path).name.lower()
        return basename in DOC_BUILD_FILE_NAMES and any(part in DOC_DIR_HINTS for part in parts[:-1])

    def _is_infra_path(self, path: str) -> bool:
        lower_path = path.lower()
        basename = PurePosixPath(path).name.lower()
        if self._is_docs_build_file(path):
            return False
        return lower_path.startswith(".github/workflows/") or basename in CONFIG_NAMES or "dockerfile" in basename


    def _manifest_score(self, path: str) -> float:
        basename = PurePosixPath(path).name.lower()
        context = self._manifest_context_for_path(path)
        score = 68.0

        if basename in {"pyproject.toml", "package.json", "go.mod", "cargo.toml", "pom.xml", "composer.json"}:
            score += 24
        elif basename in {"build.gradle", "build.gradle.kts", "setup.cfg", "setup.py"}:
            score += 18
        elif basename == "requirements.txt":
            score += 12
        elif basename.startswith("requirements"):
            score += 4

        if context == "runtime":
            score += 16
        elif context == "dev":
            score -= 18
        elif context in {"docs", "test"}:
            score -= 24

        return score

    @staticmethod
    def _top_level_key(path: str) -> str:
        return PurePosixPath(path).parts[0] if "/" in path else "__root__"

    @staticmethod
    def _parent_key(path: str) -> str:
        parent = str(PurePosixPath(path).parent)
        return parent if parent != "." else "__root__"

    def _directory_priority(self, path: str) -> int:
        lowered = path.lower()
        if lowered.startswith("src") or lowered.startswith("app") or lowered.startswith("cmd"):
            return 5
        if lowered.startswith("tests") or lowered.startswith("test"):
            return 4
        if lowered.startswith("docs") or lowered.startswith("doc"):
            return 4
        if lowered.startswith(".github") or lowered.startswith("infra"):
            return 3
        return 1

    def _prepare_snippet_content(
        self,
        fetched: FetchedFile,
        *,
        max_chars: int,
        max_lines: int,
    ) -> tuple[str, bool]:
        if fetched.category == "manifest":
            content = self._extract_manifest_excerpt(fetched.path, fetched.raw_text)
        elif fetched.category == "source":
            content = self._extract_source_excerpt(fetched.path, fetched.raw_text)
        elif fetched.category in {"readme", "docs"}:
            content = self._extract_document_excerpt(fetched.raw_text)
        elif fetched.category == "infra":
            content = self._extract_infra_excerpt(fetched.path, fetched.raw_text)
        else:
            content = self._extract_generic_excerpt(fetched.raw_text)
        if not content:
            content = fetched.raw_text
        return self._truncate_text(content, max_chars=max_chars, max_lines=max_lines)

    def _extract_document_excerpt(self, text: str) -> str:
        output: list[str] = []
        prose_count = 0
        heading_count = 0
        bullet_count = 0
        lines = text.splitlines()
        index = 0
        while index < len(lines) and len(output) < 18:
            stripped = lines[index].strip()
            if not stripped:
                index += 1
                continue
            next_line = lines[index + 1].strip() if index + 1 < len(lines) else ""
            is_rst_heading = bool(
                next_line
                and len(next_line) >= max(3, len(stripped) // 2)
                and set(next_line) <= {"=", "-", "~", "^", "`"}
            )
            if (stripped.startswith("#") or is_rst_heading) and heading_count < 6:
                heading_text = stripped.lstrip("#").strip()
                if heading_text:
                    output.append(heading_text)
                    heading_count += 1
                index += 2 if is_rst_heading else 1
                continue
            if stripped.startswith(("- ", "* ", "1. ", "2. ", "3. ")) and bullet_count < 6:
                output.append(stripped)
                bullet_count += 1
                index += 1
                continue
            if prose_count < 10:
                output.append(stripped)
                prose_count += 1
            index += 1
        return "\n".join(output)

    def _extract_manifest_excerpt(self, path: str, text: str) -> str:
        basename = PurePosixPath(path).name.lower()
        lines: list[str] = []
        dependencies = self._extract_manifest_dependency_names(path, text)
        if dependencies:
            prefix = "Dependencies"
            if basename == "go.mod":
                prefix = "Modules"
            elif basename == "cargo.toml":
                prefix = "Crates"
            elif basename in {"pom.xml", "build.gradle", "build.gradle.kts"}:
                prefix = "Packages"
            lines.append(f"{prefix}: {', '.join(dependencies[:10])}")

        if basename == "package.json":
            scripts = self._parse_package_json_scripts(text)
            if scripts:
                lines.append(f"Scripts: {', '.join(scripts[:6])}")
        elif basename == "dockerfile":
            images = self._parse_dockerfile_images(text)
            if images:
                lines.append(f"Base images: {', '.join(images[:3])}")
        elif basename in {"docker-compose.yml", "docker-compose.yaml"}:
            services = self._parse_compose_services(text)
            if services:
                lines.append(f"Services: {', '.join(services[:6])}")
        elif basename in {"setup.py", "setup.cfg"}:
            setup_lines = self._extract_key_lines(text, limit=6, patterns=("name=", "install_requires", "extras_require", "entry_points"))
            lines.extend(setup_lines[:3])

        if not lines and basename.startswith("requirements"):
            requirement_lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
            if requirement_lines:
                lines.append(f"Requirements entries: {', '.join(requirement_lines[:6])}")

        if not lines:
            return "\n".join(self._extract_key_lines(text, limit=6))
        return "\n".join(_unique(lines))

    def _extract_infra_excerpt(self, path: str, text: str) -> str:
        basename = PurePosixPath(path).name.lower()
        lines: list[str] = []
        if basename == "dockerfile":
            images = self._parse_dockerfile_images(text)
            if images:
                lines.append(f"Base images: {', '.join(images[:3])}")
        if basename in {"docker-compose.yml", "docker-compose.yaml"}:
            services = self._parse_compose_services(text)
            if services:
                lines.append(f"Services: {', '.join(services[:6])}")
        lines.extend(
            self._extract_key_lines(
                text,
                limit=8,
                patterns=(
                    "from ",
                    "cmd",
                    "entrypoint",
                    "expose",
                    "image:",
                    "services:",
                    "ports:",
                    "uses:",
                    "runs-on:",
                    "name:",
                    "on:",
                ),
            )
        )
        if not lines:
            return self._extract_generic_excerpt(text)
        return "\n".join(_unique(lines))

    def _extract_source_excerpt(self, path: str, text: str) -> str:
        lines: list[str] = []
        imports = self._extract_import_like_names(text)
        if imports:
            lines.append(f"Imports: {', '.join(imports[:10])}")

        docstring = self._extract_top_comment_or_docstring(text)
        if docstring:
            lines.append(f"Module notes: {docstring}")

        signatures = self._extract_signature_lines(text)
        if signatures:
            lines.append("Key signatures:")
            lines.extend(f"- {signature}" for signature in signatures[:6])

        notable_lines = self._extract_notable_source_lines(text)
        if notable_lines:
            lines.append("Framework clues:")
            lines.extend(f"- {line}" for line in notable_lines[:6])

        if not lines:
            generic = self._extract_generic_excerpt(text)
            if generic:
                return generic
        return "\n".join(lines)

    def _extract_generic_excerpt(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines[:12])

    def _extract_top_comment_or_docstring(self, text: str) -> str:
        docstring_match = re.search(r'^[\s\r\n]*[rubfRUBF]*(["\']{3})(.*?)(?:\1)', text, flags=re.DOTALL)
        if docstring_match:
            cleaned = " ".join(line.strip() for line in docstring_match.group(2).splitlines() if line.strip())
            return cleaned[:240]

        comments: list[str] = []
        for raw_line in text.splitlines()[:16]:
            stripped = raw_line.strip()
            if not stripped:
                if comments:
                    break
                continue
            if stripped.startswith(("#", "//", "/*", "* ", "--")):
                comments.append(stripped.lstrip("#/*- ").strip())
                if len(comments) >= 4:
                    break
            elif comments:
                break
        return " ".join(comment for comment in comments if comment)[:240]

    @staticmethod
    def _extract_signature_lines(text: str) -> list[str]:
        patterns = [
            r"^\s*(?:async\s+def|def|class)\s+[A-Za-z_][A-Za-z0-9_]*.*",
            r"^\s*func\s+[A-Za-z_][A-Za-z0-9_]*.*",
            r"^\s*(?:export\s+)?(?:async\s+)?function\s+[A-Za-z_][A-Za-z0-9_]*.*",
            r"^\s*(?:export\s+)?class\s+[A-Za-z_][A-Za-z0-9_]*.*",
            r"^\s*(?:const|let)\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*(?:async\s*)?\(.*",
        ]
        matches: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or len(stripped) > 160:
                continue
            if any(re.match(pattern, stripped) for pattern in patterns):
                matches.append(stripped)
        return _unique(matches)

    @staticmethod
    def _extract_notable_source_lines(text: str) -> list[str]:
        markers = (
            "fastapi(",
            "flask(",
            "apirouter(",
            "uvicorn.run",
            "@app.",
            "@router.",
            "express(",
            "createapp(",
            "router =",
            "app =",
            "streamlit",
            "springapplication",
            "tokio::main",
        )
        notable: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            lowered = stripped.lower()
            if not stripped or len(stripped) > 180:
                continue
            if any(marker in lowered for marker in markers):
                notable.append(stripped)
        return _unique(notable)

    @staticmethod
    def _extract_key_lines(text: str, *, limit: int, patterns: tuple[str, ...] | None = None) -> list[str]:
        collected: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            lowered = stripped.lower()
            if patterns and not any(pattern in lowered for pattern in patterns):
                continue
            collected.append(stripped)
            if len(collected) >= limit:
                break
        return _unique(collected)

    def _truncate_text(self, text: str, *, max_chars: int, max_lines: int) -> tuple[str, bool]:
        stripped = text.strip()
        if not stripped:
            return "", False
        lines = stripped.splitlines()
        truncated = False
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            truncated = True
        excerpt = "\n".join(lines)
        if len(excerpt) > max_chars:
            excerpt = excerpt[:max_chars].rstrip()
            truncated = True
        return excerpt, truncated

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    @staticmethod
    def _format_languages(languages: dict[str, int]) -> str:
        if not languages:
            return "Unknown"
        total = sum(languages.values()) or 1
        parts = []
        for language, size in sorted(languages.items(), key=lambda item: item[1], reverse=True)[:5]:
            percentage = round((size / total) * 100)
            parts.append(f"{language} ({percentage}%)")
        return ", ".join(parts)

    @staticmethod
    def _map_dependency_to_technology(name: str) -> str | None:
        normalized = name.strip().lower().replace("_", "-")
        if not normalized or normalized in PACKAGE_NAME_IGNORES:
            return None
        return FRAMEWORK_DEPENDENCY_MAP.get(normalized)

    def _technology_label_from_dependency(self, name: str, *, allow_raw: bool) -> str | None:
        mapped = self._map_dependency_to_technology(name)
        if mapped:
            return mapped
        normalized = _clean_dependency_name(name).replace("_", "-")
        if not allow_raw or not self._should_keep_raw_dependency(normalized):
            return None
        return normalized

    @staticmethod
    def _should_keep_raw_dependency(name: str) -> bool:
        if not name or name in PACKAGE_NAME_IGNORES:
            return False
        if name in {"python"} or name.startswith("."):
            return False
        if len(name) < 2 or len(name) > 40:
            return False
        return bool(re.fullmatch(r"[a-z0-9][a-z0-9+._-]*", name))

    def _manifest_context_for_path(self, path: str) -> str:
        parts = [part.lower() for part in PurePosixPath(path).parts]
        basename = PurePosixPath(path).name.lower()
        if any(part in DOC_DIR_HINTS for part in parts):
            return "docs"
        if any(part in TEST_DIR_HINTS for part in parts) or basename.startswith("requirements-test"):
            return "test"
        if "dev" in basename or "lint" in basename or basename.startswith("requirements-"):
            return "dev"
        return "runtime"

    def _dependency_signal_weight(self, path: str, category: str) -> int:
        context = self._manifest_context_for_path(path)
        if context == "runtime":
            return 6 if category == "manifest" else 4
        if context in {"docs", "dev"}:
            return 2
        return 1

    def _extract_manifest_dependency_names(self, path: str, content: str) -> list[str]:
        basename = PurePosixPath(path).name.lower()
        if basename == "package.json":
            return self._parse_package_json_dependencies(content)
        if basename == "pyproject.toml":
            return self._parse_pyproject_dependencies(content)
        if basename == "setup.py":
            return self._parse_setup_py_dependencies(content)
        if basename == "setup.cfg":
            return self._parse_setup_cfg_dependencies(content)
        if basename.startswith("requirements") and basename.endswith(".txt"):
            return self._parse_requirements(content)
        if basename == "go.mod":
            return self._parse_go_mod_dependencies(content)
        if basename == "cargo.toml":
            return self._parse_cargo_dependencies(content)
        if basename == "composer.json":
            return self._parse_composer_dependencies(content)
        if basename == "pom.xml":
            return self._parse_pom_dependencies(content)
        if basename in {"build.gradle", "build.gradle.kts"}:
            return self._parse_gradle_dependencies(content)
        return []

    @staticmethod
    def _parse_package_json_dependencies(content: str) -> list[str]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return []
        deps: list[str] = []
        for key in ("dependencies", "devDependencies", "peerDependencies"):
            values = payload.get(key) or {}
            if isinstance(values, dict):
                deps.extend(values.keys())
        return _unique(deps)

    @staticmethod
    def _parse_package_json_scripts(content: str) -> list[str]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return []
        scripts = payload.get("scripts") or {}
        if not isinstance(scripts, dict):
            return []
        return _unique(scripts.keys())

    @staticmethod
    def _parse_setup_py_dependencies(content: str) -> list[str]:
        try:
            module = ast.parse(content)
        except SyntaxError:
            return []

        scope: dict[str, object] = {}
        collected: list[str] = []
        for node in module.body:
            if isinstance(node, ast.Assign):
                value = RepositoryAnalyzer._resolve_ast_value(node.value, scope)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        scope[target.id] = value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                scope[node.target.id] = RepositoryAnalyzer._resolve_ast_value(node.value, scope)
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and RepositoryAnalyzer._is_setup_call(node.value.func):
                for keyword in node.value.keywords:
                    if keyword.arg == "install_requires":
                        collected.extend(
                            RepositoryAnalyzer._dependency_names_from_ast_value(
                                RepositoryAnalyzer._resolve_ast_value(keyword.value, scope)
                            )
                        )
        return _unique(dep for dep in collected if dep)

    @staticmethod
    def _parse_setup_cfg_dependencies(content: str) -> list[str]:
        parser = configparser.ConfigParser()
        try:
            parser.read_string(content)
        except configparser.Error:
            return []
        raw = parser.get("options", "install_requires", fallback="")
        if not raw:
            return []
        dependencies = [
            _clean_dependency_name(line)
            for line in raw.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        return _unique(dep for dep in dependencies if dep)

    @staticmethod
    def _resolve_ast_value(node: ast.AST | None, scope: dict[str, object]) -> object | None:
        if node is None:
            return None
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return scope.get(node.id)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return [RepositoryAnalyzer._resolve_ast_value(item, scope) for item in node.elts]
        if isinstance(node, ast.Dict):
            return {
                RepositoryAnalyzer._resolve_ast_value(key, scope): RepositoryAnalyzer._resolve_ast_value(value, scope)
                for key, value in zip(node.keys, node.values)
            }
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = RepositoryAnalyzer._resolve_ast_value(node.left, scope)
            right = RepositoryAnalyzer._resolve_ast_value(node.right, scope)
            if isinstance(left, list) and isinstance(right, list):
                return [*left, *right]
            if isinstance(left, str) and isinstance(right, str):
                return left + right
        return None

    @staticmethod
    def _dependency_names_from_ast_value(value: object | None) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            cleaned = _clean_dependency_name(value)
            return [cleaned] if cleaned else []
        if isinstance(value, dict):
            dependencies: list[str] = []
            for nested in value.values():
                dependencies.extend(RepositoryAnalyzer._dependency_names_from_ast_value(nested))
            return _unique(dependencies)
        if isinstance(value, (list, tuple, set)):
            dependencies: list[str] = []
            for item in value:
                dependencies.extend(RepositoryAnalyzer._dependency_names_from_ast_value(item))
            return _unique(dependencies)
        return []

    @staticmethod
    def _is_setup_call(func: ast.AST) -> bool:
        if isinstance(func, ast.Name):
            return func.id == "setup"
        if isinstance(func, ast.Attribute):
            return func.attr == "setup"
        return False

    @staticmethod
    def _parse_pyproject_dependencies(content: str) -> list[str]:
        try:
            payload = tomllib.loads(content)
        except Exception:
            return []

        deps: list[str] = []
        project = payload.get("project") or {}
        for dependency in project.get("dependencies") or []:
            deps.append(_clean_dependency_name(str(dependency)))

        optional_dependencies = project.get("optional-dependencies") or {}
        for values in optional_dependencies.values():
            for dependency in values:
                deps.append(_clean_dependency_name(str(dependency)))

        poetry_dependencies = (((payload.get("tool") or {}).get("poetry") or {}).get("dependencies") or {})
        if isinstance(poetry_dependencies, dict):
            deps.extend(poetry_dependencies.keys())

        return _unique(dep for dep in deps if dep and dep.lower() != "python")

    @staticmethod
    def _parse_requirements(content: str) -> list[str]:
        dependencies: list[str] = []
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            dependencies.append(_clean_dependency_name(line))
        return _unique(dep for dep in dependencies if dep)

    @staticmethod
    def _parse_go_mod_dependencies(content: str) -> list[str]:
        matches = re.findall(r"^\s*require\s+([^\s]+)", content, flags=re.MULTILINE)
        block_matches = re.findall(r"^\s*([^\s]+)\s+v[^\s]+", content, flags=re.MULTILINE)
        return _unique([*matches, *block_matches])

    @staticmethod
    def _parse_cargo_dependencies(content: str) -> list[str]:
        try:
            payload = tomllib.loads(content)
        except Exception:
            return []
        dependencies = payload.get("dependencies") or {}
        if not isinstance(dependencies, dict):
            return []
        return _unique(dependencies.keys())

    @staticmethod
    def _parse_composer_dependencies(content: str) -> list[str]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return []
        deps: list[str] = []
        for key in ("require", "require-dev"):
            values = payload.get(key) or {}
            if isinstance(values, dict):
                deps.extend(values.keys())
        return _unique(deps)

    @staticmethod
    def _parse_pom_dependencies(content: str) -> list[str]:
        return _unique(re.findall(r"<artifactId>([^<]+)</artifactId>", content))

    @staticmethod
    def _parse_gradle_dependencies(content: str) -> list[str]:
        matches = re.findall(r"(?:implementation|api|compileOnly|runtimeOnly)\s+[\"']([^:\"']+):([^:\"']+):", content)
        package_names = [artifact for _, artifact in matches]
        plugins = re.findall(r"id\s+[\"']([^\"']+)[\"']", content)
        return _unique([*package_names, *plugins])

    @staticmethod
    def _parse_dockerfile_images(content: str) -> list[str]:
        return _unique(re.findall(r"^\s*FROM\s+([^\s]+)", content, flags=re.MULTILINE))

    @staticmethod
    def _parse_compose_services(content: str) -> list[str]:
        match = re.search(r"^services:\s*$([\s\S]+)", content, flags=re.MULTILINE)
        if not match:
            return []
        block = match.group(1)
        services = re.findall(r"^\s{2}([A-Za-z0-9_.-]+):\s*$", block, flags=re.MULTILINE)
        return _unique(services)

    @staticmethod
    def _extract_import_like_names(content: str) -> list[str]:
        matches = re.findall(r"(?:from|import)\s+([A-Za-z0-9_./@-]+)", content)
        cleaned = []
        for item in matches:
            token = item.split(".")[0]
            if token.startswith("@") and "/" in token:
                token = token.split("/")[0] + "/" + token.split("/")[1]
            else:
                token = token.split("/")[0]
            cleaned.append(token)
        return _unique(cleaned)


def _clean_dependency_name(value: str) -> str:
    cleaned = re.split(r"[<>=!~\[\];\s]", value, maxsplit=1)[0]
    return cleaned.strip().lower()


def _unique(values) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value).strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result





























