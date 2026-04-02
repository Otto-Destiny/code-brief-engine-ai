from __future__ import annotations

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
    DOC_DIR_HINTS,
    ENTRYPOINT_NAMES,
    EXCLUDED_DIRECTORIES,
    FRAMEWORK_DEPENDENCY_MAP,
    LOCKFILES,
    MANIFEST_FILE_NAMES,
    PACKAGE_NAME_IGNORES,
    README_NAMES,
    SOURCE_DIR_HINTS,
    TEST_DIR_HINTS,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
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
            basename = PurePosixPath(file.path).name.lower()
            if basename == "package.json":
                deps = self._parse_package_json_dependencies(file.raw_text)
                if deps:
                    highlights.append(f"{file.path}: dependencies include {', '.join(deps[:6])}")
            elif basename == "pyproject.toml":
                deps = self._parse_pyproject_dependencies(file.raw_text)
                if deps:
                    highlights.append(f"{file.path}: runtime dependencies include {', '.join(deps[:6])}")
            elif basename.startswith("requirements") and basename.endswith(".txt"):
                deps = self._parse_requirements(file.raw_text)
                if deps:
                    highlights.append(f"{file.path}: requirements include {', '.join(deps[:6])}")
            elif basename == "go.mod":
                deps = self._parse_go_mod_dependencies(file.raw_text)
                if deps:
                    highlights.append(f"{file.path}: Go modules include {', '.join(deps[:6])}")
            elif basename == "cargo.toml":
                deps = self._parse_cargo_dependencies(file.raw_text)
                if deps:
                    highlights.append(f"{file.path}: Rust crates include {', '.join(deps[:6])}")
            elif basename == "pom.xml":
                deps = self._parse_pom_dependencies(file.raw_text)
                if deps:
                    highlights.append(f"{file.path}: Maven artifacts include {', '.join(deps[:6])}")
            elif basename in {"build.gradle", "build.gradle.kts"}:
                deps = self._parse_gradle_dependencies(file.raw_text)
                if deps:
                    highlights.append(f"{file.path}: Gradle dependencies include {', '.join(deps[:6])}")
            elif basename == "dockerfile":
                images = self._parse_dockerfile_images(file.raw_text)
                if images:
                    highlights.append(f"{file.path}: base images include {', '.join(images[:3])}")
        return highlights[:6]

    def extract_technologies(
        self,
        snapshot: RepositorySnapshot,
        fetched_files: list[FetchedFile],
    ) -> list[TechnologySignal]:
        collector: dict[str, dict[str, object]] = {}

        def add_signal(name: str, confidence: int, evidence: str) -> None:
            record = collector.setdefault(name, {"confidence": 0, "evidence": set()})
            record["confidence"] = int(record["confidence"]) + confidence
            record["evidence"].add(evidence)

        languages = sorted(snapshot.metadata.languages.items(), key=lambda item: item[1], reverse=True)
        for index, (language, _) in enumerate(languages[:3]):
            add_signal(language, 5 - index, "repo_languages")

        for entry in snapshot.tree_entries:
            path_lower = entry.path.lower()
            if path_lower.endswith(".tf") or path_lower.endswith(".tfvars"):
                add_signal("Terraform", 3, entry.path)
            if path_lower.startswith(".github/workflows/"):
                add_signal("GitHub Actions", 4, entry.path)
            if "dockerfile" in path_lower or "docker-compose" in path_lower:
                add_signal("Docker", 4, entry.path)
            if "k8s/" in path_lower or "kubernetes/" in path_lower or path_lower.endswith("kustomization.yaml"):
                add_signal("Kubernetes", 3, entry.path)
            if path_lower.endswith("chart.yaml") or "/charts/" in path_lower:
                add_signal("Helm", 3, entry.path)

        for file in fetched_files:
            basename = PurePosixPath(file.path).name.lower()
            path_lower = file.path.lower()
            if basename == "package.json":
                for dependency in self._parse_package_json_dependencies(file.raw_text):
                    tech = self._map_dependency_to_technology(dependency)
                    if tech:
                        add_signal(tech, 5, file.path)
            elif basename == "pyproject.toml":
                for dependency in self._parse_pyproject_dependencies(file.raw_text):
                    tech = self._map_dependency_to_technology(dependency)
                    if tech:
                        add_signal(tech, 5, file.path)
            elif basename.startswith("requirements") and basename.endswith(".txt"):
                for dependency in self._parse_requirements(file.raw_text):
                    tech = self._map_dependency_to_technology(dependency)
                    if tech:
                        add_signal(tech, 4, file.path)
            elif basename == "cargo.toml":
                for dependency in self._parse_cargo_dependencies(file.raw_text):
                    tech = self._map_dependency_to_technology(dependency)
                    if tech:
                        add_signal(tech, 5, file.path)
            elif basename == "go.mod":
                for dependency in self._parse_go_mod_dependencies(file.raw_text):
                    tech = self._map_dependency_to_technology(dependency)
                    if tech:
                        add_signal(tech, 4, file.path)
            elif basename == "composer.json":
                for dependency in self._parse_composer_dependencies(file.raw_text):
                    tech = self._map_dependency_to_technology(dependency)
                    if tech:
                        add_signal(tech, 4, file.path)
            elif basename in {"pom.xml", "build.gradle", "build.gradle.kts"}:
                parser = self._parse_pom_dependencies if basename == "pom.xml" else self._parse_gradle_dependencies
                for dependency in parser(file.raw_text):
                    tech = self._map_dependency_to_technology(dependency)
                    if tech:
                        add_signal(tech, 4, file.path)

            for import_name in self._extract_import_like_names(file.raw_text):
                tech = self._map_dependency_to_technology(import_name)
                if tech:
                    add_signal(tech, 2, file.path)
            if path_lower.endswith("dockerfile"):
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
            ordered_files = ordered_files[: min(6, len(ordered_files))]

        for fetched in ordered_files:
            excerpt, truncated = self._truncate_text(fetched.raw_text, max_chars=max_chars, max_lines=max_lines)
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
            f"- {signal.name} (confidence {signal.confidence}; evidence: {', '.join(signal.evidence[:4])})"
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
                        f"category={snippet.category}; score={snippet.score:.1f}; reasons={reasons}",
                        "```text",
                        snippet.content,
                        "```",
                    ]
                )
            )

        snippets_text = "\n\n".join(snippet_sections) if snippet_sections else "No text file excerpts were selected."
        topics = ", ".join(snapshot.metadata.topics) if snapshot.metadata.topics else "None"
        description = snapshot.metadata.description or "No repository description provided."

        return "\n".join(
            [
                "Repository overview",
                f"- Full name: {snapshot.metadata.full_name}",
                f"- URL: {snapshot.metadata.html_url}",
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
        basename = PurePosixPath(entry.path).name.lower()
        depth = max(len(PurePosixPath(entry.path).parts) - 1, 0)

        if self._is_readme(entry.path):
            score += 110
            reasons.append("readme")
        if basename in MANIFEST_FILE_NAMES:
            score += 92
            reasons.append("manifest")
        if entry.path.lower().startswith("docs/") or basename in {"mkdocs.yml", "readthedocs.yml"}:
            score += 56
            reasons.append("documentation")
        if entry.path.lower().startswith(".github/workflows/"):
            score += 50
            reasons.append("ci")
        if basename in CONFIG_NAMES or "dockerfile" in basename:
            score += 52
            reasons.append("runtime-config")
        stem = PurePosixPath(entry.path).stem.lower()
        if stem in ENTRYPOINT_NAMES:
            score += 45
            reasons.append("entrypoint")
        if self._looks_like_source(entry.path):
            score += 28
            reasons.append("source")
        if depth == 0:
            score += 18
            reasons.append("top-level")
        score += max(0, 12 - (depth * 2))
        if entry.size:
            score -= min(entry.size / 8192, 18)
        return FileCandidate(
            path=entry.path,
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
            "manifest": 3,
            "infra": 2,
            "docs": 1,
            "source": 5,
            "other": 2,
        }
        selected: list[FileCandidate] = []
        selected_paths: set[str] = set()
        category_counts: dict[str, int] = defaultdict(int)

        for candidate in candidates:
            limit = quotas.get(candidate.category, 1)
            if category_counts[candidate.category] >= limit:
                continue
            selected.append(candidate)
            selected_paths.add(candidate.path)
            category_counts[candidate.category] += 1
            if len(selected) >= self._settings.max_selected_files:
                return selected

        for candidate in candidates:
            if candidate.path in selected_paths:
                continue
            selected.append(candidate)
            selected_paths.add(candidate.path)
            if len(selected) >= self._settings.max_selected_files:
                break
        return selected

    def _categorize(self, path: str) -> str:
        lower_path = path.lower()
        basename = PurePosixPath(path).name.lower()
        if self._is_readme(path):
            return "readme"
        if basename in MANIFEST_FILE_NAMES:
            return "manifest"
        if lower_path.startswith(".github/workflows/") or basename in CONFIG_NAMES or "dockerfile" in basename:
            return "infra"
        if any(part in DOC_DIR_HINTS for part in lower_path.split("/")):
            return "docs"
        if any(part in SOURCE_DIR_HINTS for part in lower_path.split("/")) or PurePosixPath(path).stem.lower() in ENTRYPOINT_NAMES:
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
