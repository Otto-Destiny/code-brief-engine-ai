from __future__ import annotations

import logging
import re
import time
from typing import Callable

from app.api_models import LLMOutput, SummarizeResponse
from app.config import Settings
from app.domain_models import EvidenceAuditTrail, EvidencePacket, RepoMetadata, SummaryOutcome
from app.exceptions import AppError
from app.logging_utils import log_event
from app.repo_rules import NON_STACK_LANGUAGE_IGNORES
from app.services.cache import TTLCache
from app.services.github_client import GitHubRepositoryClient
from app.services.llm_service import OpenAILLMService
from app.services.repository_analysis import RepositoryAnalyzer

LOGGER = logging.getLogger("app.summarizer")


class RepositorySummarizer:
    def __init__(
        self,
        *,
        settings: Settings,
        github_client: GitHubRepositoryClient,
        analyzer: RepositoryAnalyzer,
        llm_service: OpenAILLMService,
        cache: TTLCache[SummarizeResponse],
    ) -> None:
        self._settings = settings
        self._github_client = github_client
        self._analyzer = analyzer
        self._llm_service = llm_service
        self._cache = cache

    async def summarize(self, github_url: str, request_id: str) -> SummaryOutcome:
        started = time.perf_counter()
        phase_timings: dict[str, float] = {}

        owner, repo = self._github_client.parse_repository_url(github_url)

        phase_started = time.perf_counter()
        snapshot = await self._github_client.fetch_repository_snapshot(owner=owner, repo=repo, request_id=request_id)
        phase_timings["snapshot_fetch_ms"] = _elapsed_ms(phase_started)

        cache_key = self._build_cache_key(snapshot.metadata)
        phase_started = time.perf_counter()
        cached = await self._cache.get(cache_key)
        phase_timings["cache_lookup_ms"] = _elapsed_ms(phase_started)
        if cached is not None:
            phase_timings["total_ms"] = _elapsed_ms(started)
            audit = EvidenceAuditTrail(
                selected_files=[],
                skipped_counts={},
                skipped_examples={},
                technology_evidence={},
                truncation_notes=["Result served from in-memory cache."],
                evidence_chars=0,
                estimated_input_tokens=0,
                tree_source=snapshot.tree_source,
                tree_truncated=snapshot.tree_truncated,
                phase_timings_ms=phase_timings,
                llm_retry_used=False,
                cache_hit=True,
            )
            log_event(
                LOGGER,
                "summary_cache_hit",
                request_id=request_id,
                repo=snapshot.metadata.full_name,
                cache_key=cache_key,
                phase_timings_ms=phase_timings,
                duration_ms=phase_timings["total_ms"],
            )
            return SummaryOutcome(response=cached, audit=audit, cache_key=cache_key)

        phase_started = time.perf_counter()
        analysis = self._analyzer.plan(snapshot)
        phase_timings["analysis_plan_ms"] = _elapsed_ms(phase_started)

        phase_started = time.perf_counter()
        fetched_files = await self._github_client.fetch_selected_files(
            metadata=snapshot.metadata,
            candidates=analysis.candidates,
            request_id=request_id,
        )
        phase_timings["file_fetch_ms"] = _elapsed_ms(phase_started)

        phase_started = time.perf_counter()
        manifest_highlights = self._analyzer.build_manifest_highlights(fetched_files)
        technology_signals = self._analyzer.extract_technologies(snapshot, fetched_files)
        packet = self._analyzer.build_evidence_packet(
            snapshot=snapshot,
            analysis=analysis,
            fetched_files=fetched_files,
            manifest_highlights=manifest_highlights,
            technology_signals=technology_signals,
        )
        phase_timings["evidence_build_ms"] = _elapsed_ms(phase_started)

        response_model, usage, retry_used, final_packet, llm_duration_ms = await self._run_llm_with_retry(
            request_id=request_id,
            repo_full_name=snapshot.metadata.full_name,
            packet=packet,
            retry_factory=lambda: self._analyzer.build_evidence_packet(
                snapshot=snapshot,
                analysis=analysis,
                fetched_files=fetched_files,
                manifest_highlights=manifest_highlights,
                technology_signals=technology_signals,
                reduced=True,
            ),
        )
        phase_timings["llm_ms"] = llm_duration_ms

        phase_started = time.perf_counter()
        response = self._normalize_response(
            llm_output=response_model,
            packet=final_packet,
            metadata=snapshot.metadata,
            technology_signals=technology_signals,
        )
        phase_timings["response_normalization_ms"] = _elapsed_ms(phase_started)

        phase_started = time.perf_counter()
        await self._cache.set(cache_key, response)
        phase_timings["cache_store_ms"] = _elapsed_ms(phase_started)
        phase_timings["total_ms"] = _elapsed_ms(started)

        audit = EvidenceAuditTrail(
            selected_files=[
                {
                    "path": snippet.path,
                    "category": snippet.category,
                    "score": snippet.score,
                    "truncated": snippet.truncated,
                    "reasons": snippet.reasons,
                }
                for snippet in final_packet.selected_files
            ],
            skipped_counts=analysis.skipped_counts,
            skipped_examples=analysis.skipped_examples,
            technology_evidence={signal.name: signal.evidence for signal in technology_signals},
            truncation_notes=final_packet.truncation_notes,
            evidence_chars=final_packet.total_chars,
            estimated_input_tokens=final_packet.estimated_input_tokens,
            tree_source=snapshot.tree_source,
            tree_truncated=snapshot.tree_truncated,
            phase_timings_ms=phase_timings,
            llm_retry_used=retry_used,
            cache_hit=False,
        )
        duration_ms = phase_timings["total_ms"]
        log_event(
            LOGGER,
            "summary_completed",
            request_id=request_id,
            repo=snapshot.metadata.full_name,
            cache_key=cache_key,
            selected_files=len(final_packet.selected_files),
            skipped_counts=analysis.skipped_counts,
            evidence_chars=final_packet.total_chars,
            estimated_input_tokens=final_packet.estimated_input_tokens,
            actual_input_tokens=usage.get("input_tokens", 0),
            actual_output_tokens=usage.get("output_tokens", 0),
            model=self._settings.llm_model,
            retry_used=retry_used,
            duration_ms=duration_ms,
        )
        log_event(
            LOGGER,
            "summary_phase_timings",
            request_id=request_id,
            repo=snapshot.metadata.full_name,
            phase_timings_ms=phase_timings,
            retry_used=retry_used,
        )
        log_event(
            LOGGER,
            "evidence_audit",
            request_id=request_id,
            repo=snapshot.metadata.full_name,
            audit={
                "selected_files": audit.selected_files,
                "skipped_counts": audit.skipped_counts,
                "technology_evidence": audit.technology_evidence,
                "truncation_notes": audit.truncation_notes,
                "evidence_chars": audit.evidence_chars,
                "estimated_input_tokens": audit.estimated_input_tokens,
                "tree_source": audit.tree_source,
                "tree_truncated": audit.tree_truncated,
                "phase_timings_ms": audit.phase_timings_ms,
                "llm_retry_used": audit.llm_retry_used,
            },
        )
        return SummaryOutcome(response=response, audit=audit, cache_key=cache_key)

    async def _run_llm_with_retry(
        self,
        *,
        request_id: str,
        repo_full_name: str,
        packet: EvidencePacket,
        retry_factory: Callable[[], EvidencePacket],
    ) -> tuple[LLMOutput, dict[str, int], bool, EvidencePacket, float]:
        llm_started = time.perf_counter()
        try:
            response, usage = await self._llm_service.summarize_repository(packet, request_id, repo_full_name)
            if self._is_low_quality(response):
                raise AppError(502, "LLM output was too weak to trust.")
            return response, usage, False, packet, _elapsed_ms(llm_started)
        except AppError as exc:
            if not self._should_retry(exc):
                raise
            retry_packet = retry_factory()
            response, retry_usage = await self._llm_service.summarize_repository(retry_packet, request_id, repo_full_name)
            return response, retry_usage, True, retry_packet, _elapsed_ms(llm_started)

    def _normalize_response(
        self,
        *,
        llm_output: LLMOutput,
        packet: EvidencePacket,
        metadata: RepoMetadata,
        technology_signals,
    ) -> SummarizeResponse:
        technologies = self._select_output_technologies(
            llm_technologies=llm_output.technologies,
            technology_signals=technology_signals,
            metadata=metadata,
        )

        summary = llm_output.summary.strip()
        if not summary:
            summary = self._fallback_summary(metadata)
        summary = self._normalize_summary_text(summary, metadata)

        structure = llm_output.structure.strip()
        if not structure:
            structure = self._fallback_structure(packet.directory_outline)

        return SummarizeResponse(
            summary=summary,
            technologies=technologies,
            structure=structure,
        )

    @staticmethod
    def _build_cache_key(metadata: RepoMetadata) -> str:
        freshness = metadata.updated_at or "unknown"
        return f"{metadata.full_name}:{freshness}"

    @staticmethod
    def _is_low_quality(output: LLMOutput) -> bool:
        return len(output.summary.split()) < 8 or len(output.structure.split()) < 8 or len(output.technologies) < 1

    @staticmethod
    def _should_retry(exc: AppError) -> bool:
        retryable_messages = {
            "LLM returned invalid JSON output.",
            "LLM output did not match the required schema.",
            "LLM returned no structured output.",
            "LLM output was too weak to trust.",
        }
        return exc.status_code == 502 and exc.message in retryable_messages

    @staticmethod
    def _select_output_technologies(
        *,
        llm_technologies: list[str],
        technology_signals,
        metadata: RepoMetadata,
    ) -> list[str]:
        ranked = RepositorySummarizer._rank_evidence_backed_technologies(technology_signals, metadata)
        if len(ranked) >= 2:
            return ranked[:6]

        selected: list[str] = []
        seen: set[str] = set()
        ranked_lookup = {RepositorySummarizer._technology_key(name): name for name in ranked}
        for technology in llm_technologies:
            resolved = ranked_lookup.get(RepositorySummarizer._technology_key(technology))
            if resolved is None:
                continue
            key = RepositorySummarizer._technology_key(resolved)
            if key in seen:
                continue
            seen.add(key)
            selected.append(resolved)
        for technology in ranked:
            key = RepositorySummarizer._technology_key(technology)
            if key in seen:
                continue
            seen.add(key)
            selected.append(technology)
        return selected[:6]

    @staticmethod
    def _rank_evidence_backed_technologies(technology_signals, metadata: RepoMetadata) -> list[str]:
        ranked: list[str] = []
        seen: set[str] = set()
        prioritized_languages = [
            language
            for language in metadata.languages.keys()
            if language not in NON_STACK_LANGUAGE_IGNORES
        ]
        signal_lookup = {RepositorySummarizer._technology_key(signal.name): signal.name for signal in technology_signals}

        for language in prioritized_languages:
            key = RepositorySummarizer._technology_key(language)
            resolved = signal_lookup.get(key, language)
            resolved_key = RepositorySummarizer._technology_key(resolved)
            if resolved_key in seen:
                continue
            seen.add(resolved_key)
            ranked.append(resolved)

        for signal in technology_signals:
            key = RepositorySummarizer._technology_key(signal.name)
            if key in seen:
                continue
            seen.add(key)
            ranked.append(signal.name)
        return ranked

    @staticmethod
    def _technology_key(name: str) -> str:
        return "".join(character for character in name.casefold() if character.isalnum())

    @staticmethod
    def _normalize_summary_text(summary: str, metadata: RepoMetadata) -> str:
        cleaned = " ".join(summary.split())
        if not cleaned:
            return cleaned

        cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
        cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)

        for name in RepositorySummarizer._project_name_variants(metadata):
            quoted_pattern = rf"[\"'??](?P<project>{re.escape(name)})[\"'??]"
            cleaned = re.sub(quoted_pattern, lambda match: match.group("project"), cleaned, flags=re.IGNORECASE)
            opener_pattern = rf"^This repository (?:hosts|contains|provides|implements)\s+(?P<project>{re.escape(name)}),\s+(?=(?:a|an|the)\b)"
            cleaned = re.sub(opener_pattern, lambda match: f"{match.group('project')} is ", cleaned, count=1, flags=re.IGNORECASE)

        cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
        return cleaned.strip()

    @staticmethod
    def _project_name_variants(metadata: RepoMetadata) -> list[str]:
        base = metadata.name.strip()
        if not base:
            return []

        candidates = [
            base,
            re.sub(r"[-_]+", " ", base).strip(),
            base.capitalize(),
            re.sub(r"[-_]+", " ", base).strip().title(),
        ]

        variants: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = candidate.strip()
            if not normalized:
                continue
            key = normalized.casefold()
            if key in seen:
                continue
            seen.add(key)
            variants.append(normalized)
        return variants

    @staticmethod
    def _fallback_summary(metadata: RepoMetadata) -> str:
        if metadata.description:
            return f"{metadata.full_name} is a public repository described as: {metadata.description}"
        return f"{metadata.full_name} is a public repository with limited descriptive metadata."

    @staticmethod
    def _fallback_structure(directory_outline: str) -> str:
        return (
            "The project structure follows the visible repository layout. "
            f"Key areas include: {directory_outline.replace(chr(10), '; ')}"
        )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)










