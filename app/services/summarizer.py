from __future__ import annotations

import logging
import time

from app.api_models import LLMOutput, SummarizeResponse
from app.config import Settings
from app.domain_models import EvidenceAuditTrail, EvidencePacket, RepoMetadata, SummaryOutcome
from app.exceptions import AppError
from app.logging_utils import log_event
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
        owner, repo = self._github_client.parse_repository_url(github_url)
        snapshot = await self._github_client.fetch_repository_snapshot(owner=owner, repo=repo, request_id=request_id)
        cache_key = self._build_cache_key(snapshot.metadata)
        cached = await self._cache.get(cache_key)
        if cached is not None:
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
                llm_retry_used=False,
                cache_hit=True,
            )
            log_event(
                LOGGER,
                "summary_cache_hit",
                request_id=request_id,
                repo=snapshot.metadata.full_name,
                cache_key=cache_key,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            return SummaryOutcome(response=cached, audit=audit, cache_key=cache_key)

        analysis = self._analyzer.plan(snapshot)
        fetched_files = await self._github_client.fetch_selected_files(
            metadata=snapshot.metadata,
            candidates=analysis.candidates,
            request_id=request_id,
        )
        manifest_highlights = self._analyzer.build_manifest_highlights(fetched_files)
        technology_signals = self._analyzer.extract_technologies(snapshot, fetched_files)
        packet = self._analyzer.build_evidence_packet(
            snapshot=snapshot,
            analysis=analysis,
            fetched_files=fetched_files,
            manifest_highlights=manifest_highlights,
            technology_signals=technology_signals,
        )

        response_model, usage, retry_used = await self._run_llm_with_retry(
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
        response = self._normalize_response(
            llm_output=response_model,
            packet=packet,
            metadata=snapshot.metadata,
            technology_signals=technology_signals,
        )

        await self._cache.set(cache_key, response)
        audit = EvidenceAuditTrail(
            selected_files=[
                {
                    "path": snippet.path,
                    "category": snippet.category,
                    "score": snippet.score,
                    "truncated": snippet.truncated,
                    "reasons": snippet.reasons,
                }
                for snippet in packet.selected_files
            ],
            skipped_counts=analysis.skipped_counts,
            skipped_examples=analysis.skipped_examples,
            technology_evidence={signal.name: signal.evidence for signal in technology_signals},
            truncation_notes=packet.truncation_notes,
            evidence_chars=packet.total_chars,
            estimated_input_tokens=packet.estimated_input_tokens,
            tree_source=snapshot.tree_source,
            tree_truncated=snapshot.tree_truncated,
            llm_retry_used=retry_used,
            cache_hit=False,
        )
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        log_event(
            LOGGER,
            "summary_completed",
            request_id=request_id,
            repo=snapshot.metadata.full_name,
            cache_key=cache_key,
            selected_files=len(packet.selected_files),
            skipped_counts=analysis.skipped_counts,
            evidence_chars=packet.total_chars,
            estimated_input_tokens=packet.estimated_input_tokens,
            actual_input_tokens=usage.get("input_tokens", 0),
            actual_output_tokens=usage.get("output_tokens", 0),
            model=self._settings.llm_model,
            retry_used=retry_used,
            duration_ms=duration_ms,
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
        retry_factory,
    ) -> tuple[LLMOutput, dict[str, int], bool]:
        try:
            response, usage = await self._llm_service.summarize_repository(packet, request_id, repo_full_name)
            if self._is_low_quality(response):
                raise AppError(502, "LLM output was too weak to trust.")
            return response, usage, False
        except AppError:
            retry_packet = retry_factory()
            response, retry_usage = await self._llm_service.summarize_repository(retry_packet, request_id, repo_full_name)
            return response, retry_usage, True

    def _normalize_response(
        self,
        *,
        llm_output: LLMOutput,
        packet: EvidencePacket,
        metadata: RepoMetadata,
        technology_signals,
    ) -> SummarizeResponse:
        technologies = list(llm_output.technologies)
        if not technologies:
            technologies = [signal.name for signal in technology_signals[:5]]
        if not technologies and metadata.languages:
            technologies = list(metadata.languages.keys())[:3]

        summary = llm_output.summary.strip()
        if not summary:
            summary = self._fallback_summary(metadata)

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
