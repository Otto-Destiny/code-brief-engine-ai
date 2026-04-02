from __future__ import annotations

from dataclasses import dataclass, field

from app.api_models import SummarizeResponse


@dataclass(slots=True)
class RepoMetadata:
    owner: str
    name: str
    full_name: str
    html_url: str
    default_branch: str
    description: str | None = None
    topics: list[str] = field(default_factory=list)
    languages: dict[str, int] = field(default_factory=dict)
    updated_at: str | None = None


@dataclass(slots=True)
class RepoTreeEntry:
    path: str
    type: str
    size: int | None = None


@dataclass(slots=True)
class RepositorySnapshot:
    metadata: RepoMetadata
    tree_entries: list[RepoTreeEntry]
    tree_truncated: bool = False
    tree_source: str = "git_tree"


@dataclass(slots=True)
class FileCandidate:
    path: str
    category: str
    score: float
    reasons: list[str]
    size_bytes: int | None = None
    depth: int = 0


@dataclass(slots=True)
class FetchedFile:
    path: str
    category: str
    score: float
    reasons: list[str]
    raw_text: str
    size_bytes: int | None = None


@dataclass(slots=True)
class FileSnippet:
    path: str
    category: str
    score: float
    reasons: list[str]
    content: str
    size_bytes: int | None
    content_chars: int
    content_lines: int
    truncated: bool = False


@dataclass(slots=True)
class TechnologySignal:
    name: str
    confidence: int
    evidence: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RepositoryAnalysis:
    directory_outline: str
    candidates: list[FileCandidate]
    skipped_counts: dict[str, int] = field(default_factory=dict)
    skipped_examples: dict[str, list[str]] = field(default_factory=dict)
    tree_warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvidencePacket:
    text: str
    estimated_input_tokens: int
    selected_files: list[FileSnippet]
    manifest_highlights: list[str]
    candidate_technologies: list[TechnologySignal]
    directory_outline: str
    total_chars: int
    truncation_notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvidenceAuditTrail:
    selected_files: list[dict[str, object]]
    skipped_counts: dict[str, int]
    skipped_examples: dict[str, list[str]]
    technology_evidence: dict[str, list[str]]
    truncation_notes: list[str]
    evidence_chars: int
    estimated_input_tokens: int
    tree_source: str
    tree_truncated: bool
    llm_retry_used: bool = False
    cache_hit: bool = False


@dataclass(slots=True)
class SummaryOutcome:
    response: SummarizeResponse
    audit: EvidenceAuditTrail
    cache_key: str
