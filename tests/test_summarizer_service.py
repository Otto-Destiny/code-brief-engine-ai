from __future__ import annotations

import asyncio

from app.api_models import LLMOutput
from app.config import Settings
from app.domain_models import FetchedFile, RepoMetadata, RepoTreeEntry, RepositorySnapshot
from app.exceptions import AppError
from app.services.cache import TTLCache
from app.services.repository_analysis import RepositoryAnalyzer
from app.services.summarizer import RepositorySummarizer


class FakeGitHubClient:
    def __init__(self) -> None:
        self.snapshot = RepositorySnapshot(
            metadata=RepoMetadata(
                owner="acme",
                name="demo",
                full_name="acme/demo",
                html_url="https://github.com/acme/demo",
                default_branch="main",
                description="Demo API built with FastAPI.",
                languages={"Python": 9000},
                updated_at="2026-03-31T00:00:00Z",
            ),
            tree_entries=[
                RepoTreeEntry(path="README.md", type="blob", size=900),
                RepoTreeEntry(path="pyproject.toml", type="blob", size=240),
                RepoTreeEntry(path="app/main.py", type="blob", size=600),
                RepoTreeEntry(path="Dockerfile", type="blob", size=120),
            ],
        )
        self.file_map = {
            "README.md": FetchedFile(
                path="README.md",
                category="readme",
                score=110,
                reasons=["readme"],
                raw_text="Demo API\nA FastAPI service that summarizes repositories.\n",
                size_bytes=120,
            ),
            "pyproject.toml": FetchedFile(
                path="pyproject.toml",
                category="manifest",
                score=95,
                reasons=["manifest"],
                raw_text="""
[project]
dependencies = [\"fastapi>=0.110\", \"pydantic>=2\", \"uvicorn>=0.30\"]
""".strip(),
                size_bytes=120,
            ),
            "app/main.py": FetchedFile(
                path="app/main.py",
                category="source",
                score=70,
                reasons=["entrypoint", "source"],
                raw_text="from fastapi import FastAPI\napp = FastAPI()\n",
                size_bytes=90,
            ),
            "Dockerfile": FetchedFile(
                path="Dockerfile",
                category="infra",
                score=60,
                reasons=["runtime-config"],
                raw_text="FROM python:3.12-slim\n",
                size_bytes=30,
            ),
        }

    @staticmethod
    def parse_repository_url(github_url: str) -> tuple[str, str]:
        return ("acme", "demo")

    async def fetch_repository_snapshot(self, owner: str, repo: str, request_id: str) -> RepositorySnapshot:
        return self.snapshot

    async def fetch_selected_files(self, metadata: RepoMetadata, candidates, request_id: str) -> list[FetchedFile]:
        return [self.file_map[candidate.path] for candidate in candidates if candidate.path in self.file_map]


class FakeLLMService:
    def __init__(
        self,
        fail_once: bool = False,
        fail_message: str | None = None,
        output: LLMOutput | None = None,
    ) -> None:
        self.fail_once = fail_once
        self.fail_message = fail_message
        self.output = output
        self.calls = 0

    async def summarize_repository(self, packet, request_id: str, repo_full_name: str):
        self.calls += 1
        if self.fail_message:
            raise AppError(502, self.fail_message)
        if self.fail_once and self.calls == 1:
            raise AppError(502, "LLM returned invalid JSON output.")
        return (
            self.output
            or LLMOutput(
                summary="Demo is a FastAPI-based repository summarizer that analyzes GitHub repositories and produces concise project briefs.",
                technologies=["Python", "FastAPI", "Pydantic", "Docker"],
                structure="The project centers on the application package, with configuration in pyproject.toml, an entrypoint in app/main.py, and supporting container setup in the Dockerfile.",
            ),
            {"input_tokens": 1000, "output_tokens": 120},
        )


def test_summarizer_service_retries_once_and_captures_phase_timings() -> None:
    async def run_test() -> None:
        settings = Settings()
        service = RepositorySummarizer(
            settings=settings,
            github_client=FakeGitHubClient(),
            analyzer=RepositoryAnalyzer(settings),
            llm_service=FakeLLMService(fail_once=True),
            cache=TTLCache(ttl_seconds=900, max_entries=16),
        )

        first = await service.summarize("https://github.com/acme/demo", request_id="req-1")
        second = await service.summarize("https://github.com/acme/demo", request_id="req-2")

        assert first.response.technologies[:2] == ["Python", "FastAPI"]
        assert first.audit.llm_retry_used is True
        assert first.audit.selected_files
        assert first.audit.phase_timings_ms["llm_ms"] >= 0
        assert first.audit.phase_timings_ms["evidence_build_ms"] >= 0
        assert second.audit.cache_hit is True
        assert second.audit.phase_timings_ms["total_ms"] >= 0
        assert second.response == first.response

    asyncio.run(run_test())


def test_summarizer_service_does_not_retry_non_retryable_llm_failures() -> None:
    async def run_test() -> None:
        settings = Settings()
        service = RepositorySummarizer(
            settings=settings,
            github_client=FakeGitHubClient(),
            analyzer=RepositoryAnalyzer(settings),
            llm_service=FakeLLMService(fail_message="LLM request failed."),
            cache=TTLCache(ttl_seconds=900, max_entries=16),
        )

        try:
            await service.summarize("https://github.com/acme/demo", request_id="req-3")
        except AppError as exc:
            assert exc.message == "LLM request failed."
        else:  # pragma: no cover - defensive failure branch
            raise AssertionError("Expected summarize() to propagate the non-retryable LLM failure.")

    asyncio.run(run_test())


class RequestsLikeGitHubClient(FakeGitHubClient):
    def __init__(self) -> None:
        self.snapshot = RepositorySnapshot(
            metadata=RepoMetadata(
                owner="psf",
                name="requests",
                full_name="psf/requests",
                html_url="https://github.com/psf/requests",
                default_branch="main",
                description="A simple, yet elegant, HTTP library.",
                languages={"Python": 9000},
                updated_at="2026-04-02T10:42:43Z",
            ),
            tree_entries=[
                RepoTreeEntry(path="README.md", type="blob", size=900),
                RepoTreeEntry(path="pyproject.toml", type="blob", size=1600),
                RepoTreeEntry(path="setup.py", type="blob", size=120),
                RepoTreeEntry(path="docs/requirements.txt", type="blob", size=200),
                RepoTreeEntry(path="src/requests/sessions.py", type="blob", size=1800),
                RepoTreeEntry(path=".github/workflows/run-tests.yml", type="blob", size=200),
            ],
        )
        self.file_map = {
            "README.md": FetchedFile(
                path="README.md",
                category="readme",
                score=110,
                reasons=["readme"],
                raw_text="Requests is an elegant HTTP library for Python.",
                size_bytes=80,
            ),
            "pyproject.toml": FetchedFile(
                path="pyproject.toml",
                category="manifest",
                score=116,
                reasons=["manifest", "top-level"],
                raw_text='''
[project]
dependencies = [
    "charset_normalizer>=2,<4",
    "idna>=2.5,<4",
    "urllib3>=1.21.1,<3",
    "certifi>=2017.4.17",
]
requires-python = ">=3.10"
'''.strip(),
                size_bytes=220,
            ),
            "setup.py": FetchedFile(
                path="setup.py",
                category="manifest",
                score=102,
                reasons=["manifest", "top-level"],
                raw_text='from setuptools import setup\n\nsetup()\n',
                size_bytes=40,
            ),
            "docs/requirements.txt": FetchedFile(
                path="docs/requirements.txt",
                category="manifest",
                score=60,
                reasons=["manifest", "documentation"],
                raw_text="sphinx\npytest\n",
                size_bytes=40,
            ),
            ".github/workflows/run-tests.yml": FetchedFile(
                path=".github/workflows/run-tests.yml",
                category="infra",
                score=70,
                reasons=["ci", "runtime-config"],
                raw_text="name: tests\nuses: actions/checkout@v4\n",
                size_bytes=60,
            ),
            "src/requests/sessions.py": FetchedFile(
                path="src/requests/sessions.py",
                category="source",
                score=58,
                reasons=["source"],
                raw_text='''
"""Session management for Requests."""

from urllib3 import PoolManager

class Session:
    pass
'''.strip(),
                size_bytes=96,
            ),
        }


def test_summarizer_service_prefers_evidence_backed_technologies_for_library_repos() -> None:
    async def run_test() -> None:
        settings = Settings()
        service = RepositorySummarizer(
            settings=settings,
            github_client=RequestsLikeGitHubClient(),
            analyzer=RepositoryAnalyzer(settings),
            llm_service=FakeLLMService(
                output=LLMOutput(
                    summary="Requests is a Python HTTP library.",
                    technologies=["Python", "Sphinx", "GitHub Actions", "Makefile", "TLS/SSL", "HTTP"],
                    structure="The library is organized into source, docs, and workflows.",
                )
            ),
            cache=TTLCache(ttl_seconds=900, max_entries=16),
        )

        result = await service.summarize("https://github.com/psf/requests", request_id="req-requests")

        assert result.response.technologies[:5] == ["Python", "certifi", "charset-normalizer", "idna", "urllib3"]
        assert "Makefile" not in result.response.technologies
        assert "HTTP" not in result.response.technologies
        assert "TLS/SSL" not in result.response.technologies

    asyncio.run(run_test())


def test_summarizer_service_normalizes_summary_style_artifacts() -> None:
    async def run_test() -> None:
        settings = Settings()
        service = RepositorySummarizer(
            settings=settings,
            github_client=RequestsLikeGitHubClient(),
            analyzer=RepositoryAnalyzer(settings),
            llm_service=FakeLLMService(
                output=LLMOutput(
                    summary='This repository hosts "Requests", a simple yet elegant HTTP library for Python. It uses `urllib3` internally and supports **sessions**.',
                    technologies=["Python", "urllib3"],
                    structure="The library is organized into source, docs, and workflows.",
                )
            ),
            cache=TTLCache(ttl_seconds=900, max_entries=16),
        )

        result = await service.summarize("https://github.com/psf/requests", request_id="req-summary-style")

        assert result.response.summary == (
            "Requests is a simple yet elegant HTTP library for Python. "
            "It uses urllib3 internally and supports sessions."
        )

    asyncio.run(run_test())

