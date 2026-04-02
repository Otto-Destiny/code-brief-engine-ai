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
    def __init__(self, fail_once: bool = False) -> None:
        self.fail_once = fail_once
        self.calls = 0

    async def summarize_repository(self, packet, request_id: str, repo_full_name: str):
        self.calls += 1
        if self.fail_once and self.calls == 1:
            raise AppError(502, "LLM returned invalid JSON output.")
        return (
            LLMOutput(
                summary="Demo is a FastAPI-based repository summarizer that analyzes GitHub repositories and produces concise project briefs.",
                technologies=["Python", "FastAPI", "Pydantic", "Docker"],
                structure="The project centers on the application package, with configuration in pyproject.toml, an entrypoint in app/main.py, and supporting container setup in the Dockerfile.",
            ),
            {"input_tokens": 1000, "output_tokens": 120},
        )


def test_summarizer_service_retries_once_and_logs_cache() -> None:
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
        assert second.audit.cache_hit is True
        assert second.response == first.response

    asyncio.run(run_test())
