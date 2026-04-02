from __future__ import annotations

from app.config import Settings
from app.domain_models import FetchedFile, RepoMetadata, RepoTreeEntry, RepositorySnapshot
from app.services.repository_analysis import RepositoryAnalyzer


def test_repository_analysis_skips_noise_and_prioritizes_good_evidence() -> None:
    analyzer = RepositoryAnalyzer(Settings())
    snapshot = RepositorySnapshot(
        metadata=RepoMetadata(
            owner="acme",
            name="demo",
            full_name="acme/demo",
            html_url="https://github.com/acme/demo",
            default_branch="main",
        ),
        tree_entries=[
            RepoTreeEntry(path="README.md", type="blob", size=1200),
            RepoTreeEntry(path="pyproject.toml", type="blob", size=400),
            RepoTreeEntry(path="app/main.py", type="blob", size=900),
            RepoTreeEntry(path="tests/test_main.py", type="blob", size=500),
            RepoTreeEntry(path="node_modules/react/index.js", type="blob", size=300),
            RepoTreeEntry(path="package-lock.json", type="blob", size=1000),
            RepoTreeEntry(path="assets/logo.png", type="blob", size=3000),
        ],
    )

    analysis = analyzer.plan(snapshot)

    candidate_paths = [candidate.path for candidate in analysis.candidates]
    assert "README.md" in candidate_paths
    assert "pyproject.toml" in candidate_paths
    assert "app/main.py" in candidate_paths
    assert analysis.skipped_counts["excluded_directory"] == 1
    assert analysis.skipped_counts["lockfile"] == 1
    assert analysis.skipped_counts["binary_or_media"] == 1


def test_repository_analysis_extracts_technologies_from_manifests_and_source() -> None:
    analyzer = RepositoryAnalyzer(Settings())
    snapshot = RepositorySnapshot(
        metadata=RepoMetadata(
            owner="acme",
            name="demo",
            full_name="acme/demo",
            html_url="https://github.com/acme/demo",
            default_branch="main",
            languages={"Python": 9000, "Dockerfile": 1000},
        ),
        tree_entries=[
            RepoTreeEntry(path="pyproject.toml", type="blob", size=300),
            RepoTreeEntry(path="app/main.py", type="blob", size=600),
            RepoTreeEntry(path="Dockerfile", type="blob", size=120),
            RepoTreeEntry(path=".github/workflows/test.yml", type="blob", size=200),
        ],
    )
    fetched_files = [
        FetchedFile(
            path="pyproject.toml",
            category="manifest",
            score=90,
            reasons=["manifest"],
            raw_text="""
[project]
dependencies = [\"fastapi>=0.1\", \"pydantic>=2\", \"uvicorn>=0.30\"]
""".strip(),
            size_bytes=120,
        ),
        FetchedFile(
            path="app/main.py",
            category="source",
            score=70,
            reasons=["entrypoint"],
            raw_text="from fastapi import FastAPI\napp = FastAPI()\n",
            size_bytes=50,
        ),
        FetchedFile(
            path="Dockerfile",
            category="infra",
            score=65,
            reasons=["runtime-config"],
            raw_text="FROM python:3.12-slim\n",
            size_bytes=40,
        ),
    ]

    technologies = analyzer.extract_technologies(snapshot, fetched_files)
    names = [signal.name for signal in technologies]
    assert "Python" in names
    assert "FastAPI" in names
    assert "Pydantic" in names
    assert "Docker" in names
    assert "GitHub Actions" in names


def test_repository_analysis_builds_evidence_packet_with_budget() -> None:
    analyzer = RepositoryAnalyzer(Settings())
    snapshot = RepositorySnapshot(
        metadata=RepoMetadata(
            owner="acme",
            name="demo",
            full_name="acme/demo",
            html_url="https://github.com/acme/demo",
            default_branch="main",
            description="Demo service",
        ),
        tree_entries=[
            RepoTreeEntry(path="README.md", type="blob", size=1200),
            RepoTreeEntry(path="src/demo/app.py", type="blob", size=900),
        ],
    )
    analysis = analyzer.plan(snapshot)
    fetched_files = [
        FetchedFile(
            path="README.md",
            category="readme",
            score=100,
            reasons=["readme"],
            raw_text="Overview\n" + ("A" * 8000),
            size_bytes=8001,
        ),
        FetchedFile(
            path="src/demo/app.py",
            category="source",
            score=60,
            reasons=["source"],
            raw_text="from fastapi import FastAPI\n" + ("print('x')\n" * 500),
            size_bytes=5000,
        ),
    ]
    technologies = analyzer.extract_technologies(snapshot, fetched_files)
    packet = analyzer.build_evidence_packet(
        snapshot=snapshot,
        analysis=analysis,
        fetched_files=fetched_files,
        manifest_highlights=[],
        technology_signals=technologies,
    )
    assert packet.total_chars <= Settings().evidence_char_budget
    assert packet.selected_files
    assert packet.estimated_input_tokens > 0
