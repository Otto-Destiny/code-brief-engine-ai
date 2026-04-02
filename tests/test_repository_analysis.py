from __future__ import annotations

from app.config import Settings
from app.domain_models import FetchedFile, RepoMetadata, RepoTreeEntry, RepositorySnapshot
from app.services.repository_analysis import RepositoryAnalyzer


def test_repository_analysis_skips_noise_and_prioritizes_diverse_core_evidence() -> None:
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
            RepoTreeEntry(path="project/README.md", type="blob", size=900),
            RepoTreeEntry(path="pyproject.toml", type="blob", size=400),
            RepoTreeEntry(path="Dockerfile", type="blob", size=120),
            RepoTreeEntry(path="app/main.py", type="blob", size=900),
            RepoTreeEntry(path="app/routes.py", type="blob", size=700),
            RepoTreeEntry(path="docs/index.md", type="blob", size=600),
            RepoTreeEntry(path="docs/Makefile", type="blob", size=300),
            RepoTreeEntry(path="tests/testserver/server.py", type="blob", size=500),
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
    assert "docs/Makefile" not in candidate_paths
    assert "tests/testserver/server.py" not in candidate_paths
    assert sum(path.lower().endswith("readme.md") for path in candidate_paths) == 1
    assert analysis.skipped_counts["excluded_directory"] == 1
    assert analysis.skipped_counts["lockfile"] == 1
    assert analysis.skipped_counts["binary_or_media"] == 1


def test_repository_analysis_extracts_technologies_without_noisy_language_labels() -> None:
    analyzer = RepositoryAnalyzer(Settings())
    snapshot = RepositorySnapshot(
        metadata=RepoMetadata(
            owner="acme",
            name="demo",
            full_name="acme/demo",
            html_url="https://github.com/acme/demo",
            default_branch="main",
            languages={"Python": 9000, "Makefile": 1200, "Dockerfile": 1000},
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
    assert "Makefile" not in names
    assert "Dockerfile" not in names


def test_repository_analysis_builds_structured_evidence_packet_with_budget() -> None:
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
            RepoTreeEntry(path="pyproject.toml", type="blob", size=200),
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
            raw_text="# Overview\nA FastAPI service that summarizes repositories.\n\n## Usage\nRun the API locally.\n" + ("A" * 4000),
            size_bytes=4500,
        ),
        FetchedFile(
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
        FetchedFile(
            path="src/demo/app.py",
            category="source",
            score=60,
            reasons=["source"],
            raw_text='from fastapi import FastAPI\n\napp = FastAPI()\n\n\"\"\"Demo application entrypoint.\"\"\"\n\ndef create_app():\n    return app\n' + ("print('x')\n" * 200),
            size_bytes=3000,
        ),
    ]
    technologies = analyzer.extract_technologies(snapshot, fetched_files)
    packet = analyzer.build_evidence_packet(
        snapshot=snapshot,
        analysis=analysis,
        fetched_files=fetched_files,
        manifest_highlights=analyzer.build_manifest_highlights(fetched_files),
        technology_signals=technologies,
    )

    assert packet.total_chars <= Settings().evidence_char_budget
    assert packet.selected_files
    assert packet.estimated_input_tokens > 0
    source_snippet = next(snippet for snippet in packet.selected_files if snippet.path == "src/demo/app.py")
    manifest_snippet = next(snippet for snippet in packet.selected_files if snippet.path == "pyproject.toml")
    assert "Imports: fastapi" in source_snippet.content
    assert "Framework clues:" in source_snippet.content
    assert "Dependencies: fastapi, pydantic, uvicorn" in manifest_snippet.content


def test_repository_analysis_prioritizes_runtime_manifests_over_docs_manifests() -> None:
    analyzer = RepositoryAnalyzer(Settings())
    snapshot = RepositorySnapshot(
        metadata=RepoMetadata(
            owner="psf",
            name="requests",
            full_name="psf/requests",
            html_url="https://github.com/psf/requests",
            default_branch="main",
        ),
        tree_entries=[
            RepoTreeEntry(path="README.md", type="blob", size=1200),
            RepoTreeEntry(path="pyproject.toml", type="blob", size=1600),
            RepoTreeEntry(path="setup.py", type="blob", size=120),
            RepoTreeEntry(path="docs/requirements.txt", type="blob", size=200),
            RepoTreeEntry(path="docs/index.rst", type="blob", size=900),
            RepoTreeEntry(path="src/requests/sessions.py", type="blob", size=4200),
            RepoTreeEntry(path=".github/workflows/tests.yml", type="blob", size=240),
        ],
    )

    analysis = analyzer.plan(snapshot)

    manifest_paths = [candidate.path for candidate in analysis.candidates if candidate.category == "manifest"]
    assert manifest_paths[:2] == ["pyproject.toml", "setup.py"]
    assert "docs/requirements.txt" not in manifest_paths
def test_repository_analysis_extracts_runtime_dependencies_from_setup_py() -> None:
    analyzer = RepositoryAnalyzer(Settings())
    snapshot = RepositorySnapshot(
        metadata=RepoMetadata(
            owner="psf",
            name="requests",
            full_name="psf/requests",
            html_url="https://github.com/psf/requests",
            default_branch="main",
            languages={"Python": 1000},
        ),
        tree_entries=[
            RepoTreeEntry(path="setup.py", type="blob", size=800),
        ],
    )
    fetched_files = [
        FetchedFile(
            path="setup.py",
            category="manifest",
            score=90,
            reasons=["manifest"],
            raw_text='''
requires = [
    "charset_normalizer>=2,<4",
    "idna>=2.5,<4",
    "urllib3>=1.21.1,<3",
    "certifi>=2017.4.17",
]

setup(
    name="requests",
    install_requires=requires,
)
'''.strip(),
            size_bytes=320,
        )
    ]

    technologies = analyzer.extract_technologies(snapshot, fetched_files)
    names = [signal.name for signal in technologies]
    assert "Python" in names
    assert "certifi" in names
    assert "charset-normalizer" in names
    assert "idna" in names
    assert "urllib3" in names


