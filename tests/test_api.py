from __future__ import annotations

from fastapi.testclient import TestClient

from app.api_models import SummarizeResponse
from app.domain_models import EvidenceAuditTrail, SummaryOutcome
from app.exceptions import AppError
from app.main import create_app


class StubSummarizer:
    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail

    async def summarize(self, github_url: str, request_id: str) -> SummaryOutcome:
        if self.should_fail:
            raise AppError(400, "github_url must be a public GitHub repository URL.")
        response = SummarizeResponse(
            summary="Requests is a Python HTTP library used to make and manage web requests.",
            technologies=["Python", "urllib3", "certifi"],
            structure="The repository is organized around the main package, tests, and project metadata files.",
        )
        audit = EvidenceAuditTrail(
            selected_files=[],
            skipped_counts={},
            skipped_examples={},
            technology_evidence={},
            truncation_notes=[],
            evidence_chars=0,
            estimated_input_tokens=0,
            tree_source="git_tree",
            tree_truncated=False,
        )
        return SummaryOutcome(response=response, audit=audit, cache_key="cache-key")


def test_post_summarize_returns_expected_shape() -> None:
    client = TestClient(create_app(summarizer=StubSummarizer()))
    response = client.post(
        "/summarize",
        json={"github_url": "https://github.com/psf/requests"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "summary": "Requests is a Python HTTP library used to make and manage web requests.",
        "technologies": ["Python", "urllib3", "certifi"],
        "structure": "The repository is organized around the main package, tests, and project metadata files.",
    }
    assert response.headers["X-Request-ID"]


def test_post_summarize_uses_error_shape_for_app_errors() -> None:
    client = TestClient(create_app(summarizer=StubSummarizer(should_fail=True)))
    response = client.post(
        "/summarize",
        json={"github_url": "not-a-url"},
    )
    assert response.status_code == 400
    assert response.json()["status"] == "error"


def test_post_summarize_uses_error_shape_for_validation_errors() -> None:
    client = TestClient(create_app(summarizer=StubSummarizer()))
    response = client.post("/summarize", json={})
    assert response.status_code == 422
    assert response.json() == {
        "status": "error",
        "message": "Invalid request body. Provide JSON like {\"github_url\": \"https://github.com/owner/repo\"}.",
    }
