from __future__ import annotations

import pytest

from app.exceptions import AppError
from app.services.github_client import GitHubRepositoryClient


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://github.com/psf/requests", ("psf", "requests")),
        ("https://github.com/psf/requests/", ("psf", "requests")),
        ("https://github.com/psf/requests.git", ("psf", "requests")),
    ],
)
def test_parse_repository_url_accepts_repo_root_variants(url: str, expected: tuple[str, str]) -> None:
    assert GitHubRepositoryClient.parse_repository_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "",
        "https://gitlab.com/psf/requests",
        "https://github.com/psf/requests/tree/main",
        "https://github.com/psf",
    ],
)
def test_parse_repository_url_rejects_non_root_urls(url: str) -> None:
    with pytest.raises(AppError):
        GitHubRepositoryClient.parse_repository_url(url)
