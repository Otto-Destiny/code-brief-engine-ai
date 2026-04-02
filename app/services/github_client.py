from __future__ import annotations

import asyncio
import base64
import logging
from collections import deque
from urllib.parse import urlparse

import httpx

from app.config import Settings
from app.domain_models import FetchedFile, FileCandidate, RepoMetadata, RepoTreeEntry, RepositorySnapshot
from app.exceptions import AppError
from app.logging_utils import log_event
from app.repo_rules import EXCLUDED_DIRECTORIES

LOGGER = logging.getLogger("app.github")


class GitHubRepositoryClient:
    def __init__(self, http_client: httpx.AsyncClient, settings: Settings) -> None:
        self._client = http_client
        self._settings = settings
        self._file_semaphore = asyncio.Semaphore(settings.github_file_concurrency)

    @staticmethod
    def parse_repository_url(github_url: str) -> tuple[str, str]:
        raw_url = github_url.strip()
        if not raw_url:
            raise AppError(400, "github_url must not be empty.")

        parsed = urlparse(raw_url)
        if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() not in {"github.com", "www.github.com"}:
            raise AppError(400, "github_url must be a public GitHub repository URL.")

        segments = [segment for segment in parsed.path.split("/") if segment]
        if len(segments) != 2:
            raise AppError(
                400,
                "github_url must point to the repository root, like https://github.com/owner/repo.",
            )

        owner, repo = segments
        if repo.endswith(".git"):
            repo = repo[:-4]
        if not owner or not repo:
            raise AppError(400, "github_url must point to the repository root, like https://github.com/owner/repo.")
        return owner, repo

    async def fetch_repository_snapshot(self, owner: str, repo: str, request_id: str) -> RepositorySnapshot:
        repo_payload = await self._get_json(f"/repos/{owner}/{repo}")
        languages_task = asyncio.create_task(self._safe_fetch_languages(owner, repo))
        tree_entries, tree_truncated, tree_source = await self._fetch_tree_entries(
            owner=owner,
            repo=repo,
            branch=repo_payload["default_branch"],
            request_id=request_id,
        )
        languages = await languages_task

        metadata = RepoMetadata(
            owner=owner,
            name=repo_payload["name"],
            full_name=repo_payload["full_name"],
            html_url=repo_payload["html_url"],
            default_branch=repo_payload["default_branch"],
            description=repo_payload.get("description"),
            topics=repo_payload.get("topics") or [],
            languages=languages,
            updated_at=repo_payload.get("updated_at"),
        )
        return RepositorySnapshot(
            metadata=metadata,
            tree_entries=tree_entries,
            tree_truncated=tree_truncated,
            tree_source=tree_source,
        )

    async def fetch_selected_files(
        self,
        metadata: RepoMetadata,
        candidates: list[FileCandidate],
        request_id: str,
    ) -> list[FetchedFile]:
        tasks = [
            asyncio.create_task(self._fetch_candidate_file(metadata, candidate, request_id))
            for candidate in candidates
        ]
        fetched = await asyncio.gather(*tasks)
        return [item for item in fetched if item is not None]

    async def _fetch_candidate_file(
        self,
        metadata: RepoMetadata,
        candidate: FileCandidate,
        request_id: str,
    ) -> FetchedFile | None:
        if candidate.size_bytes and candidate.size_bytes > self._settings.max_file_bytes_to_fetch:
            return None

        async with self._file_semaphore:
            payload = await self._get_json(
                f"/repos/{metadata.owner}/{metadata.name}/contents/{candidate.path}",
                params={"ref": metadata.default_branch},
            )

        if isinstance(payload, list):
            return None

        encoded_content = payload.get("content")
        encoding = payload.get("encoding")
        if not encoded_content or encoding != "base64":
            log_event(
                LOGGER,
                "github_file_skipped",
                request_id=request_id,
                path=candidate.path,
                reason="missing_base64_content",
            )
            return None

        try:
            decoded = base64.b64decode(encoded_content, validate=False).decode("utf-8")
        except UnicodeDecodeError:
            log_event(
                LOGGER,
                "github_file_skipped",
                request_id=request_id,
                path=candidate.path,
                reason="non_utf8_content",
            )
            return None

        if "\x00" in decoded:
            return None

        return FetchedFile(
            path=candidate.path,
            category=candidate.category,
            score=candidate.score,
            reasons=candidate.reasons,
            raw_text=decoded,
            size_bytes=payload.get("size") or candidate.size_bytes,
        )

    async def _safe_fetch_languages(self, owner: str, repo: str) -> dict[str, int]:
        try:
            payload = await self._get_json(f"/repos/{owner}/{repo}/languages")
        except AppError as exc:
            log_event(
                LOGGER,
                "github_languages_unavailable",
                owner=owner,
                repo=repo,
                status_code=exc.status_code,
                message=exc.message,
            )
            return {}
        if not isinstance(payload, dict):
            return {}
        return {str(language): int(size) for language, size in payload.items()}

    async def _fetch_tree_entries(
        self,
        owner: str,
        repo: str,
        branch: str,
        request_id: str,
    ) -> tuple[list[RepoTreeEntry], bool, str]:
        try:
            payload = await self._get_json(
                f"/repos/{owner}/{repo}/git/trees/{branch}",
                params={"recursive": 1},
            )
        except AppError as exc:
            if exc.status_code == 409:
                return [], False, "empty_repo"
            raise

        entries = [
            RepoTreeEntry(path=item["path"], type=item["type"], size=item.get("size"))
            for item in payload.get("tree", [])
            if item.get("type") in {"blob", "tree"}
        ]
        if payload.get("truncated"):
            log_event(
                LOGGER,
                "github_tree_truncated",
                request_id=request_id,
                owner=owner,
                repo=repo,
                branch=branch,
            )
            fallback_entries = await self._fetch_bounded_contents_tree(owner=owner, repo=repo, branch=branch)
            return fallback_entries, True, "contents_bfs"
        return entries, False, "git_tree"

    async def _fetch_bounded_contents_tree(self, owner: str, repo: str, branch: str) -> list[RepoTreeEntry]:
        entries: list[RepoTreeEntry] = []
        seen: set[str] = set()
        queue: deque[tuple[str, int]] = deque([("", 0)])
        while queue and len(entries) < 2_000:
            current_path, current_depth = queue.popleft()
            endpoint = f"/repos/{owner}/{repo}/contents"
            if current_path:
                endpoint = f"{endpoint}/{current_path}"
            payload = await self._get_json(endpoint, params={"ref": branch})
            items = payload if isinstance(payload, list) else [payload]
            for item in items:
                item_path = item.get("path") or ""
                if not item_path or item_path in seen:
                    continue
                seen.add(item_path)
                item_type = item.get("type")
                if item_type == "dir":
                    entries.append(RepoTreeEntry(path=item_path, type="tree"))
                    directory_name = item.get("name", "")
                    if current_depth < 2 and directory_name not in EXCLUDED_DIRECTORIES:
                        queue.append((item_path, current_depth + 1))
                elif item_type == "file":
                    entries.append(RepoTreeEntry(path=item_path, type="blob", size=item.get("size")))
        return entries

    async def _get_json(self, path: str, params: dict[str, object] | None = None) -> dict | list:
        try:
            response = await self._client.get(path, params=params)
        except httpx.TimeoutException as exc:
            raise AppError(504, "GitHub API timed out.") from exc
        except httpx.HTTPError as exc:
            raise AppError(502, "GitHub API request failed.") from exc

        if response.status_code == 404:
            raise AppError(404, "Repository not found, private, or inaccessible.")
        if response.status_code == 403:
            remaining = response.headers.get("X-RateLimit-Remaining")
            if remaining == "0":
                raise AppError(502, "GitHub API rate limit exceeded. Try again later or set GITHUB_TOKEN.")
            raise AppError(502, "GitHub API rejected the request.")
        if response.status_code == 409:
            raise AppError(409, "Repository data is unavailable.")
        if response.status_code >= 500:
            raise AppError(502, "GitHub API returned an upstream error.")
        if response.status_code >= 400:
            raise AppError(502, "GitHub API returned an unexpected error.")
        return response.json()
