# Standout-But-Safe Plan for the Repo Summarizer Assignment

## Summary

- Build a FastAPI service that keeps the graded API exact: `POST /summarize` with `github_url` in, `summary` / `technologies` / `structure` out.
- Use OpenAI as the primary provider via `OPENAI_API_KEY`.
- Make the solution stand out through a hybrid pipeline: deterministic repository analysis, ranked evidence selection, strict context budgeting, and one structured LLM synthesis step.
- Optimize for evaluator success and AI-performance credibility: async I/O, no repo cloning, bounded concurrency, retry and fallback behavior, caching, and structured latency and token logs.

## Implementation Changes

### 1. Foundation

- Use FastAPI, Pydantic models, `httpx.AsyncClient`, and a small service-oriented layout with separate GitHub, analysis, and LLM layers.
- Keep setup friction minimal with `requirements.txt`, `python -m venv`, `pip install -r requirements.txt`, and `uvicorn` run instructions.
- Keep the public API boring and exact; put the sophistication behind the endpoint, not in extra required surface area.

### 2. GitHub Acquisition and Repo Understanding

- Normalize GitHub URLs and accept only public repository roots, while tolerating trailing slashes and `.git`; reject malformed or non-repo URLs with `400`.
- Fetch repository metadata and the full recursive tree through the GitHub REST API; do not clone repositories.
- Build a deterministic evidence set that prioritizes `README*`, top-level docs, repo description and topics, primary manifests, runtime and deploy configs, CI files, and likely entrypoints such as `main.*`, `app.*`, `server.*`, `index.*`, `src/*`, and `cmd/*`.
- Generate a compact directory outline covering the top two levels, including source, test, docs, and config areas with simple counts.
- Exclude binaries, media, archives, vendored or generated directories, build output, cache directories, and lockfiles from content inclusion.
- Score files rather than relying only on hardcoded names; increase rank for manifest and framework signal, top-level placement, source relevance, and likely entrypoint patterns.
- Fetch at most 12 files, cap each snippet at 200 lines or 6 KB, and cap the full evidence packet at 45 KB before the LLM call.

### 3. Hybrid Analysis and LLM Synthesis

- Extract candidate technologies deterministically before the LLM step from repository language metadata, manifests, dependency names, Docker and runtime files, and framework fingerprints.
- Build one compact evidence packet containing repo metadata, the ranked directory summary, candidate technologies, and selected file snippets.
- Use one OpenAI structured-output call that must return exact JSON keys: `summary`, `technologies`, and `structure`.
- Constrain `summary` to 2 to 4 sentences that explain the project purpose and likely use case.
- Constrain `technologies` to 3 to 8 deduplicated items backed by visible evidence.
- Constrain `structure` to 2 to 4 sentences that describe the main directories and how the code is organized.
- Validate the LLM response with Pydantic and retry once with a shorter, stricter evidence packet if parsing or quality fails.

### 4. Performance and Reliability Polish

- Use async GitHub requests with explicit timeouts and bounded concurrency of 5 for file-content fetches.
- Add a small in-memory TTL cache keyed by repo identity plus freshness metadata so repeated requests are faster and cheaper.
- Log request ID, repo, selected and skipped file counts, evidence size, model name, GitHub latency, LLM latency, and total duration.
- Record an internal evidence audit trail for each request showing selected files, skipped-file reasons, truncation decisions, and which files supported each reported technology.
- Return clean errors: `400` for invalid URLs, `404` for missing, private, or inaccessible repos, `502` for upstream failures, and `504` for timeouts.
- Never expose raw provider exceptions or stack traces in API responses.

### 5. What Will Make It Exceptional

- Make the pipeline visibly hybrid, not "send README to an LLM." Deterministic technology detection plus ranked evidence selection is the main differentiator.
- Show clear token and latency discipline in code and README: one tree call, at most 12 content calls, one LLM call, and one retry maximum.
- Handle README-poor repos gracefully by falling back to manifests, tree structure, and entrypoints instead of collapsing into weak summaries.
- Make the system explainable in interviews and docs by showing that every summary and technology claim can be traced back to concrete repository evidence.
- Write the README like an evaluator guide: quickstart first, then model choice, then repo-selection strategy, then error handling and tradeoffs.

## Public Interfaces and Types

- `POST /summarize`
- Request body: `{ "github_url": "<public GitHub repo URL>" }`
- Success body: `{ "summary": string, "technologies": string[], "structure": string }`
- Error body: `{ "status": "error", "message": string }`
- Keep evidence visibility internal by default so the graded response shape stays exact; do not return debug evidence unless explicitly added as a separate non-default mode later.
- Define internal types up front: `SummarizeRequest`, `SummarizeResponse`, `ErrorResponse`, `RepoMetadata`, `RepoTreeEntry`, `FileCandidate`, `EvidencePacket`, and `LLMOutput`.

## Test Plan

- Unit test URL normalization and rejection of malformed or unsupported GitHub URLs.
- Unit test tree filtering so binaries, generated directories, lockfiles, and oversized files are skipped correctly.
- Unit test ranking so README, manifests, and entrypoints consistently outrank low-signal files.
- Unit test deterministic technology extraction across Python, Node, Go, and Rust examples.
- Integration test the happy path with mocked GitHub and mocked OpenAI and verify the exact response schema.
- Integration test repos with no README, very large trees, no analyzable text files, private or not-found repos, GitHub failures, and LLM schema-failure retry behavior.
- Manually smoke test at least four public repos with different shapes: a small Python library, a JavaScript app, a backend service, and a larger multi-directory project.
- Verify the README from a clean virtual environment and confirm the provided `curl` command works unchanged.

## Assumptions and Defaults

- Primary provider is OpenAI because that was the chosen direction; use `OPENAI_API_KEY` and make the model configurable via `LLM_MODEL`.
- Default model choice is `gpt-4.1-mini` if available in your account; if not, keep the code model-configurable and lock the exact model name in the README before submission.
- FastAPI is the framework because typed schemas, validation, and generated docs help more here than Flask.
- GitHub access is unauthenticated by default for public repos, but optional `GITHUB_TOKEN` support should be added to reduce rate-limit risk.
- Do not add extra required response fields or rely on extra endpoints for grading.
- If time gets tight, keep the exact endpoint, filtering and ranking logic, strict context budget, retry path, logging, and README quality; trim only optional polish beyond that.


{
  "github_url": "https://github.com/psf/requests"
}

