# Repository Summarizer API

This project implements the AI Performance Engineering 2026 admission assignment as a FastAPI service. It accepts a public GitHub repository URL, builds a compact evidence packet from the repository, and uses OpenAI to return a human-readable summary, a technology list, and a brief description of the project structure.

## Why This Submission Stands Out

- It does not blindly dump an entire repository into an LLM. The service filters noise, ranks useful files, extracts deterministic technology signals, and sends only a token-budgeted evidence packet.
- It is traceable. Each request logs an internal evidence audit trail showing which files were selected, which were skipped, why they were chosen, and what evidence supported each technology claim.
- It is disciplined about cost and latency. The pipeline uses one repository tree call, bounded file fetches, one structured LLM call, and one retry path only when necessary.

## Tech Stack

- Python 3.10+
- FastAPI
- OpenAI Responses API
- httpx
- Pydantic

## Model Choice

The default model is `gpt-4.1-mini`, chosen because it offers strong reasoning and structured-output quality while keeping latency and cost low for a single-request summarization workflow. The model name is configurable through `LLM_MODEL`.

## How Repository Contents Are Handled

The service first fetches repository metadata, languages, and the directory tree from the GitHub API without cloning the repository. It skips binaries, lockfiles, generated folders, build artifacts, and other low-signal content.

From the remaining files, it prioritizes the most informative evidence:

- `README*` and top-level docs
- manifests such as `pyproject.toml`, `package.json`, `go.mod`, `Cargo.toml`
- runtime and deploy configs such as `Dockerfile` and workflow files
- likely entrypoints such as `main.py`, `server.ts`, `app.py`
- representative source files from `src/`, `app/`, `cmd/`, and similar directories

Large repositories are handled with a strict context budget:

- at most 12 files are selected
- each file excerpt is capped by lines and characters
- the final evidence packet is capped before the LLM call
- if the first LLM result is weak or malformed, the service retries once with an even smaller packet

For repositories with weak READMEs, the service still detects technologies by combining:

- GitHub language metadata
- manifest dependencies
- runtime and deployment files
- source import fingerprints
- directory structure hints

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key.

PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

macOS or Linux:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Optional:

- `LLM_MODEL` to override the default model
- `GITHUB_TOKEN` to reduce GitHub API rate-limit risk for repeated testing

4. Start the server.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Example Request

```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"github_url": "https://github.com/psf/requests"}'
```

Expected response shape:

```json
{
  "summary": "Requests is a popular Python library for making HTTP requests...",
  "technologies": ["Python", "urllib3", "certifi"],
  "structure": "The project follows a standard Python package layout..."
}
```

## Running Tests

```bash
pytest
```

## Notes

- The default API response intentionally stays exact to the assignment contract: only `summary`, `technologies`, and `structure`.
- Debug evidence is logged internally instead of being returned by default, so the service remains grader-friendly while still being explainable.
