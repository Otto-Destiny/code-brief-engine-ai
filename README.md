# Repository Summarizer API

This project is a FastAPI service that accepts a public GitHub repository URL and returns a human-readable project summary, a list of evidence-backed technologies, and a short description of the repository structure.

## Prerequisites

- Python 3.10 or newer
- Internet access for GitHub API and OpenAI API calls
- A valid `OPENAI_API_KEY`

## Quickstart

Run these commands from the project root.

### Windows PowerShell

1. Create a virtual environment:

```powershell
python -m venv .venv
```

2. Install dependencies.

If PowerShell activation works on your machine:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the virtual environment directly without activating it:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

3. Set the API key:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

Optional:

```powershell
$env:GITHUB_TOKEN="your_github_token_here"
$env:LLM_MODEL="gpt-4.1-mini"
```

4. Start the server:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### macOS or Linux

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Set the API key:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Optional:

```bash
export GITHUB_TOKEN="your_github_token_here"
export LLM_MODEL="gpt-4.1-mini"
```

4. Start the server:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Verify the API

- Exposes `POST /summarize`
- Accepts a request body like `{"github_url": "https://github.com/psf/requests"}`
- Fetches repository metadata and selected contents from the GitHub API
- Builds a compact evidence packet instead of sending the whole repository to the LLM
- Uses the OpenAI Responses API to generate structured output

The default response shape is:

```json
{
  "summary": "Requests is a popular Python library for making HTTP requests...",
  "technologies": ["Python", "urllib3", "certifi"],
  "structure": "The project follows a standard Python package layout..."
}
```

On error, the service returns:

```json
{
  "status": "error",
  "message": "Description of what went wrong"
}
```

## API Usage

### Request

`POST /summarize`

Request body:

```json
{
  "github_url": "https://github.com/psf/requests"
}
```

### Bash or macOS/Linux `curl`

```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"github_url": "https://github.com/psf/requests"}'
```

### PowerShell

Using `Invoke-RestMethod` is the most reliable option in PowerShell:

```powershell
$body = @{ github_url = "https://github.com/psf/requests" } | ConvertTo-Json -Compress
Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/summarize" `
  -ContentType "application/json" `
  -Body $body
```

### Interactive docs

FastAPI docs are available at:

- `http://localhost:8000/docs`

## Design Highlights

- It does not dump an entire repository into the LLM. The service filters noise, ranks useful files, extracts deterministic technology signals, and sends only a bounded evidence packet.
- It is traceable. Runtime logs include an internal evidence audit showing which files were selected, which were skipped, why they were chosen, and what evidence supported each technology claim.
- It is disciplined about performance. The pipeline uses one repository tree call, bounded file fetches, one structured LLM call, and one retry path only when necessary.
- It is careful when evidence is weak. The system prefers evidence-backed technologies over speculative or noisy labels.

## Stack

- Python 3.10+
- FastAPI
- OpenAI Responses API
- httpx
- Pydantic

## Model Choice

The default model is `gpt-4.1-mini`. I chose it because it provides strong structured-output reliability and good summarization quality at a reasonable latency and cost for a single-request repository analysis workflow.

The model name can be changed with `LLM_MODEL`.

## How Repository Contents Are Handled

The service uses the GitHub API and does not clone repositories locally.

It first fetches:

- repository metadata
- the default branch
- GitHub language metadata
- the repository tree

It then filters out low-value or unsafe content such as:

- binaries and media files
- lock files
- generated folders
- vendored dependencies
- build artifacts
- cache directories
- oversized files
- non-UTF8 content

From the remaining candidates, it prioritizes high-signal files such as:

- `README*`
- top-level manifests like `pyproject.toml`, `package.json`, `go.mod`, `Cargo.toml`
- runtime and deployment configs like `Dockerfile`
- GitHub workflow files when they add operational context
- likely entrypoints like `main.py`, `server.ts`, `app.py`
- representative source files from `src/`, `app/`, `cmd/`, and similar directories

## Context Management Strategy

Large repositories are handled with strict limits before the LLM call:

- at most 8 files are selected by default
- source files are compressed into imports, signatures, framework clues, and short module notes
- manifests and infra files are distilled into dependency, script, service, and runtime highlights
- the final evidence packet is capped before it is sent to the model
- if the first LLM result is malformed or too weak, the service retries once with a smaller packet

This keeps the service usable on larger repositories and avoids sending unnecessary tokens.

## How Technologies Are Inferred

For repositories with weak or incomplete READMEs, the service still detects technologies by combining:

- GitHub language metadata
- manifest dependencies
- runtime and deployment files
- source import fingerprints
- source signatures and framework clues
- directory structure hints

The final `technologies` output is biased toward evidence-backed items rather than whatever the LLM happens to guess.

## Output Style

The service asks the model to return:

- a short human-readable summary of what the project does
- a deduplicated technology list
- a short description of the repository structure

The summarizer also lightly normalizes the final summary so that it reads like plain prose rather than markdown or code-style text.

## Running Tests

```bash
pytest
```

## Runtime Logs

The public API response stays minimal, but the service logs useful internal diagnostics:

- `summary_completed`
- `summary_phase_timings`
- `evidence_audit`

These logs make it easy to inspect:

- selected and skipped files
- evidence size and token estimates
- actual input and output token usage
- per-phase latency
- evidence supporting each detected technology

## Troubleshooting

### PowerShell blocks virtual environment activation

Use the environment directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### The API returns a provider quota error

Make sure the API key is valid and has active API quota or billing enabled.

### GitHub rate limits occur during repeated testing

Set `GITHUB_TOKEN` to reduce GitHub API rate-limit risk for public repositories.

### A response looks cached during repeated local testing

The service keeps a small in-memory cache keyed by repository freshness. Restart the server if you want to guarantee a fresh uncached local run after code changes.

## Notes

- The endpoint contract matches the documented request and response format.
- API keys are loaded from environment variables and are never hardcoded.
- The implementation uses OpenAI as the LLM provider.
- The README is written so the project can be started from a clean machine with Python installed.
