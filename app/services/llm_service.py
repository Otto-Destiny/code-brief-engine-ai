from __future__ import annotations

import asyncio
import json
from typing import Any

from openai import BadRequestError, OpenAI

from app.api_models import LLMOutput
from app.config import Settings
from app.domain_models import EvidencePacket
from app.exceptions import AppError


class OpenAILLMService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(api_key=settings.openai_api_key, timeout=settings.llm_timeout_seconds) if settings.openai_api_key else None

    async def summarize_repository(
        self,
        packet: EvidencePacket,
        request_id: str,
        repo_full_name: str,
    ) -> tuple[LLMOutput, dict[str, int]]:
        if self._client is None:
            raise AppError(500, "OPENAI_API_KEY is not configured.")
        return await asyncio.to_thread(self._summarize_sync, packet, request_id, repo_full_name)

    def _summarize_sync(
        self,
        packet: EvidencePacket,
        request_id: str,
        repo_full_name: str,
    ) -> tuple[LLMOutput, dict[str, int]]:
        try:
            response = self._create_response(
                packet=packet,
                request_id=request_id,
                repo_full_name=repo_full_name,
                use_json_schema=True,
            )
        except BadRequestError as exc:
            if self._should_retry_with_json_object(exc):
                try:
                    response = self._create_response(
                        packet=packet,
                        request_id=request_id,
                        repo_full_name=repo_full_name,
                        use_json_schema=False,
                    )
                except Exception as retry_exc:
                    raise AppError(
                        502,
                        "LLM request failed.",
                        details=self._extract_error_details(retry_exc),
                    ) from retry_exc
            else:
                raise AppError(
                    502,
                    "LLM request failed.",
                    details=self._extract_error_details(exc),
                ) from exc
        except Exception as exc:  # pragma: no cover
            raise AppError(502, "LLM request failed.", details=self._extract_error_details(exc)) from exc

        output_text = getattr(response, "output_text", None)
        if not output_text:
            raise AppError(502, "LLM returned no structured output.")

        try:
            payload = json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise AppError(502, "LLM returned invalid JSON output.") from exc

        try:
            parsed = LLMOutput.model_validate(payload)
        except Exception as exc:
            raise AppError(502, "LLM output did not match the required schema.") from exc

        return parsed, self._extract_usage(response)

    def _create_response(
        self,
        *,
        packet: EvidencePacket,
        request_id: str,
        repo_full_name: str,
        use_json_schema: bool,
    ):
        text_config: dict[str, object]
        if use_json_schema:
            text_config = {
                "format": {
                    "type": "json_schema",
                    "name": "repository_summary",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["summary", "technologies", "structure"],
                        "properties": {
                            "summary": {"type": "string"},
                            "technologies": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                                "maxItems": 8,
                            },
                            "structure": {"type": "string"},
                        },
                    },
                }
            }
        else:
            text_config = {"format": {"type": "json_object"}}

        return self._client.responses.create(
            model=self._settings.llm_model,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are a precise software analyst. Use only the repository evidence you are given. "
                                "Do not guess missing facts. If the evidence is ambiguous, stay conservative and prefer broader labels. "
                                "Return concise JSON only."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Analyze this repository evidence and return JSON with keys summary, technologies, and structure. "
                                "Requirements: summary must be 2-4 sentences; technologies must contain 3-8 deduplicated technologies, "
                                "languages, frameworks, libraries, or infrastructure components directly supported by the evidence; "
                                "structure must be 2-4 sentences describing how the repository is organized. "
                                "Write plain natural prose, not markdown or code formatting. "
                                "Do not use backticks, markdown emphasis, or quotation marks around the project name in the summary.\n\n"
                                f"{packet.text}"
                            ),
                        }
                    ],
                },
            ],
            text=text_config,
            max_output_tokens=400,
            metadata={
                "request_id": request_id[:64],
                "repository": repo_full_name[:64],
            },
        )

    @staticmethod
    def _should_retry_with_json_object(exc: Exception) -> bool:
        details = OpenAILLMService._extract_error_details(exc)
        blob = json.dumps(details, sort_keys=True).casefold()
        triggers = [
            "json_schema",
            "structured output",
            "response format",
            "unsupported_parameter",
            "unknown parameter",
            "text.format",
            "schema",
            "verbosity",
        ]
        return any(trigger in blob for trigger in triggers)

    @staticmethod
    def _extract_error_details(exc: Exception) -> dict[str, object]:
        details: dict[str, object] = {
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        body = getattr(exc, "body", None)
        if body is not None:
            details["body"] = body
        code = getattr(exc, "code", None)
        if code is not None:
            details["code"] = code
        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            details["status_code"] = status_code
        return details

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        if isinstance(usage, dict):
            return {
                "input_tokens": int(usage.get("input_tokens", 0)),
                "output_tokens": int(usage.get("output_tokens", 0)),
            }
        return {
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        }



