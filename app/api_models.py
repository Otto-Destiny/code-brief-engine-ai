from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SummarizeRequest(BaseModel):
    github_url: str = Field(..., description="URL of a public GitHub repository.")


class SummarizeResponse(BaseModel):
    summary: str
    technologies: list[str]
    structure: str

    @field_validator("summary", "structure")
    @classmethod
    def validate_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("Field must not be empty.")
        return text

    @field_validator("technologies")
    @classmethod
    def validate_technologies(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            tech = item.strip()
            if not tech:
                continue
            key = tech.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(tech)
        if not normalized:
            raise ValueError("At least one technology is required.")
        return normalized[:8]


class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    message: str


class LLMOutput(BaseModel):
    summary: str
    technologies: list[str]
    structure: str

    @field_validator("summary", "structure")
    @classmethod
    def validate_output_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("Field must not be empty.")
        return text

    @field_validator("technologies")
    @classmethod
    def validate_output_technologies(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            tech = item.strip()
            if not tech:
                continue
            key = tech.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(tech)
        if not normalized:
            raise ValueError("At least one technology is required.")
        return normalized[:8]
