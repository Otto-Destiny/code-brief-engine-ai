from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Repository Summarizer"
    github_api_base: str = "https://api.github.com"

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    llm_model: str = Field(default="gpt-4.1-mini", alias="LLM_MODEL")
    github_token: str | None = Field(default=None, alias="GITHUB_TOKEN")

    github_timeout_seconds: float = 20.0
    llm_timeout_seconds: float = 45.0
    github_file_concurrency: int = 5

    max_selected_files: int = 12
    max_file_bytes_to_fetch: int = 250_000
    max_file_excerpt_chars: int = 6_000
    max_file_excerpt_lines: int = 200
    retry_file_excerpt_chars: int = 3_000
    retry_file_excerpt_lines: int = 120
    evidence_char_budget: int = 45_000

    cache_ttl_seconds: int = 900
    max_cache_entries: int = 128

    model_config = SettingsConfigDict(extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
