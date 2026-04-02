from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app import __version__
from app.api_models import ErrorResponse, SummarizeRequest, SummarizeResponse
from app.config import Settings, get_settings
from app.domain_models import SummaryOutcome
from app.exceptions import AppError
from app.logging_utils import configure_logging, log_event
from app.services.cache import TTLCache
from app.services.github_client import GitHubRepositoryClient
from app.services.llm_service import OpenAILLMService
from app.services.repository_analysis import RepositoryAnalyzer
from app.services.summarizer import RepositorySummarizer

LOGGER = logging.getLogger("app.main")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log_event(
            LOGGER,
            "http_request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        return response



def create_app(summarizer: RepositorySummarizer | None = None) -> FastAPI:
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if summarizer is not None:
            app.state.summarizer = summarizer
            yield
            return

        settings = get_settings()
        http_client = httpx.AsyncClient(
            base_url=settings.github_api_base,
            headers=_build_github_headers(settings),
            timeout=httpx.Timeout(settings.github_timeout_seconds),
        )
        github_client = GitHubRepositoryClient(http_client=http_client, settings=settings)
        analyzer = RepositoryAnalyzer(settings=settings)
        llm_service = OpenAILLMService(settings=settings)
        cache = TTLCache(ttl_seconds=settings.cache_ttl_seconds, max_entries=settings.max_cache_entries)
        app.state.summarizer = RepositorySummarizer(
            settings=settings,
            github_client=github_client,
            analyzer=analyzer,
            llm_service=llm_service,
            cache=cache,
        )
        try:
            yield
        finally:
            await http_client.aclose()

    app = FastAPI(
        title="AI Performance Engineering Repository Summarizer",
        version=__version__,
        lifespan=lifespan,
    )
    if summarizer is not None:
        app.state.summarizer = summarizer
    app.add_middleware(RequestContextMiddleware)
    _register_exception_handlers(app)

    @app.post(
        "/summarize",
        response_model=SummarizeResponse,
        responses={
            400: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            502: {"model": ErrorResponse},
            504: {"model": ErrorResponse},
        },
    )
    async def summarize_repository(payload: SummarizeRequest, request: Request) -> SummarizeResponse:
        service: RepositorySummarizer = request.app.state.summarizer
        outcome: SummaryOutcome = await service.summarize(
            github_url=payload.github_url,
            request_id=request.state.request_id,
        )
        return outcome.response

    return app



def _build_github_headers(settings: Settings) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "repository-summarizer-api",
    }
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    return headers



def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        log_event(
            LOGGER,
            "app_error",
            request_id=getattr(request.state, "request_id", "unknown"),
            status_code=exc.status_code,
            message=exc.message,
            details=exc.details,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(message=exc.message).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        message = "Invalid request body. Provide JSON like {\"github_url\": \"https://github.com/owner/repo\"}."
        log_event(
            LOGGER,
            "validation_error",
            request_id=getattr(request.state, "request_id", "unknown"),
            errors=exc.errors(),
        )
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(message=message).model_dump(),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        log_event(
            LOGGER,
            "unhandled_error",
            request_id=getattr(request.state, "request_id", "unknown"),
            error_type=type(exc).__name__,
            message=str(exc),
        )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(message="An unexpected error occurred.").model_dump(),
        )


app = create_app()

