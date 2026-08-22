"""Google Gemini implementation of BaseLLMAdapter.

Wraps the google-genai SDK's async client (`client.aio`): retries with
exponential backoff on rate-limit (429) errors, and enforces a hard
timeout on the whole call. Same shape as anthropic_adapter.py — both
failure modes surface as base.py's shared typed exceptions
(LLMTimeoutError/LLMRateLimitExceededError), not a raw SDK exception.
"""

import asyncio
import time
from typing import Any

from google import genai
from google.genai import errors as genai_errors

from app.config import get_settings
from app.llm_gateway.base import (
    BaseLLMAdapter,
    LLMRateLimitExceededError,
    LLMResponse,
    LLMTimeoutError,
)

RATE_LIMIT_STATUS_CODE = 429


class GeminiAdapter(BaseLLMAdapter):
    """BaseLLMAdapter backed by the Google Gemini API (via Google AI Studio)."""

    def __init__(
        self,
        client: genai.Client | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int | None = None,
        timeout_seconds: float | None = None,
    ):
        """
        Args:
            client: Inject an existing (or mocked) genai.Client — used
                by tests to avoid making real API calls. Defaults to a
                fresh client built from `api_key`/settings.
            api_key: Overrides settings.gemini_api_key when `client`
                isn't given.
            model, max_retries, timeout_seconds: Override the
                corresponding setting; each defaults to
                settings.gemini_model / llm_max_retries /
                llm_timeout_seconds.
        """
        settings = get_settings()
        self._client = client if client is not None else genai.Client(
            api_key=api_key if api_key is not None else settings.gemini_api_key
        )
        self._model = model if model is not None else settings.gemini_model
        self._max_retries = max_retries if max_retries is not None else settings.llm_max_retries
        self._timeout_seconds = (
            timeout_seconds if timeout_seconds is not None else settings.llm_timeout_seconds
        )

    async def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate a completion for `prompt`.

        Raises:
            LLMTimeoutError: if the call doesn't complete within
                `timeout_seconds`.
            LLMRateLimitExceededError: if every retry attempt after a
                429 also gets rate-limited.
        """
        start = time.perf_counter()
        model = kwargs.get("model", self._model)

        last_error: genai_errors.ClientError | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    self._client.aio.models.generate_content(model=model, contents=prompt),
                    timeout=self._timeout_seconds,
                )
            except TimeoutError as exc:
                raise LLMTimeoutError(
                    f"LLM call exceeded {self._timeout_seconds}s timeout"
                ) from exc
            except genai_errors.ClientError as exc:
                if exc.code != RATE_LIMIT_STATUS_CODE:
                    raise
                last_error = exc
                if attempt < self._max_retries:
                    # Exponential backoff: 1s, 2s, 4s, ...
                    await asyncio.sleep(2**attempt)
                    continue
                raise LLMRateLimitExceededError(
                    f"Rate limited after {self._max_retries} retries"
                ) from exc
            else:
                latency_ms = (time.perf_counter() - start) * 1000
                usage = {
                    "input_tokens": response.usage_metadata.prompt_token_count or 0,
                    "output_tokens": response.usage_metadata.candidates_token_count or 0,
                }
                return LLMResponse(
                    text=response.text or "",
                    model=response.model_version or model,
                    usage=usage,
                    latency_ms=latency_ms,
                )

        # Unreachable: the loop above always either returns or raises.
        raise LLMRateLimitExceededError(
            f"Rate limited after {self._max_retries} retries"
        ) from last_error
