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
from google.genai import types as genai_types

from app.config import get_settings
from app.llm_gateway.base import (
    BaseLLMAdapter,
    LLMDailyQuotaExceededError,
    LLMRateLimitExceededError,
    LLMResponse,
    LLMTimeoutError,
)

RATE_LIMIT_STATUS_CODE = 429

# Google returns per-day and per-minute exhaustion under the same 429,
# distinguishable only by the quota id in the error body (e.g.
# "GenerateRequestsPerDayPerProjectPerModel-FreeTier" vs the per-minute
# equivalent). Matched on "PerDay" rather than the full id so a tier
# rename doesn't silently turn daily-quota errors back into retryable
# ones.
_DAILY_QUOTA_MARKER = "perday"


def _violations(exc: genai_errors.ClientError) -> list[dict]:
    """Best-effort extraction of the QuotaFailure violations block.

    Defensive throughout: this parses a third-party error payload whose
    exact shape isn't guaranteed, and a parsing slip here must not mask
    the rate-limit error itself.
    """
    details = exc.details if isinstance(getattr(exc, "details", None), dict) else {}
    error = details.get("error") if isinstance(details.get("error"), dict) else {}
    found: list[dict] = []
    for detail in error.get("details", []) or []:
        if isinstance(detail, dict) and isinstance(detail.get("violations"), list):
            found.extend(v for v in detail["violations"] if isinstance(v, dict))
    return found


def _is_daily_quota_error(exc: genai_errors.ClientError) -> bool:
    """True when the 429 is a per-day quota, which backoff cannot clear."""
    for violation in _violations(exc):
        if _DAILY_QUOTA_MARKER in str(violation.get("quotaId", "")).lower():
            return True
    return False


def _quota_message(exc: genai_errors.ClientError) -> str:
    for violation in _violations(exc):
        quota_id = str(violation.get("quotaId", ""))
        if _DAILY_QUOTA_MARKER in quota_id.lower():
            limit = violation.get("quotaValue")
            model = (violation.get("quotaDimensions") or {}).get("model", "the model")
            limit_text = f" (limit: {limit}/day)" if limit else ""
            return f"Daily quota exhausted for {model}{limit_text}"
    return "Daily quota exhausted"


class GeminiAdapter(BaseLLMAdapter):
    """BaseLLMAdapter backed by the Google Gemini API (via Google AI Studio)."""

    def __init__(
        self,
        client: genai.Client | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int | None = None,
        timeout_seconds: float | None = None,
        max_output_tokens: int | None = None,
        thinking_level: str | None = None,
    ):
        """
        Args:
            client: Inject an existing (or mocked) genai.Client — used
                by tests to avoid making real API calls. Defaults to a
                fresh client built from `api_key`/settings.
            api_key: Overrides settings.gemini_api_key when `client`
                isn't given.
            model, max_retries, timeout_seconds, max_output_tokens,
                thinking_level: Override the corresponding setting; each
                defaults to settings.gemini_model / llm_max_retries /
                llm_timeout_seconds / llm_max_output_tokens /
                gemini_thinking_level.
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
        self._max_output_tokens = (
            max_output_tokens if max_output_tokens is not None else settings.llm_max_output_tokens
        )
        self._thinking_level = (
            thinking_level if thinking_level is not None else settings.gemini_thinking_level
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
        config = genai_types.GenerateContentConfig(
            max_output_tokens=self._max_output_tokens,
            thinking_config=genai_types.ThinkingConfig(thinking_level=self._thinking_level),
        )

        last_error: genai_errors.ClientError | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    self._client.aio.models.generate_content(
                        model=model, contents=prompt, config=config
                    ),
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
                # A *per-day* quota exhaustion is not something backoff
                # can clear — the window is hours wide, so retrying just
                # burns more of an already-spent quota and delays the
                # error the caller needs to see. Observed live on
                # Gemini's free tier (limit: 20 requests/day/model),
                # where the old unconditional backoff turned an
                # immediate, actionable "daily quota gone" into ~7s of
                # pointless waiting plus 3 extra billed attempts.
                if _is_daily_quota_error(exc):
                    raise LLMDailyQuotaExceededError(_quota_message(exc)) from exc
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
