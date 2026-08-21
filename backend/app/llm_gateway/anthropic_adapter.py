"""Anthropic implementation of BaseLLMAdapter — the default target LLM.

Wraps the anthropic SDK's async client: retries with exponential
backoff on rate-limit (429) errors, and enforces a hard timeout on the
whole call rather than trusting the SDK's own default. Both failure
modes surface as this module's own typed exceptions
(LLMTimeoutError/LLMRateLimitExceededError) instead of a raw SDK
exception, so callers don't need to import anthropic's exception types
to handle them.
"""

import asyncio
import time
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

from app.config import get_settings
from app.llm_gateway.base import BaseLLMAdapter, LLMResponse

DEFAULT_MAX_TOKENS = 1024


class LLMTimeoutError(Exception):
    """Raised when a call to the target LLM exceeds the configured timeout."""


class LLMRateLimitExceededError(Exception):
    """Raised when retries on repeated rate-limit (429) responses are exhausted."""


class AnthropicAdapter(BaseLLMAdapter):
    """BaseLLMAdapter backed by the Anthropic API."""

    def __init__(
        self,
        client: AsyncAnthropic | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_retries: int | None = None,
        timeout_seconds: float | None = None,
    ):
        """
        Args:
            client: Inject an existing (or mocked) AsyncAnthropic client
                — used by tests to avoid making real API calls. Defaults
                to a fresh client built from `api_key`/settings.
            api_key: Overrides settings.anthropic_api_key when `client`
                isn't given.
            model, max_retries, timeout_seconds: Override the
                corresponding setting; each defaults to
                settings.anthropic_model / llm_max_retries /
                llm_timeout_seconds.
        """
        settings = get_settings()
        self._client = client if client is not None else AsyncAnthropic(
            api_key=api_key if api_key is not None else settings.anthropic_api_key
        )
        self._model = model if model is not None else settings.anthropic_model
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
        max_tokens = kwargs.get("max_tokens", DEFAULT_MAX_TOKENS)

        last_error: anthropic.RateLimitError | None = None

        for attempt in range(self._max_retries + 1):
            try:
                message = await asyncio.wait_for(
                    self._client.messages.create(
                        model=model,
                        max_tokens=max_tokens,
                        messages=[{"role": "user", "content": prompt}],
                    ),
                    timeout=self._timeout_seconds,
                )
            except TimeoutError as exc:
                raise LLMTimeoutError(
                    f"LLM call exceeded {self._timeout_seconds}s timeout"
                ) from exc
            except anthropic.RateLimitError as exc:
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
                text = "".join(
                    block.text for block in message.content if hasattr(block, "text")
                )
                usage = {
                    "input_tokens": message.usage.input_tokens,
                    "output_tokens": message.usage.output_tokens,
                }
                return LLMResponse(
                    text=text, model=message.model, usage=usage, latency_ms=latency_ms
                )

        # Unreachable: the loop above always either returns or raises.
        raise LLMRateLimitExceededError(
            f"Rate limited after {self._max_retries} retries"
        ) from last_error


_adapter: BaseLLMAdapter | None = None


def get_llm_adapter() -> BaseLLMAdapter:
    """Process-wide default adapter (lazy singleton) — a plain function
    (not @lru_cache), same pattern as get_context_buffer()/
    get_policy_engine(), so it stays overridable (e.g. pipeline.py's
    injectable `llm_adapter` param, or a future FastAPI dependency)."""
    global _adapter
    if _adapter is None:
        _adapter = AnthropicAdapter()
    return _adapter
