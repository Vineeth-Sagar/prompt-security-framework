from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import errors as genai_errors

from app.llm_gateway.base import (
    LLMDailyQuotaExceededError,
    LLMRateLimitExceededError,
    LLMTimeoutError,
)
from app.llm_gateway.gemini_adapter import GeminiAdapter


def _make_client_error(code: int) -> genai_errors.ClientError:
    return genai_errors.ClientError(code, {"error": {"message": f"error {code}"}})


def _make_mock_response(text: str = "Hello!", model: str = "gemini-3.6-flash") -> MagicMock:
    response = MagicMock()
    response.text = text
    response.model_version = model
    response.usage_metadata.prompt_token_count = 10
    response.usage_metadata.candidates_token_count = 5
    return response


def _make_mock_client(generate_content) -> MagicMock:
    client = MagicMock()
    client.aio.models.generate_content = generate_content
    return client


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", AsyncMock())


@pytest.mark.asyncio
async def test_successful_generate_returns_normalized_response():
    client = _make_mock_client(AsyncMock(return_value=_make_mock_response()))

    adapter = GeminiAdapter(client=client, model="gemini-3.6-flash")
    response = await adapter.generate("hello")

    assert response.text == "Hello!"
    assert response.model == "gemini-3.6-flash"
    assert response.usage == {"input_tokens": 10, "output_tokens": 5}
    assert response.latency_ms >= 0


@pytest.mark.asyncio
async def test_no_real_api_call_is_ever_made():
    mock_generate = AsyncMock(return_value=_make_mock_response())
    client = _make_mock_client(mock_generate)

    adapter = GeminiAdapter(client=client)
    await adapter.generate("hello")

    mock_generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_retries_on_rate_limit_then_succeeds():
    call_count = 0

    async def flaky_generate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise _make_client_error(429)
        return _make_mock_response()

    client = _make_mock_client(flaky_generate)
    adapter = GeminiAdapter(client=client, max_retries=3)

    response = await adapter.generate("hello")

    assert call_count == 3
    assert response.text == "Hello!"


@pytest.mark.asyncio
async def test_exhausting_retries_raises_typed_rate_limit_error():
    async def always_rate_limited(*args, **kwargs):
        raise _make_client_error(429)

    client = _make_mock_client(always_rate_limited)
    adapter = GeminiAdapter(client=client, max_retries=2)

    with pytest.raises(LLMRateLimitExceededError, match="after 2 retries"):
        await adapter.generate("hello")


@pytest.mark.asyncio
async def test_non_rate_limit_client_error_is_not_retried_and_propagates():
    call_count = 0

    async def not_found(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise _make_client_error(404)

    client = _make_mock_client(not_found)
    adapter = GeminiAdapter(client=client, max_retries=3)

    with pytest.raises(genai_errors.ClientError):
        await adapter.generate("hello")

    assert call_count == 1  # not retried — only 429 is treated as rate limiting


@pytest.mark.asyncio
async def test_retry_count_respects_max_retries_setting():
    call_count = 0

    async def always_rate_limited(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise _make_client_error(429)

    client = _make_mock_client(always_rate_limited)
    adapter = GeminiAdapter(client=client, max_retries=4)

    with pytest.raises(LLMRateLimitExceededError):
        await adapter.generate("hello")

    assert call_count == 5  # initial attempt + 4 retries


@pytest.mark.asyncio
async def test_timeout_raises_clean_typed_exception():
    import asyncio

    async def hangs_forever(*args, **kwargs):
        return await asyncio.get_event_loop().create_future()

    client = _make_mock_client(hangs_forever)
    adapter = GeminiAdapter(client=client, timeout_seconds=0.05, max_retries=0)

    with pytest.raises(LLMTimeoutError, match="0.05s timeout"):
        await adapter.generate("hello")


@pytest.mark.asyncio
async def test_timeout_does_not_retry():
    import asyncio

    call_count = 0

    async def hangs_forever(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return await asyncio.get_event_loop().create_future()

    client = _make_mock_client(hangs_forever)
    adapter = GeminiAdapter(client=client, timeout_seconds=0.05, max_retries=3)

    with pytest.raises(LLMTimeoutError):
        await adapter.generate("hello")

    assert call_count == 1


@pytest.mark.asyncio
async def test_model_can_be_overridden_per_call():
    mock_generate = AsyncMock(return_value=_make_mock_response())
    client = _make_mock_client(mock_generate)

    adapter = GeminiAdapter(client=client, model="gemini-3.6-flash")
    await adapter.generate("hello", model="gemini-3.6-pro")

    _, kwargs = mock_generate.call_args
    assert kwargs["model"] == "gemini-3.6-pro"


# --- per-day quota exhaustion is NOT a retryable rate limit ---
#
# Observed live on Gemini's free tier (limit: 20 requests/day/model).
# Google signals per-day and per-minute exhaustion with the same 429,
# distinguishable only by the quota id inside the error body. Treating
# the daily one as retryable meant ~7s of exponential backoff that could
# not possibly succeed, three extra requests against an already-spent
# quota, and finally a message telling the user to "wait a few seconds
# and submit again" — advice that is wrong by hours.


def _make_daily_quota_error() -> genai_errors.ClientError:
    return genai_errors.ClientError(
        429,
        {
            "error": {
                "code": 429,
                "message": "You exceeded your current quota",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {
                                "quotaMetric": (
                                    "generativelanguage.googleapis.com/"
                                    "generate_content_free_tier_requests"
                                ),
                                "quotaId": ("GenerateRequestsPerDayPerProjectPerModel-FreeTier"),
                                "quotaDimensions": {"model": "gemini-3.6-flash"},
                                "quotaValue": "20",
                            }
                        ],
                    }
                ],
            }
        },
    )


def _make_per_minute_quota_error() -> genai_errors.ClientError:
    return genai_errors.ClientError(
        429,
        {
            "error": {
                "code": 429,
                "message": "You exceeded your current quota",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {
                                "quotaId": ("GenerateRequestsPerMinutePerProjectPerModel-FreeTier"),
                                "quotaValue": "10",
                            }
                        ],
                    }
                ],
            }
        },
    )


@pytest.mark.asyncio
async def test_daily_quota_error_raises_distinct_type_without_retrying():
    mock_generate = AsyncMock(side_effect=_make_daily_quota_error())
    adapter = GeminiAdapter(client=_make_mock_client(mock_generate), max_retries=3)

    with pytest.raises(LLMDailyQuotaExceededError):
        await adapter.generate("hello")

    # The whole point: no backoff loop against a quota that can't clear.
    assert mock_generate.await_count == 1


@pytest.mark.asyncio
async def test_daily_quota_message_names_the_model_and_limit():
    adapter = GeminiAdapter(
        client=_make_mock_client(AsyncMock(side_effect=_make_daily_quota_error()))
    )

    with pytest.raises(LLMDailyQuotaExceededError) as excinfo:
        await adapter.generate("hello")

    message = str(excinfo.value)
    assert "gemini-3.6-flash" in message
    assert "20" in message


@pytest.mark.asyncio
async def test_per_minute_quota_error_is_still_retried():
    # The complement, so the fix can't silently turn every 429 into a
    # non-retryable one: a per-minute throttle genuinely does clear.
    mock_generate = AsyncMock(side_effect=_make_per_minute_quota_error())
    adapter = GeminiAdapter(client=_make_mock_client(mock_generate), max_retries=2)

    with pytest.raises(LLMRateLimitExceededError):
        await adapter.generate("hello")

    assert mock_generate.await_count == 3


@pytest.mark.asyncio
async def test_malformed_429_body_falls_back_to_retryable_rate_limit():
    # Defensive: this parses a third-party payload whose shape isn't
    # guaranteed. A parsing miss must degrade to the old retry
    # behaviour, never crash the adapter.
    mock_generate = AsyncMock(side_effect=_make_client_error(429))
    adapter = GeminiAdapter(client=_make_mock_client(mock_generate), max_retries=1)

    with pytest.raises(LLMRateLimitExceededError):
        await adapter.generate("hello")

    assert mock_generate.await_count == 2


# --- generation is bounded, so a long answer can't drift into a 504 ---


@pytest.mark.asyncio
async def test_generate_bounds_output_tokens_and_thinking_level():
    mock_generate = AsyncMock(return_value=_make_mock_response())
    adapter = GeminiAdapter(
        client=_make_mock_client(mock_generate),
        max_output_tokens=256,
        thinking_level="MINIMAL",
    )

    await adapter.generate("hello")

    config = mock_generate.await_args.kwargs["config"]
    assert config.max_output_tokens == 256
    assert config.thinking_config.thinking_level == "MINIMAL"
