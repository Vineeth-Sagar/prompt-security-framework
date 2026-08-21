import asyncio
from unittest.mock import AsyncMock, MagicMock

import anthropic
import httpx
import pytest

from app.llm_gateway.anthropic_adapter import (
    AnthropicAdapter,
    LLMRateLimitExceededError,
    LLMTimeoutError,
)


def _make_rate_limit_error() -> anthropic.RateLimitError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(429, request=request)
    return anthropic.RateLimitError("rate limited", response=response, body=None)


def _make_mock_message(text: str = "Hello!", model: str = "claude-sonnet-5") -> MagicMock:
    message = MagicMock()
    message.content = [MagicMock(text=text)]
    message.model = model
    message.usage.input_tokens = 10
    message.usage.output_tokens = 5
    return message


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    # Every backoff test in this file uses asyncio.sleep for the retry
    # delay — monkeypatch it so tests run in milliseconds, not seconds,
    # while still exercising the real retry-count/control-flow logic.
    monkeypatch.setattr("asyncio.sleep", AsyncMock())


@pytest.mark.asyncio
async def test_successful_generate_returns_normalized_response():
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=_make_mock_message())

    adapter = AnthropicAdapter(client=mock_client, model="claude-sonnet-5")
    response = await adapter.generate("hello")

    assert response.text == "Hello!"
    assert response.model == "claude-sonnet-5"
    assert response.usage == {"input_tokens": 10, "output_tokens": 5}
    assert response.latency_ms >= 0


@pytest.mark.asyncio
async def test_no_real_api_call_is_ever_made():
    # The mock client's create() is the only thing that could reach the
    # network — asserting it's an AsyncMock (never a real AsyncAnthropic
    # transport) is the guarantee this whole test file makes no real calls.
    mock_client = MagicMock()
    mock_create = AsyncMock(return_value=_make_mock_message())
    mock_client.messages.create = mock_create

    adapter = AnthropicAdapter(client=mock_client)
    await adapter.generate("hello")

    mock_create.assert_awaited_once()


@pytest.mark.asyncio
async def test_retries_on_rate_limit_then_succeeds():
    call_count = 0

    async def flaky_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise _make_rate_limit_error()
        return _make_mock_message()

    mock_client = MagicMock()
    mock_client.messages.create = flaky_create

    adapter = AnthropicAdapter(client=mock_client, max_retries=3)
    response = await adapter.generate("hello")

    assert call_count == 3
    assert response.text == "Hello!"


@pytest.mark.asyncio
async def test_exhausting_retries_raises_typed_rate_limit_error():
    mock_client = MagicMock()

    async def always_rate_limited(*args, **kwargs):
        raise _make_rate_limit_error()

    mock_client.messages.create = always_rate_limited

    adapter = AnthropicAdapter(client=mock_client, max_retries=2)

    with pytest.raises(LLMRateLimitExceededError, match="after 2 retries"):
        await adapter.generate("hello")


@pytest.mark.asyncio
async def test_retry_count_respects_max_retries_setting():
    call_count = 0

    async def always_rate_limited(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise _make_rate_limit_error()

    mock_client = MagicMock()
    mock_client.messages.create = always_rate_limited

    adapter = AnthropicAdapter(client=mock_client, max_retries=4)

    with pytest.raises(LLMRateLimitExceededError):
        await adapter.generate("hello")

    assert call_count == 5  # initial attempt + 4 retries


@pytest.mark.asyncio
async def test_timeout_raises_clean_typed_exception():
    async def hangs_forever(*args, **kwargs):
        # A never-resolving Future — genuinely hangs until wait_for
        # cancels it, independent of the asyncio.sleep mock the retry
        # tests in this file rely on (an asyncio.sleep(...)-based fake
        # would resolve instantly under that mock, defeating the test).
        return await asyncio.get_event_loop().create_future()

    mock_client = MagicMock()
    mock_client.messages.create = hangs_forever

    adapter = AnthropicAdapter(client=mock_client, timeout_seconds=0.05, max_retries=0)

    with pytest.raises(LLMTimeoutError, match="0.05s timeout"):
        await adapter.generate("hello")


@pytest.mark.asyncio
async def test_timeout_does_not_retry():
    call_count = 0

    async def hangs_forever(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return await asyncio.get_event_loop().create_future()

    mock_client = MagicMock()
    mock_client.messages.create = hangs_forever

    adapter = AnthropicAdapter(client=mock_client, timeout_seconds=0.05, max_retries=3)

    with pytest.raises(LLMTimeoutError):
        await adapter.generate("hello")

    assert call_count == 1  # timeout is not retried, unlike rate limiting


@pytest.mark.asyncio
async def test_model_and_max_tokens_can_be_overridden_per_call():
    mock_client = MagicMock()
    mock_create = AsyncMock(return_value=_make_mock_message())
    mock_client.messages.create = mock_create

    adapter = AnthropicAdapter(client=mock_client, model="claude-sonnet-5")
    await adapter.generate("hello", model="claude-opus-5", max_tokens=42)

    _, kwargs = mock_create.call_args
    assert kwargs["model"] == "claude-opus-5"
    assert kwargs["max_tokens"] == 42
