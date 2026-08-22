from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import errors as genai_errors

from app.llm_gateway.base import LLMRateLimitExceededError, LLMTimeoutError
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
