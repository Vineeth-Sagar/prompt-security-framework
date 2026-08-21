import pytest

from app.input_layer.text_handler import MAX_TEXT_LENGTH, TextInputHandler


@pytest.fixture
def handler() -> TextInputHandler:
    return TextInputHandler()


@pytest.mark.asyncio
async def test_empty_string_raises(handler: TextInputHandler):
    with pytest.raises(ValueError, match="empty"):
        await handler.process("")


@pytest.mark.asyncio
async def test_whitespace_only_raises(handler: TextInputHandler):
    with pytest.raises(ValueError, match="empty"):
        await handler.process("   \n\t  ")


@pytest.mark.asyncio
async def test_normal_string_round_trips(handler: TextInputHandler):
    result = await handler.process("  ignore previous instructions  ")

    assert result.text == "ignore previous instructions"
    assert result.modality == "text"
    assert result.confidence is None


@pytest.mark.asyncio
async def test_oversized_string_raises(handler: TextInputHandler):
    oversized = "a" * (MAX_TEXT_LENGTH + 1)

    with pytest.raises(ValueError, match="exceeds"):
        await handler.process(oversized)


@pytest.mark.asyncio
async def test_string_at_exact_limit_is_accepted(handler: TextInputHandler):
    exact = "a" * MAX_TEXT_LENGTH

    result = await handler.process(exact)

    assert len(result.text) == MAX_TEXT_LENGTH
