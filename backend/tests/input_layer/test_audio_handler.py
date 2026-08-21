from pathlib import Path

import pytest
from fastapi import HTTPException

from app.input_layer.audio_handler import AudioInputHandler

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def handler() -> AudioInputHandler:
    return AudioInputHandler()


@pytest.mark.asyncio
async def test_transcription_contains_expected_keyword(handler: AudioInputHandler):
    raw = (FIXTURES / "sample_speech.wav").read_bytes()

    result = await handler.process(raw)

    assert "ignore" in result.text.lower()
    assert result.modality == "audio"
    assert result.confidence is not None
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.asyncio
async def test_signal_check_runs_and_does_not_flag_clean_speech(handler: AudioInputHandler):
    raw = (FIXTURES / "sample_speech.wav").read_bytes()

    result = await handler.process(raw)

    assert result.metadata["signal_check"]["performed"] is True
    assert result.metadata["signal_check"]["flagged"] is False


@pytest.mark.asyncio
async def test_corrupt_audio_raises_http_400_not_a_stack_trace(handler: AudioInputHandler):
    garbage = b"this is not a valid wav/audio file, just noise \x00\x01\x02" * 10

    with pytest.raises(HTTPException) as exc_info:
        await handler.process(garbage)

    assert exc_info.value.status_code == 400
