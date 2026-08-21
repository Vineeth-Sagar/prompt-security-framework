import shutil
from pathlib import Path

import pytest

from app.input_layer.image_handler import ImageInputHandler

FIXTURES = Path(__file__).parent / "fixtures"

# pytesseract shells out to the `tesseract` binary. It's installed in the
# Docker image (backend/Dockerfile) and in CI (.github/workflows/ci.yml),
# but may not be present on every local dev machine — skip OCR-dependent
# tests rather than fail on a missing system dependency unrelated to the
# code under test.
requires_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None,
    reason="tesseract binary not found on PATH",
)


@pytest.fixture
def handler() -> ImageInputHandler:
    return ImageInputHandler()


@requires_tesseract
@pytest.mark.asyncio
async def test_ocr_extracts_expected_substring(handler: ImageInputHandler):
    raw = (FIXTURES / "sample_text.png").read_bytes()

    result = await handler.process(raw)

    assert "HELLO WORLD" in result.text.upper()
    assert result.modality == "image"


@requires_tesseract
@pytest.mark.asyncio
async def test_blank_image_returns_empty_text_without_raising(handler: ImageInputHandler):
    raw = (FIXTURES / "sample_blank.png").read_bytes()

    result = await handler.process(raw)

    assert result.text == ""
    assert result.metadata["suspicious_metadata"] is False


@pytest.mark.asyncio
async def test_corrupted_bytes_raise_http_400_not_a_stack_trace(handler: ImageInputHandler):
    from fastapi import HTTPException

    garbage = b"this is not a valid image file, just noise \x00\x01\x02"

    with pytest.raises(HTTPException) as exc_info:
        await handler.process(garbage)

    assert exc_info.value.status_code == 400
