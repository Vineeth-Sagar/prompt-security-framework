"""POST /api/v1/input — the single entry point for all three modalities.

Accepts a multipart upload (`file`) plus a `modality` form field, runs it
through the matching input handler, normalizes the extracted text (Phase
2), and returns the `InputResult` as JSON — with no unhandled exceptions
on malformed input.
"""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.input_layer.base import InputResult
from app.input_layer.router import get_handler, resolve_modality
from app.preprocessing.normalizer import normalize

router = APIRouter(prefix="/api/v1", tags=["input"])


@router.post("/input", response_model=InputResult)
async def submit_input(
    file: UploadFile = File(...),  # noqa: B008 — FastAPI's DI idiom, not a real default-call bug
    modality: str | None = Form(None),  # noqa: B008
) -> InputResult:
    """Route an uploaded file to the matching handler and return its InputResult."""
    raw = await file.read()

    try:
        resolved_modality = resolve_modality(modality, file.content_type, file.filename)
        handler = get_handler(resolved_modality)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if resolved_modality == "text":
        try:
            content: str | bytes = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=400, detail="Text input must be UTF-8 encoded."
            ) from exc
    else:
        content = raw

    try:
        result = await handler.process(content)
    except ValueError as exc:
        # Handlers raise ValueError for application-level validation
        # failures (empty/oversized text, ...). Image/audio handlers may
        # instead raise HTTPException directly for undecodable bytes —
        # that propagates through FastAPI unchanged.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _apply_normalization(result)


def _apply_normalization(result: InputResult) -> InputResult:
    """Overwrite `result.text` with its normalized form before it leaves the API.

    The lowercased form becomes the primary `text` field so every
    downstream consumer sees the same normalized string regardless of
    modality. The cased variant and token list aren't discarded — later
    layers (e.g. a role-escalation detector caring about "SYSTEM:" vs
    "system:") need them, so they're preserved under
    `metadata["normalized"]` instead of being dropped.
    """
    normalized = normalize(result.text)

    result.text = normalized.text
    result.metadata["normalized"] = {
        "text_cased": normalized.text_cased,
        "tokens": normalized.tokens,
    }

    return result
