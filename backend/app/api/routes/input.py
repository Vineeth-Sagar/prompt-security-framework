"""POST /api/v1/input — the single entry point for all three modalities.

Accepts a multipart upload (`file`) plus a `modality` form field and
returns the normalized `InputResult` as JSON. This is the last stop
before the input layer hands off to preprocessing (added in Phase 2) —
nothing downstream exists yet, so this route's job for now is just
"accept upload -> normalize -> return", with no unhandled exceptions on
malformed input.
"""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.input_layer.base import InputResult
from app.input_layer.router import get_handler, resolve_modality

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
        return await handler.process(content)
    except ValueError as exc:
        # Handlers raise ValueError for application-level validation
        # failures (empty/oversized text, ...). Image/audio handlers may
        # instead raise HTTPException directly for undecodable bytes —
        # that propagates through FastAPI unchanged.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
