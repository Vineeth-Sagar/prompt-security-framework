"""Resolves which InputHandler should process a given upload.

Modality is normally declared explicitly by the caller (the API route
requires it as a form field — see api/routes/input.py), but this module
can also infer it from Content-Type or filename extension, so a client
that only sets one of those still gets routed correctly.
"""

from pathlib import Path

from app.input_layer.audio_handler import AudioInputHandler
from app.input_layer.base import InputHandler, Modality
from app.input_layer.image_handler import ImageInputHandler
from app.input_layer.text_handler import TextInputHandler

# One handler instance per modality, shared across requests — cheap for
# text/image, and important for audio, where re-instantiating would
# reload the whisper model every call.
_HANDLERS: dict[Modality, InputHandler] = {
    "text": TextInputHandler(),
    "image": ImageInputHandler(),
    "audio": AudioInputHandler(),
}

_CONTENT_TYPE_MODALITY: dict[str, Modality] = {
    "text/plain": "text",
    "image/png": "image",
    "image/jpeg": "image",
    "image/jpg": "image",
    "audio/wav": "audio",
    "audio/x-wav": "audio",
    "audio/mpeg": "audio",
    "audio/mp3": "audio",
}

_EXTENSION_MODALITY: dict[str, Modality] = {
    ".txt": "text",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".wav": "audio",
    ".mp3": "audio",
}


def resolve_modality(
    declared: str | None,
    content_type: str | None = None,
    filename: str | None = None,
) -> Modality:
    """Determine the modality for a request.

    Preference order: an explicitly declared modality, then Content-Type,
    then filename extension.

    Raises:
        ValueError: if none of the three signals resolve to a known
            modality (empty declared value, unrecognized content type,
            unrecognized/missing extension).
    """
    if declared in _HANDLERS:
        return declared  # type: ignore[return-value]

    if content_type in _CONTENT_TYPE_MODALITY:
        return _CONTENT_TYPE_MODALITY[content_type]

    if filename:
        ext = Path(filename).suffix.lower()
        if ext in _EXTENSION_MODALITY:
            return _EXTENSION_MODALITY[ext]

    raise ValueError(
        "Could not determine input modality — pass one of "
        f"{sorted(_HANDLERS)} explicitly, or use a recognized "
        "Content-Type/file extension."
    )


def get_handler(modality: Modality) -> InputHandler:
    return _HANDLERS[modality]
