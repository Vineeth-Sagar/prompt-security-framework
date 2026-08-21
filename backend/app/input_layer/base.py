"""Common interface every modality-specific input handler implements.

Each handler normalizes its raw input (text string, image bytes, audio
bytes) into the same `InputResult` shape so everything downstream of the
input layer (preprocessing, context buffer, SWCSA, ...) works against one
schema regardless of what the user actually sent.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field

Modality = Literal["text", "audio", "image"]


class InputResult(BaseModel):
    """Normalized output of any input handler.

    Attributes:
        text: The extracted/normalized text content (transcription, OCR
            result, or the passed-through text itself).
        modality: Which handler produced this result.
        confidence: Handler-specific confidence score in [0, 1], or None
            when the modality has no meaningful confidence signal (e.g.
            plain text passthrough).
        metadata: Handler-specific extra data (e.g. EXIF fields, a
            suspicious-metadata flag, audio signal-check results). Kept as
            a free-form dict rather than per-modality subclasses so the
            downstream pipeline can stay modality-agnostic.
    """

    text: str
    modality: Modality
    confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class InputHandler(ABC):
    """Base class for a modality-specific input handler."""

    @abstractmethod
    async def process(self, raw: Any) -> InputResult:
        """Normalize `raw` input into an `InputResult`.

        Implementations must raise `ValueError` for input that is
        malformed at the application level (empty, oversized, wrong
        shape) — the API layer is responsible for turning that into a
        clean HTTP error rather than a stack trace.
        """
        raise NotImplementedError
