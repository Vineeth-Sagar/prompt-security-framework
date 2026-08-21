"""Text modality input handler.

The simplest handler: no transcription/OCR needed, just validation and
whitespace trimming. Still goes through the same `InputHandler` interface
so the router (added in step 1.5) can treat every modality uniformly.
"""

from app.input_layer.base import InputHandler, InputResult

MAX_TEXT_LENGTH = 8000


class TextInputHandler(InputHandler):
    """Validates and passes through raw text input."""

    async def process(self, raw: str) -> InputResult:
        """Trim `raw` and wrap it in an `InputResult`.

        Raises:
            ValueError: if `raw` is empty/whitespace-only, or exceeds
                `MAX_TEXT_LENGTH` characters after trimming.
        """
        text = raw.strip()

        if not text:
            raise ValueError("Text input must not be empty.")

        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"Text input exceeds the {MAX_TEXT_LENGTH}-character limit "
                f"(got {len(text)})."
            )

        return InputResult(text=text, modality="text")
