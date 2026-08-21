"""Image modality input handler: OCR + EXIF metadata extraction.

Runs OCR (pytesseract, which shells out to the `tesseract` binary — see
backend/Dockerfile, which installs it) to recover any text embedded in an
image, and inspects EXIF metadata for fields that look like they're trying
to smuggle executable content past a human reviewer (e.g. an EXIF Comment
field containing a shell command). That check is a cheap heuristic, not a
real malware scanner — it just flags for the policy/output-governance
layers, it doesn't block anything itself.
"""

import io
import re

from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError

from app.input_layer.base import InputHandler, InputResult

# Deliberately simple: substrings that show up in shell/script payloads.
# False positives (a photo whose EXIF comment happens to mention "cmd")
# are acceptable here — this only sets a flag for later layers to weigh,
# it never blocks on its own.
_SUSPICIOUS_PATTERNS = re.compile(
    r"<script|javascript:|cmd\.exe|powershell|/bin/(sh|bash)|eval\(|exec\(",
    re.IGNORECASE,
)


class ImageInputHandler(InputHandler):
    """Extracts text (OCR) and flags suspicious metadata from image bytes."""

    async def process(self, raw: bytes) -> InputResult:
        """Decode `raw` image bytes, OCR them, and inspect EXIF metadata.

        Raises:
            HTTPException: 400, if `raw` cannot be decoded as an image.
                Raised directly (rather than ValueError) so a corrupted
                upload always surfaces as a clean 4xx response instead of
                bubbling into a 500 with a stack trace.
        """
        try:
            image = Image.open(io.BytesIO(raw))
            image.load()  # force decode now, not lazily on first use
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(
                status_code=400, detail="Could not decode image input."
            ) from exc

        text = self._extract_text(image)
        exif, suspicious_fields = self._extract_exif(image)

        return InputResult(
            text=text.strip(),
            modality="image",
            metadata={
                "exif": exif,
                "suspicious_metadata": bool(suspicious_fields),
                "suspicious_fields": suspicious_fields,
            },
        )

    @staticmethod
    def _extract_text(image: Image.Image) -> str:
        import pytesseract

        try:
            return pytesseract.image_to_string(image)
        except pytesseract.TesseractNotFoundError as exc:
            # Distinct from a decode failure: the image was fine, the
            # host is just missing the `tesseract` binary. Surface this
            # as a 500 (server misconfiguration), not a 400 (bad input).
            raise HTTPException(
                status_code=500,
                detail="OCR engine (tesseract) is not installed on this server.",
            ) from exc

    @staticmethod
    def _extract_exif(image: Image.Image) -> tuple[dict[str, str], list[str]]:
        """Return (exif_as_str_dict, field_names_flagged_as_suspicious)."""
        raw_exif = image.getexif()
        exif: dict[str, str] = {}
        suspicious: list[str] = []

        for tag_id, value in raw_exif.items():
            tag_name = str(tag_id)
            value_str = str(value)
            exif[tag_name] = value_str

            if _SUSPICIOUS_PATTERNS.search(value_str):
                suspicious.append(tag_name)

        return exif, suspicious
