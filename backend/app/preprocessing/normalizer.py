"""Text normalization: the last cleanup step before any downstream analysis
(SWCSA, IFS-R, ...) ever sees the text.

Handles two categories of unicode-based prompt-injection obfuscation:

1. **Invisible/format characters** (zero-width space, word joiner, BOM,
   ...) inserted mid-word to defeat naive keyword matching — e.g.
   "ig\\u200bnore previous instructions" still *reads* as "ignore" to a
   human but won't substring-match "ignore" unless the zero-width
   character is stripped, not just replaced with whitespace.
2. **Compatibility lookalikes** (fullwidth Latin letters, ligatures, ...)
   that visually or semantically resemble ASCII but bypass literal string
   matching — NFKC normalization folds these back to their canonical
   ASCII form.

This module does not decide what's malicious; it just makes sure the text
handed to layers that *do* make that decision can't be trivially evaded
by unicode tricks.
"""

import re
import unicodedata

import spacy
from pydantic import BaseModel, ConfigDict
from spacy.tokens import Doc

# Loaded once at import time, matching the singleton pattern used for the
# whisper model in the input layer — the model is the expensive resource,
# not the normalize() call.
_nlp = spacy.load("en_core_web_sm")

_WHITESPACE_RE = re.compile(r"\s+")

# Unicode general categories treated as "invisible" and stripped outright
# (not replaced with a space, since that would defeat the mid-word
# smuggling trick above): Cf = format (zero-width space/joiner, BOM, ...),
# Cc = control characters. Standard whitespace controls (\t \n \r \f \v)
# are excluded here — they're handled by the whitespace-collapse step
# below instead of being deleted outright.
_PRESERVED_WHITESPACE_CONTROLS = {"\t", "\n", "\r", "\f", "\v"}
_INVISIBLE_CATEGORIES = {"Cf", "Cc"}


class NormalizedText(BaseModel):
    """Result of `normalize()`.

    Attributes:
        original: The raw, untouched input.
        text: Fully normalized text — NFKC-folded, invisible characters
            stripped, whitespace collapsed, and (by default) lowercased.
            This is the primary field downstream layers should use.
        text_cased: Same cleanup, but casing preserved. Kept separately
            because casing itself can be a signal later (e.g. a
            role-escalation detector treating "SYSTEM:" differently from
            "system:") that would be lost if only the lowercased form
            were available.
        tokens: Token texts from spaCy's tokenizer, run on `text_cased`
            (tokenization quality benefits from real casing).
        doc: The spaCy `Doc` itself, for callers that need more than
            token text (POS tags, entities, dependency parse, ...).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)  # spaCy Doc isn't a pydantic type

    original: str
    text: str
    text_cased: str
    tokens: list[str]
    doc: Doc


def _strip_invisible_chars(text: str) -> str:
    def _keep(ch: str) -> bool:
        return ch in _PRESERVED_WHITESPACE_CONTROLS or unicodedata.category(ch) not in (
            _INVISIBLE_CATEGORIES
        )

    return "".join(ch for ch in text if _keep(ch))


def normalize(text: str, *, lowercase: bool = True) -> NormalizedText:
    """Normalize `text` for downstream analysis.

    Pipeline: NFKC normalization -> strip invisible/format characters ->
    collapse whitespace -> (optionally) lowercase -> spaCy tokenization.

    Idempotent: `normalize(normalize(x).text).text == normalize(x).text`,
    since every step here is itself idempotent (NFKC on already-NFKC text
    is a no-op, there are no invisible characters left to strip the
    second time, whitespace is already single-spaced, and lowercasing
    already-lowercased text is a no-op).

    Does not raise on empty/whitespace-only input — that's an
    application-level validation decision (see
    `input_layer.text_handler.TextInputHandler`), not this function's
    concern. Empty input simply normalizes to an empty result.
    """
    nfkc = unicodedata.normalize("NFKC", text)
    without_invisible = _strip_invisible_chars(nfkc)
    collapsed = _WHITESPACE_RE.sub(" ", without_invisible).strip()

    text_cased = collapsed
    final_text = collapsed.lower() if lowercase else collapsed

    doc = _nlp(text_cased)
    tokens = [token.text for token in doc]

    return NormalizedText(
        original=text,
        text=final_text,
        text_cased=text_cased,
        tokens=tokens,
        doc=doc,
    )
