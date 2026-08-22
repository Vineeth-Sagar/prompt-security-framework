"""PII/sensitive-data scanner — the first output-governance check on every
LLM response, before it ever reaches the user.

Combines spaCy NER (catches names/places/organizations mentioned in
free text — things no regex can reliably find) with regex for
structurally-recognizable sensitive strings (email, phone, credit card,
API-key-shaped tokens). Neither alone is enough: NER misses a bare email
address, regex can't recognize "Sarah Chen" as a person's name.

Honest limitations, worth stating rather than discovering later:
- Phone regex is US-centric (a 10-digit NANP-style pattern);
  international formats will be missed.
- Credit card regex matches the common 4x4-digit *shape*, with no Luhn
  checksum validation — a random 16-digit number shaped like a card
  number will false-positive, and a validly-formatted-but-unusually-
  spaced real number could be missed.
- API-key detection is prefix-based (sk-, sk-ant-, AIza, ghp_, xox*,
  AQ.) for known providers' formats; a key format not in that list
  won't be caught.
- NER quality is only as good as en_core_web_sm's (the small model) —
  expect missed/misclassified entities on unusual names or phrasing.
  Observed directly while building this: "Here is my API key: ..."
  gets "API" tagged as ORG, redacting a harmless acronym alongside the
  real key. Harmless over-redaction, not a missed detection, but a
  concrete example of the model's real error rate rather than a
  hypothetical caveat.
"""

import re

from pydantic import BaseModel

from app.preprocessing.normalizer import _nlp

_NER_TYPES = {"PERSON", "GPE", "ORG"}

_REGEX_PATTERNS: dict[str, re.Pattern] = {
    "EMAIL": re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),
    "PHONE": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
    "API_KEY": re.compile(
        r"\b(?:"
        r"sk-ant-[A-Za-z0-9\-_]{20,}"  # Anthropic
        r"|sk-[A-Za-z0-9]{20,}"  # OpenAI
        r"|AIza[A-Za-z0-9_\-]{30,}"  # Google API key
        r"|AQ\.[A-Za-z0-9_\-]{20,}"  # Google AI Studio key
        r"|ghp_[A-Za-z0-9]{30,}"  # GitHub personal access token
        r"|xox[baprs]-[A-Za-z0-9\-]{10,}"  # Slack token
        r")\b"
    ),
}


class PIIMatch(BaseModel):
    type: str
    span: tuple[int, int]


class PIIScanResult(BaseModel):
    """Result of scanning one piece of text for PII/sensitive data.

    Attributes:
        found: Every match, in the order it appears in the original
            text — each a {type, span} pair (span is character offsets
            into the *original*, pre-redaction text).
        redacted_text: The text with every matched span replaced by
            `[REDACTED:{type}]`; everything else byte-for-byte
            unchanged, so surrounding sentence structure survives.
    """

    found: list[PIIMatch]
    redacted_text: str


def _collect_candidates(text: str) -> list[tuple[int, int, str]]:
    candidates: list[tuple[int, int, str]] = []

    doc = _nlp(text)
    for ent in doc.ents:
        if ent.label_ in _NER_TYPES:
            candidates.append((ent.start_char, ent.end_char, ent.label_))

    for type_name, pattern in _REGEX_PATTERNS.items():
        for match in pattern.finditer(text):
            candidates.append((match.start(), match.end(), type_name))

    return candidates


def _resolve_overlaps(candidates: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Greedily select non-overlapping spans, preferring longer matches
    when candidates overlap (e.g. a regex API-key match fully containing
    a shorter, spurious NER hit)."""
    ordered = sorted(candidates, key=lambda c: (c[0], -(c[1] - c[0])))

    selected: list[tuple[int, int, str]] = []
    last_end = -1
    for start, end, type_name in ordered:
        if start >= last_end:
            selected.append((start, end, type_name))
            last_end = end

    return sorted(selected, key=lambda c: c[0])


def scan(text: str) -> PIIScanResult:
    """Scan `text` for PII/sensitive data and produce a redacted version."""
    if not text:
        return PIIScanResult(found=[], redacted_text=text)

    selected = _resolve_overlaps(_collect_candidates(text))
    found = [PIIMatch(type=type_name, span=(start, end)) for start, end, type_name in selected]

    parts: list[str] = []
    cursor = 0
    for start, end, type_name in selected:
        parts.append(text[cursor:start])
        parts.append(f"[REDACTED:{type_name}]")
        cursor = end
    parts.append(text[cursor:])

    return PIIScanResult(found=found, redacted_text="".join(parts))
