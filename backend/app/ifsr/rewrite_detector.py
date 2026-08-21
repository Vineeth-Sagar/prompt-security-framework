"""MinHash-based detector for paraphrased/obfuscated rewrites of known attacks.

Complements the rule-based detectors (role_escalation, subintent_classifier)
rather than replacing them: those match exact phrasing via regex, so an
attacker who mangles a known phrase just enough — leetspeak
("1gn0r3 pr3v10us 1nstruct10ns"), spacing tricks ("i g n o r e ...") —
can dodge every regex while a human reading it would still recognize it
instantly. Character-shingle MinHash similarity against a small
known-attack corpus catches exactly that: leetspeak/spacing survive
normalization + shingling essentially unchanged (empirically ~1.0
Jaccard similarity to the canonical phrase), while genuine paraphrases
or coincidental word overlap score much lower (~0.1-0.25 in testing).

Not a semantic detector — a fragment saying "please don't ignore
previous formatting instructions" (benign, talking *about* not
ignoring something) can still score high on raw character overlap with
the canonical "ignore previous instructions". This is one signal for
the classifier/reconstruction pipeline to weigh, not a standalone
blocker.
"""

from pathlib import Path

import yaml
from datasketch import MinHash
from pydantic import BaseModel

_CORPUS_PATH = Path(__file__).parent / "rules" / "known_attack_corpus.yaml"

NUM_PERM = 128
SHINGLE_SIZE = 4
DEFAULT_THRESHOLD = 0.5

# Common leetspeak digit/symbol substitutions, mapped back to the letter
# they're standing in for. Deliberately small and conservative (not
# every '1' is an 'i') — good enough to defeat casual obfuscation, not a
# general leetspeak decoder.
_LEET_TRANSLATION = str.maketrans({
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "@": "a",
    "$": "s",
})


class CorpusEntry(BaseModel):
    id: str
    text: str


def _load_corpus(path: Path) -> list[CorpusEntry]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [CorpusEntry(**entry) for entry in data["corpus"]]


def _normalize(text: str) -> str:
    """Lowercase, fold common leetspeak, and strip everything but letters/digits.

    Stripping whitespace/punctuation entirely (not just collapsing it)
    is what defeats spacing-trick obfuscation ("i g n o r e") — after
    this, it's indistinguishable from the unspaced original.
    """
    lowered = text.lower().translate(_LEET_TRANSLATION)
    return "".join(ch for ch in lowered if ch.isalnum())


def _shingles(text: str, k: int = SHINGLE_SIZE) -> set[str]:
    if len(text) < k:
        return {text} if text else set()
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def _minhash(text: str) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    for shingle in _shingles(_normalize(text)):
        m.update(shingle.encode("utf-8"))
    return m


# Corpus signatures computed once at import time — same rationale as
# every other rule/pattern set in this project: cheap to keep in memory,
# wasteful to recompute per call.
_CORPUS = _load_corpus(_CORPUS_PATH)
_CORPUS_SIGNATURES = [(entry, _minhash(entry.text)) for entry in _CORPUS]


def rewrite_score(text: str, threshold: float = DEFAULT_THRESHOLD) -> tuple[float, str | None]:
    """Best MinHash Jaccard similarity between `text` and the known-attack corpus.

    Returns:
        (best_similarity, matched_id) — matched_id is the corpus entry
        id that scored highest, but is only returned (non-None) when
        that score also clears `threshold`; below threshold it's None
        even though best_similarity still reports the raw number, so
        callers can see how close a near-miss was.
    """
    if not text.strip():
        return 0.0, None

    fragment_signature = _minhash(text)

    best_score = 0.0
    best_id: str | None = None
    for entry, corpus_signature in _CORPUS_SIGNATURES:
        score = fragment_signature.jaccard(corpus_signature)
        if score > best_score:
            best_score = score
            best_id = entry.id

    matched = best_id if best_score >= threshold else None
    return best_score, matched
