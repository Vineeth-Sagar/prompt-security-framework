"""Semantic-similarity injection detector — a paraphrase-resistant
complement to role_escalation.py's regex layer, feeding into
subintent_classifier.classify() as a third score alongside
role_escalation_score()/subintent_rules.yaml.

Motivation, precisely: two live false negatives traced back to the
same root cause. role_escalation.py's ignore_previous_instructions
pattern requires the determiner after "ignore" to be one of a fixed,
enumerated list (none / "all" / "the" / "your"); "ignore ANY previous
instructions" isn't in that list, so the regex doesn't match at all —
regardless of how obviously malicious the phrase reads to a person.
This isn't a one-off oversight, it's the documented, structural failure
mode of keyword/regex detection: "Regular expressions miss paraphrased
attacks" (Beyond Pattern Matching: Seven Cross-Domain Techniques for
Prompt Injection Detection, arXiv:2604.18248). Extending the regex's
word list fixes today's report; the underlying gap — an unbounded set
of possible phrasings, matched against a bounded list of expected ones
— reopens on the next paraphrase someone tries.

The standard complementary technique, not a replacement for regex, is
embedding-space similarity to a small set of known-attack exemplars:
protectai/rebuff's "vector database of known attacks" layer works this
way, and the same shape appears across published embedding-based
injection-detection studies. It generalizes across wording because it
compares meaning, not literal tokens — "ignore ANY previous
instructions" sits close to "ignore ALL previous instructions" in
embedding space even though a regex built around "all" doesn't see it.

Reuses drift_embeddings.py's already-loaded all-MiniLM-L6-v2 — no new
dependency, no new model-load cost.

Threshold calibration note, stated honestly because it matters:
published cosine-similarity thresholds for this technique (commonly
~0.82 in vector-DB implementations, measuring *intra-cluster* average
similarity across large real attack corpora) do NOT transfer directly
to this project's setup — a different embedding model, a small
hand-picked exemplar set, and single-exemplar-vs-text comparison
(rather than distance to a cluster centroid) all shift the useful
range substantially. Measured empirically against this project's own
fragmenter output before picking the constants below: real paraphrased
injection fragments scored 0.41-0.81 similarity to their nearest
exemplar; every benign fragment already in this project's test suite
topped out at 0.32, with one deliberate outlier ("what are hidden
instructions in general" — an existing regex-flagged "suspicious" case)
at 0.52. SEMANTIC_SUSPICIOUS_FLOOR/SEMANTIC_MALICIOUS_CEILING below are
set from that measured gap, not the literature default. Re-measure if
the exemplar set or embedding model ever changes — Phase 12's
evaluation harness is the right place to formalize this rather than
hand-checking it again by script each time.

Deliberately scoped to IFS-R's per-fragment classify() only, not
SWCSA's whole-turn/window role-escalation signals (role_escalation.py) —
those are called up to ~window_size+1 times per request, and adding an
embedding call to each would multiply this module's cost across every
prior turn in the window for comparatively little of this project's
observed false negatives (both were IFS-R-level misses). Extending
there later is architecturally trivial (same function, same import)
if evaluation data shows it's worth the added latency.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel

from app.swcsa.drift_embeddings import cosine_similarity, embed, embed_batch

_EXEMPLARS_PATH = Path(__file__).parent / "rules" / "semantic_exemplars.yaml"

# Below this, no contribution at all.
SEMANTIC_SUSPICIOUS_FLOOR = 0.50
# At/above this, full contribution (1.0 — the same scale a strong regex
# rule match uses).
SEMANTIC_MALICIOUS_CEILING = 0.65


class _Exemplar(BaseModel):
    id: str
    text: str


def _load_exemplars(path: Path) -> list[_Exemplar]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [_Exemplar(**item) for item in data["exemplars"]]


# Loaded and embedded once at import time — same rationale as
# role_escalation.py's _RULES: rebuilding embeddings on every call
# would be wasted, repeated model inference for text that never changes
# at runtime.
_EXEMPLARS = _load_exemplars(_EXEMPLARS_PATH)
_EXEMPLAR_EMBEDDINGS = embed_batch([exemplar.text for exemplar in _EXEMPLARS])


def semantic_injection_score(text: str) -> tuple[float, list[str]]:
    """Score `text` by embedding-space closeness to known injection intents.

    Returns:
        (score, matched_exemplar_ids) — score is the highest cosine
        similarity to any exemplar, linearly remapped from
        [SEMANTIC_SUSPICIOUS_FLOOR, SEMANTIC_MALICIOUS_CEILING] onto
        [0, 1] (0 below the floor, 1 at/above the ceiling), so it sits
        on the same scale role_escalation_score()/subintent rules use
        and can be summed with them directly. matched_exemplar_ids
        lists every exemplar `text` scored at least
        SEMANTIC_SUSPICIOUS_FLOOR against (for explainability),
        prefixed "semantic_similarity:" to stay distinguishable from
        regex rule ids in a combined matched-patterns list.
    """
    if not text.strip():
        return 0.0, []

    vector = embed(text)
    similarities = [
        cosine_similarity(vector, embedding) for embedding in _EXEMPLAR_EMBEDDINGS
    ]
    best_similarity = max(similarities)

    if best_similarity < SEMANTIC_SUSPICIOUS_FLOOR:
        return 0.0, []

    matched = [
        f"semantic_similarity:{exemplar.id}"
        for exemplar, similarity in zip(_EXEMPLARS, similarities, strict=True)
        if similarity >= SEMANTIC_SUSPICIOUS_FLOOR
    ]
    span = SEMANTIC_MALICIOUS_CEILING - SEMANTIC_SUSPICIOUS_FLOOR
    score = min(1.0, (best_similarity - SEMANTIC_SUSPICIOUS_FLOOR) / span)
    return score, matched
