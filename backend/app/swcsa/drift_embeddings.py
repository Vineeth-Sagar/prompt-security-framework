"""Semantic drift scoring via sentence embeddings.

Embeds text with sentence-transformers (`all-MiniLM-L6-v2`, CPU-only,
~90MB). Worth flagging honestly: the build plan's "Tools & Technologies"
section targets a ~30MB embedding model for footprint reasons —
MiniLM-L6-v2 is the smallest widely-used model that still gives usable
semantic quality, so this trades footprint for accuracy rather than
hitting that number. Swap `_MODEL_NAME` if a smaller/distilled model
becomes acceptable.

Drift is `1 - cosine_similarity(new_text, mean(window))` — how far a new
prompt sits from the recent conversation's average direction in
embedding space. Higher = more semantically distant from established
context.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from app.context_buffer.redis_buffer import TurnRecord

_MODEL_NAME = "all-MiniLM-L6-v2"

# Loaded once at import time — same singleton pattern as the whisper
# model (input_layer/audio_handler.py) and spaCy pipeline
# (preprocessing/normalizer.py): pay the model-load cost once per
# process, not once per request.
_model = SentenceTransformer(_MODEL_NAME)


def embed(text: str) -> list[float]:
    """Embed one string. Returns a plain list (JSON-serializable, so it
    can be cached on a TurnRecord) rather than a numpy array."""
    vector = _model.encode(text, convert_to_numpy=True, normalize_embeddings=False)
    return vector.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple strings in one batched model call."""
    if not texts:
        return []
    vectors = _model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
    return vectors.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _drift_from_embeddings(
    window_embeddings: list[list[float]], new_embedding: list[float]
) -> float:
    if not window_embeddings:
        return 0.0

    mean_embedding = np.mean(np.array(window_embeddings), axis=0).tolist()
    similarity = cosine_similarity(new_embedding, mean_embedding)
    drift = 1.0 - similarity
    # cosine similarity lives in [-1, 1], so raw drift is in [0, 2];
    # clip to [0, 1] since the aggregator (drift_score.py) treats every
    # sub-score as living in that range.
    return max(0.0, min(1.0, drift))


def semantic_drift(window: list[str], new_text: str) -> float:
    """1 - cosine_similarity(new_text, mean(window_embeddings)).

    An empty window has no prior context to drift *from*, so it always
    scores 0.0 — the first turn of a session can't be flagged for
    drifting away from nothing. This recomputes every embedding fresh;
    for a window of TurnRecords that may already have cached embeddings,
    use `semantic_drift_cached` instead.
    """
    if not window:
        return 0.0

    window_embeddings = embed_batch(window)
    new_embedding = embed(new_text)
    return _drift_from_embeddings(window_embeddings, new_embedding)


def get_or_compute_embedding(turn: TurnRecord) -> list[float]:
    """Return `turn.embedding`, computing and caching it in place if unset.

    Mutates the passed-in TurnRecord but does not persist it — writing
    the cached embedding back to Redis, if desired, is the caller's
    responsibility (e.g. the context buffer's own write path), keeping
    this module decoupled from persistence concerns.
    """
    if turn.embedding is None:
        turn.embedding = embed(turn.text)
    return turn.embedding


def semantic_drift_cached(window: list[TurnRecord], new_text: str) -> float:
    """Same as `semantic_drift`, but reuses each TurnRecord's cached
    embedding instead of recomputing it for every call against the same
    window."""
    if not window:
        return 0.0

    window_embeddings = [get_or_compute_embedding(turn) for turn in window]
    new_embedding = embed(new_text)
    return _drift_from_embeddings(window_embeddings, new_embedding)


def drift_trend_score(window: list[TurnRecord]) -> float:
    """How much semantic drift is *accelerating* turn-over-turn within the window itself.

    `semantic_drift`/`semantic_drift_cached` compare the new prompt to
    the window's mean — a single snapshot. This instead looks at
    consecutive-turn drift *within* the window (turn[i] vs turn[i-1])
    and returns a simple, honestly-described trend signal: the last
    consecutive-pair drift minus the first, clipped to [0, 1] (only a
    *rising* trend counts — a window that's settling back toward its
    starting topic isn't the pattern this is meant to catch).

    This is not a rigorous statistical trend test (no regression, no
    significance check) — just last-minus-first over a handful of
    points, which is honest for a window of 3-5 turns but would be a
    poor choice at a larger window size. A short window that's already
    gradually pulling away from where it started — even if no single
    consecutive pair looks remarkable — is the slow-burn pattern this
    is a proxy for.

    Needs at least 3 turns (2 consecutive-pair drifts to compare);
    returns 0.0 below that.
    """
    if len(window) < 3:
        return 0.0

    embeddings = [get_or_compute_embedding(turn) for turn in window]
    consecutive_drifts = [
        1.0 - cosine_similarity(embeddings[i], embeddings[i - 1])
        for i in range(1, len(embeddings))
    ]

    trend = consecutive_drifts[-1] - consecutive_drifts[0]
    return max(0.0, min(1.0, trend))
