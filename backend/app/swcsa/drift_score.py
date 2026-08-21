"""Weighted drift-score aggregator — SWCSA's final output.

Combines the three sub-detectors (semantic drift, role-escalation
heuristics, topic-hopping entropy) into one `DriftBreakdown`: a single
`aggregate` score in [0, 1] plus the per-signal numbers and matched rule
ids that produced it, so a downstream explainability log (Phase 10) can
show *why* a prompt was flagged, not just that it was.

Weights and the flag threshold are read from settings (env-configurable,
see `.env.example`) rather than hardcoded, so tuning them during Phase 12
evaluation doesn't require a code change.
"""

from pydantic import BaseModel

from app.config import get_settings
from app.context_buffer.redis_buffer import TurnRecord
from app.swcsa.drift_embeddings import semantic_drift_cached
from app.swcsa.role_escalation import role_escalation_score
from app.swcsa.topic_hopping import topic_entropy


class DriftBreakdown(BaseModel):
    """Per-signal drift scores plus the weighted aggregate.

    Attributes:
        semantic: Embedding-based drift from the window's mean, in [0, 1].
        role_escalation: Role-escalation heuristic score, in [0, 1].
        topic_entropy: Topic-hopping entropy over window + new turn, in [0, 1].
        matched_patterns: Role-escalation rule ids that fired.
        aggregate: Weighted sum of the three signals above, clipped to [0, 1].
    """

    semantic: float
    role_escalation: float
    topic_entropy: float
    matched_patterns: list[str]
    aggregate: float


def compute_drift(
    window: list[TurnRecord],
    new_text: str,
    new_text_cased: str | None = None,
) -> DriftBreakdown:
    """Score `new_text` against `window` (the session's recent turns).

    Args:
        window: Recent turns (oldest first), as returned by
            `ContextBuffer.get_window`. Embeddings are cached on each
            TurnRecord (see `drift_embeddings.semantic_drift_cached`) —
            passing the same TurnRecord objects across calls avoids
            recomputation.
        new_text: The incoming prompt's normalized (lowercased) text.
            Used for semantic drift and topic entropy, where casing
            doesn't carry signal.
        new_text_cased: The same prompt with casing preserved. Used for
            role-escalation detection, where casing does carry signal
            (e.g. "SYSTEM:" vs "system:"). Falls back to `new_text` if
            not given.
    """
    settings = get_settings()
    cased = new_text_cased if new_text_cased is not None else new_text

    semantic = semantic_drift_cached(window, new_text)
    role_score, matched = role_escalation_score(cased)
    topic = topic_entropy([*(turn.text for turn in window), new_text])

    aggregate = (
        settings.swcsa_weight_semantic * semantic
        + settings.swcsa_weight_role_escalation * role_score
        + settings.swcsa_weight_topic_entropy * topic
    )
    aggregate = max(0.0, min(1.0, aggregate))

    return DriftBreakdown(
        semantic=semantic,
        role_escalation=role_score,
        topic_entropy=topic,
        matched_patterns=matched,
        aggregate=aggregate,
    )


def is_flagged(breakdown: DriftBreakdown) -> bool:
    """Whether `breakdown.aggregate` meets the configured drift threshold.

    A convenience wrapper, not a hidden decision — the policy engine
    (Phase 6) is where BLOCK/SAFE_REWRITE/PASS actually gets decided;
    this just exposes the same `settings.drift_threshold` comparison SWCSA
    itself uses for its own evaluation/reporting (e.g. Phase 12).
    """
    return breakdown.aggregate >= get_settings().drift_threshold
