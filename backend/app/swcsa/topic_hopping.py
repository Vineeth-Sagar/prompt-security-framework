"""Topic-hopping detector: how many distinct subjects appear in a short window.

Uses a simple *online* clustering pass over turn embeddings — not a
proven topic model, just a cheap proxy: assign each turn to the nearest
existing cluster centroid (by cosine distance) if it's close enough, else
open a new cluster (up to a small cap). Then compute the Shannon entropy
of the resulting cluster-label distribution, normalized to [0, 1] by the
maximum entropy possible for that window size/cluster cap.

High entropy in a short window (many turns landing in different
clusters, none dominant) is a proxy for "the conversation keeps jumping
subject" — a known slow-burn prompt-injection pattern where the attacker
deliberately hops topics to bury the eventual malicious ask among
unrelated ones.

Known limitation, found while validating this module: because each turn
is embedded independently (no conversational context carried across
turns), a short reply that leans on a pronoun or ellipsis to refer back
to an earlier turn — "when was it built?" referring to a tower named two
turns earlier — can embed as unexpectedly distant from that earlier
turn, even though a human reading the transcript sees an obviously
continuous topic. This under a real single-topic *support* conversation
(shared entities repeated per turn, e.g. an order number) still resolves
correctly. Worth revisiting if false positives on pronoun-heavy benign
chats show up in Phase 12 evaluation.
"""

import math
from collections import Counter

import numpy as np

from app.swcsa.drift_embeddings import embed_batch

DEFAULT_MAX_CLUSTERS = 3
# Cosine-distance threshold under which a turn joins an existing cluster
# rather than starting a new one. Chosen empirically-ish, not tuned
# against a validation set — a knob to revisit in Phase 12 evaluation.
DEFAULT_ASSIGNMENT_THRESHOLD = 0.5


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / denom)


def _assign_topics(
    embeddings: list[np.ndarray],
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
    threshold: float = DEFAULT_ASSIGNMENT_THRESHOLD,
) -> list[int]:
    """Online clustering: nearest-centroid assignment with a cap on cluster count."""
    centroids: list[np.ndarray] = []
    counts: list[int] = []
    labels: list[int] = []

    for emb in embeddings:
        if not centroids:
            centroids.append(emb.copy())
            counts.append(1)
            labels.append(0)
            continue

        distances = [_cosine_distance(emb, c) for c in centroids]
        nearest_idx = int(np.argmin(distances))
        nearest_dist = distances[nearest_idx]

        if nearest_dist <= threshold or len(centroids) >= max_clusters:
            idx = nearest_idx
            counts[idx] += 1
            # Running mean update, not a full recompute.
            centroids[idx] = centroids[idx] + (emb - centroids[idx]) / counts[idx]
            labels.append(idx)
        else:
            centroids.append(emb.copy())
            counts.append(1)
            labels.append(len(centroids) - 1)

    return labels


def topic_entropy(window: list[str], max_clusters: int = DEFAULT_MAX_CLUSTERS) -> float:
    """Shannon entropy of the topic-cluster distribution across `window`, in [0, 1].

    Fewer than 2 turns can't "hop" between topics, so returns 0.0.
    """
    if len(window) < 2:
        return 0.0

    embeddings = [np.array(e) for e in embed_batch(window)]
    labels = _assign_topics(embeddings, max_clusters=max_clusters)

    counts = Counter(labels)
    n = len(labels)
    probabilities = [count / n for count in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)

    max_possible_labels = min(n, max_clusters)
    max_possible_entropy = math.log2(max_possible_labels) if max_possible_labels > 1 else 0.0
    if max_possible_entropy == 0.0:
        return 0.0

    normalized = entropy / max_possible_entropy
    return max(0.0, min(1.0, normalized))
