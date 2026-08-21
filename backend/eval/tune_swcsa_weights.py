"""Grid search over SWCSA's five aggregator weights against the labeled dataset.

Computes each of the five raw sub-scores once per dataset example (the
expensive part — embeddings, rule matching), then searches weight
combinations purely as fast weighted sums, scoring each combination by
accuracy at the configured DRIFT_THRESHOLD with zero tolerance for false
positives on the benign set (a combination that flags any of the 15
ordinary conversations is rejected outright, even if its raw accuracy
looks good — false positives are the one thing this project can't trade
away for recall).

This is a coarse, small-dataset-driven search (45 examples, weights on a
0.05 grid) — good enough to move off an arbitrary manual guess, not a
rigorously validated calibration. Revisit with the larger benchmark
Phase 12 is meant to build.

Runs two searches, printed for comparison: an *unconstrained* one
(informational only — it reliably concentrates weight onto 1-2 signals
and zeroes the rest, which overfits a small hand-written set and
defeats the point of having five independent signals) and a *floored*
one (every signal weight >= MIN_WEIGHT_FLOOR) that's actually used as
the recommended config — see main()'s comments for the reasoning.

Run: `python -m eval.tune_swcsa_weights` from `backend/`.
"""

import json
from collections.abc import Iterator
from pathlib import Path

from app.config import get_settings
from app.context_buffer.redis_buffer import TurnRecord
from app.swcsa.drift_embeddings import drift_trend_score, semantic_drift_cached
from app.swcsa.role_escalation import role_escalation_score, window_role_escalation_score
from app.swcsa.topic_hopping import topic_entropy

DATASET_DIR = Path(__file__).parent.parent / "datasets" / "swcsa_eval"
SIGNAL_NAMES = ["semantic", "role_escalation", "topic_entropy", "window_escalation", "drift_trend"]
GRID_STEP = 0.05


def _load_jsonl(filename: str) -> list[dict]:
    rows = []
    with (DATASET_DIR / filename).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_dataset() -> list[dict]:
    return (
        _load_jsonl("benign_multiturn.jsonl")
        + _load_jsonl("injection_singleturn.jsonl")
        + _load_jsonl("injection_slowburn.jsonl")
    )


def _compute_raw_signals(row: dict) -> dict[str, float]:
    window = [TurnRecord(text=t, role="user") for t in row["window"]]
    window_texts = [t.text for t in window]
    new_text = row["new_text"]

    semantic = semantic_drift_cached(window, new_text)
    role_score, _ = role_escalation_score(new_text)
    topic = topic_entropy([*window_texts, new_text])
    window_role_score, _ = window_role_escalation_score(window_texts)
    trend = drift_trend_score(window)

    return {
        "semantic": semantic,
        "role_escalation": role_score,
        "topic_entropy": topic,
        "window_escalation": window_role_score,
        "drift_trend": trend,
        "should_flag": row["should_flag"],
        "id": row["id"],
    }


def _compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    """All ways to write `total` as `parts` non-negative integers summing to it.

    Direct generation (not itertools.product + filter) — the naive
    product over a 21-value grid in 5 dimensions is ~4M tuples before
    filtering down to the ~10.6K that actually sum correctly.
    """
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first, *rest)


def _weight_grid(step: float) -> list[tuple[float, ...]]:
    """All 5-tuples on the `step` grid over [0,1] that sum to exactly 1.0."""
    units = round(1 / step)
    return [
        tuple(round(part * step, 10) for part in composition)
        for composition in _compositions(units, len(SIGNAL_NAMES))
    ]


def _score(weights: tuple[float, ...], signals: list[dict], threshold: float) -> dict:
    correct = 0
    false_positives = 0
    true_positives = 0
    n_positive = sum(1 for s in signals if s["should_flag"])
    n_negative = len(signals) - n_positive

    for s in signals:
        aggregate = sum(w * s[name] for w, name in zip(weights, SIGNAL_NAMES, strict=True))
        aggregate = max(0.0, min(1.0, aggregate))
        flagged = aggregate >= threshold

        if flagged == s["should_flag"]:
            correct += 1
        if flagged and not s["should_flag"]:
            false_positives += 1
        if flagged and s["should_flag"]:
            true_positives += 1

    return {
        "weights": weights,
        "accuracy": correct / len(signals),
        "false_positives": false_positives,
        "recall": true_positives / n_positive if n_positive else 0.0,
        "n_negative": n_negative,
    }


MIN_WEIGHT_FLOOR = 0.05


def _print_result(label: str, result: dict) -> None:
    print(f"\n--- {label} ---")
    for name, w in zip(SIGNAL_NAMES, result["weights"], strict=True):
        print(f"  {name}: {w}")
    print(f"  accuracy: {result['accuracy']:.1%}")
    print(f"  recall: {result['recall']:.1%}")
    print(f"  false positives: {result['false_positives']}/{result['n_negative']}")


def main() -> None:
    settings = get_settings()
    dataset = _load_dataset()
    print(f"Computing raw sub-scores for {len(dataset)} examples...")
    signals = [_compute_raw_signals(row) for row in dataset]

    grid = _weight_grid(GRID_STEP)
    print(f"Searching {len(grid)} weight combinations (step={GRID_STEP})...")

    results = [_score(w, signals, settings.drift_threshold) for w in grid]

    # Reject any combination with false positives on the benign set,
    # regardless of how good its accuracy looks otherwise.
    zero_fp = [r for r in results if r["false_positives"] == 0]
    print(f"{len(zero_fp)}/{len(results)} combinations have zero false positives.")

    unconstrained_best = max(zero_fp, key=lambda r: (r["accuracy"], r["recall"]))
    _print_result("Unconstrained best (informational only - see caveat below)", unconstrained_best)
    print(
        "  CAVEAT: the unconstrained search reliably drives weight onto just\n"
        "  1-2 signals (usually topic_entropy) and zeroes the rest, on a\n"
        "  45-example hand-written dataset. That's the classic overfitting-\n"
        "  to-a-small-set failure mode, and it also defeats the purpose of\n"
        "  having five independent signals in the first place: a real\n"
        "  attacker could specifically craft prompts that don't move the\n"
        "  one dominant signal. NOT used as the recommended config."
    )

    # Recommended: constrain every weight to at least MIN_WEIGHT_FLOOR, so
    # no signal is fully zeroed out, then take the best accuracy under
    # that constraint. This deliberately trades some measured accuracy on
    # this small dataset for defense-in-depth robustness — every signal
    # this pipeline computes has to actually contribute.
    floored = [r for r in zero_fp if min(r["weights"]) >= MIN_WEIGHT_FLOOR - 1e-9]
    recommended = max(floored, key=lambda r: (r["accuracy"], r["recall"]))
    _print_result(f"Recommended (floored >= {MIN_WEIGHT_FLOOR} per signal)", recommended)


if __name__ == "__main__":
    main()
