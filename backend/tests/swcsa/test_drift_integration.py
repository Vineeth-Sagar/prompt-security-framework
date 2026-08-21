"""Integration test: compute_drift() against the labeled synthetic dataset.

Reports real accuracy/false-positive/recall numbers — not asserted to a
specific value yet (per the build plan: this is the layer to tune most
in Phase 12 evaluation). The one thing this test does assert is a weak,
robust sanity check: benign conversations should score lower drift on
average than flagged ones. A specific accuracy threshold isn't asserted
here because a synthetic 45-example set is too small to lock in a
number without the assertion becoming either trivially loose or
flaky as the detectors get tuned.
"""

import json
from pathlib import Path

import pytest

from app.config import get_settings
from app.context_buffer.redis_buffer import TurnRecord
from app.swcsa.drift_score import compute_drift, is_flagged

DATASET_DIR = Path(__file__).parent.parent.parent / "datasets" / "swcsa_eval"


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


def test_drift_score_separates_benign_from_flagged_and_reports_metrics():
    dataset = _load_dataset()
    threshold = get_settings().drift_threshold

    results = []
    for row in dataset:
        window = [TurnRecord(text=t, role="user") for t in row["window"]]
        breakdown = compute_drift(window, row["new_text"])
        results.append(
            {
                "id": row["id"],
                "should_flag": row["should_flag"],
                "aggregate": breakdown.aggregate,
                "predicted_flag": is_flagged(breakdown),
            }
        )

    benign_scores = [r["aggregate"] for r in results if not r["should_flag"]]
    flagged_scores = [r["aggregate"] for r in results if r["should_flag"]]

    total = len(results)
    correct = sum(1 for r in results if r["predicted_flag"] == r["should_flag"])
    accuracy = correct / total

    negatives = [r for r in results if not r["should_flag"]]
    false_positives = sum(1 for r in negatives if r["predicted_flag"])
    false_positive_rate = false_positives / len(negatives) if negatives else 0.0

    positives = [r for r in results if r["should_flag"]]
    true_positives = sum(1 for r in positives if r["predicted_flag"])
    recall = true_positives / len(positives) if positives else 0.0

    mean_benign = sum(benign_scores) / len(benign_scores)
    mean_flagged = sum(flagged_scores) / len(flagged_scores)

    print(f"\n--- SWCSA drift-score evaluation (threshold={threshold}) ---")
    print(f"dataset size: {total} ({len(negatives)} benign, {len(positives)} should-flag)")
    print(f"accuracy: {accuracy:.1%}")
    print(f"false positive rate: {false_positive_rate:.1%}")
    print(f"recall (true positive rate): {recall:.1%}")
    print(f"mean benign aggregate: {mean_benign:.3f}")
    print(f"mean flagged aggregate: {mean_flagged:.3f}")

    misclassified = [r for r in results if r["predicted_flag"] != r["should_flag"]]
    if misclassified:
        print(f"misclassified ({len(misclassified)}):")
        for r in misclassified:
            print(
                f"  {r['id']:16s} aggregate={r['aggregate']:.2f} "
                f"expected={r['should_flag']} got={r['predicted_flag']}"
            )

    # The one hard assertion: on average, benign conversations should
    # drift-score lower than ones that should be flagged. This holds
    # even though several individual slow-burn examples currently score
    # below threshold (see printed misclassifications above) — that's
    # real signal for where Phase 12 tuning should focus, not a bug to
    # paper over here.
    assert mean_benign < mean_flagged


def test_no_false_positives_on_benign_dataset_at_default_threshold():
    # A stronger, still-realistic bar: none of the 15 hand-written
    # ordinary conversations should trip the default threshold. Losing
    # recall on subtle slow-burn attacks is an accepted, visible
    # trade-off right now; flagging normal conversations would not be.
    dataset = _load_jsonl("benign_multiturn.jsonl")

    false_positives = []
    for row in dataset:
        window = [TurnRecord(text=t, role="user") for t in row["window"]]
        breakdown = compute_drift(window, row["new_text"])
        if is_flagged(breakdown):
            false_positives.append(row["id"])

    assert false_positives == []


@pytest.mark.parametrize(
    "filename",
    ["benign_multiturn.jsonl", "injection_singleturn.jsonl", "injection_slowburn.jsonl"],
)
def test_dataset_file_is_well_formed(filename: str):
    rows = _load_jsonl(filename)

    assert len(rows) == 15
    for row in rows:
        assert isinstance(row["window"], list) and len(row["window"]) >= 1
        assert isinstance(row["new_text"], str) and row["new_text"]
        assert isinstance(row["should_flag"], bool)
