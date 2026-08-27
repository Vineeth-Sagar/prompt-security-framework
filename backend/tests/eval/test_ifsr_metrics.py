"""Unit tests for eval/ifsr_metrics.py's pure aggregation logic — no
model loading, no dataset file, entirely synthetic LabeledPredictions.
This is deliberately the fast, always-runs-in-CI half of Phase 12: the
harness script itself (eval_ifsr_classifier.py) loads sentence-
transformers and is meant to be run manually/for the report, not as a
pytest case, but the arithmetic it depends on is tested here exactly
like any other production code.
"""

import pytest

from eval.ifsr_metrics import LabeledPrediction, compute_metrics


def _pred(
    true_label: str,
    predicted_label: str,
    category: str = "cat",
    latency_ms: float = 1.0,
) -> LabeledPrediction:
    return LabeledPrediction(
        id=f"{true_label}-{predicted_label}",
        category=category,
        true_label=true_label,
        predicted_label=predicted_label,
        latency_ms=latency_ms,
    )


def test_empty_predictions_raises():
    with pytest.raises(ValueError, match="empty"):
        compute_metrics([])


def test_perfect_classifier_scores_all_ones():
    predictions = [
        _pred("malicious", "malicious", category="role_escalation"),
        _pred("suspicious", "suspicious", category="weak_signal"),
        _pred("safe", "safe", category="benign_general"),
    ]

    metrics = compute_metrics(predictions)

    assert metrics["exact_accuracy"] == 1.0
    assert metrics["binary_precision"] == 1.0
    assert metrics["binary_recall"] == 1.0
    assert metrics["binary_f1"] == 1.0
    assert metrics["malicious_strict_recall"] == 1.0
    assert metrics["malicious_catch_rate"] == 1.0
    assert metrics["benign_false_positive_rate"] == 0.0


def test_missed_malicious_fragment_hurts_strict_recall_not_catch_rate():
    # A malicious fragment scored "suspicious" instead of "malicious":
    # still "caught" (flagged), but reconstructor.py would not drop it —
    # strict recall must reflect that, catch rate must not.
    predictions = [
        _pred("malicious", "suspicious", category="role_escalation"),
        _pred("malicious", "malicious", category="role_escalation"),
    ]

    metrics = compute_metrics(predictions)

    assert metrics["malicious_strict_recall"] == 0.5
    assert metrics["malicious_catch_rate"] == 1.0


def test_completely_missed_malicious_fragment_hurts_both_recalls():
    predictions = [
        _pred("malicious", "safe", category="split_injection_payload"),
        _pred("malicious", "malicious", category="role_escalation"),
    ]

    metrics = compute_metrics(predictions)

    assert metrics["malicious_strict_recall"] == 0.5
    assert metrics["malicious_catch_rate"] == 0.5


def test_false_positive_on_benign_lowers_precision_and_raises_fp_rate():
    predictions = [
        _pred("safe", "suspicious", category="overdefense_trap"),
        _pred("safe", "safe", category="benign_general"),
        _pred("malicious", "malicious", category="role_escalation"),
    ]

    metrics = compute_metrics(predictions)

    assert metrics["benign_false_positive_rate"] == 0.5
    # 2 true positives worth of "flagged" ground truth: malicious (caught)
    # + the false positive; precision = TP / (TP + FP) = 1 / (1 + 1)
    assert metrics["binary_precision"] == 0.5
    assert metrics["binary_recall"] == 1.0


def test_category_with_no_malicious_examples_does_not_crash():
    # Every true label in this synthetic run is "safe" — malicious_*
    # metrics must fall back to 0.0, not divide by zero.
    predictions = [_pred("safe", "safe", category="benign_general")]

    metrics = compute_metrics(predictions)

    assert metrics["malicious_strict_recall"] == 0.0
    assert metrics["malicious_catch_rate"] == 0.0


def test_by_category_breakdown_reports_recall_for_attack_categories():
    predictions = [
        _pred("malicious", "malicious", category="pii_solicitation"),
        _pred("malicious", "safe", category="pii_solicitation"),
    ]

    metrics = compute_metrics(predictions)

    breakdown = metrics["by_category"]["pii_solicitation"]
    assert breakdown["n"] == 2
    assert breakdown["true_label"] == "malicious"
    assert breakdown["recall"] == 0.5


def test_by_category_breakdown_reports_false_positive_rate_for_safe_categories():
    predictions = [
        _pred("safe", "safe", category="overdefense_trap"),
        _pred("safe", "malicious", category="overdefense_trap"),
    ]

    metrics = compute_metrics(predictions)

    breakdown = metrics["by_category"]["overdefense_trap"]
    assert breakdown["false_positive_rate"] == 0.5


def test_latency_percentiles_are_computed_from_the_predictions():
    predictions = [
        _pred("safe", "safe", latency_ms=1.0),
        _pred("safe", "safe", latency_ms=2.0),
        _pred("safe", "safe", latency_ms=3.0),
    ]

    metrics = compute_metrics(predictions)

    assert metrics["mean_latency_ms"] == pytest.approx(2.0)
    assert metrics["p95_latency_ms"] == 3.0
