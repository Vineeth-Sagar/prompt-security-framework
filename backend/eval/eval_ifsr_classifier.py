"""Phase 12 evaluation harness for IFS-R's per-fragment classifier
(app/ifsr/subintent_classifier.py).

Answers the question the project had no measured answer to before this
script existed: how much does each detection layer actually contribute,
and at what latency cost? Runs three variants against the same labeled
dataset (backend/datasets/ifsr_eval/fragments.jsonl, 100 hand-labeled
fragments — see that file for composition: 50 malicious across 5
categories including 5 isolated split-injection payload halves expected
to be missed, 15 suspicious/weak-signal, 35 benign including 15
deliberate "over-defense trap" cases that mention security-adjacent
vocabulary without being attacks):

- regex_only   — role_escalation_score() + subintent_rules.yaml only
                 (what this project had before the two live-bug fixes
                 this session made).
- semantic_only — semantic_injection_similarity.semantic_injection_score()
                 alone (the new signal, in isolation).
- combined      — subintent_classifier.classify() as shipped: all three
                 scores summed.

This is what turns "we fixed two reported bugs" into an actual measured
claim: which layer catches which attack category, whether combining
them costs meaningful latency, and where the known gap remains (payload-
half fragments, in isolation, addressed by classify_all()'s adjacency
logic instead — not exercised by this per-fragment harness, called out
explicitly in the results rather than silently passing).

Run: `python -m eval.eval_ifsr_classifier` from `backend/`. Writes a
human-readable ablation table to stdout and a copy to
eval/IFSR_EVAL_RESULTS.md (citable as-is in the project report).
"""

import json
import time
from pathlib import Path

from app.ifsr.fragmenter import Fragment
from app.ifsr.subintent_classifier import (
    MALICIOUS_THRESHOLD,
    SUSPICIOUS_THRESHOLD,
    _score_subintent_rules,
    classify,
)
from app.swcsa.role_escalation import role_escalation_score
from app.swcsa.semantic_injection_similarity import semantic_injection_score
from eval.ifsr_metrics import LabeledPrediction, compute_metrics

DATASET_PATH = Path(__file__).parent.parent / "datasets" / "ifsr_eval" / "fragments.jsonl"
RESULTS_PATH = Path(__file__).parent / "IFSR_EVAL_RESULTS.md"


def _load_dataset() -> list[dict]:
    rows = []
    with DATASET_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _label_from_score(score: float) -> str:
    if score >= MALICIOUS_THRESHOLD:
        return "malicious"
    if score >= SUSPICIOUS_THRESHOLD:
        return "suspicious"
    return "safe"


def _run_regex_only(rows: list[dict]) -> list[LabeledPrediction]:
    predictions = []
    for row in rows:
        start = time.perf_counter()
        role_score, _ = role_escalation_score(row["text"])
        subintent_score, _ = _score_subintent_rules(row["text"])
        score = min(1.0, role_score + subintent_score)
        elapsed_ms = (time.perf_counter() - start) * 1000
        predictions.append(
            LabeledPrediction(
                id=row["id"],
                category=row["category"],
                true_label=row["label"],
                predicted_label=_label_from_score(score),
                latency_ms=elapsed_ms,
            )
        )
    return predictions


def _run_semantic_only(rows: list[dict]) -> list[LabeledPrediction]:
    predictions = []
    for row in rows:
        start = time.perf_counter()
        score, _ = semantic_injection_score(row["text"])
        elapsed_ms = (time.perf_counter() - start) * 1000
        predictions.append(
            LabeledPrediction(
                id=row["id"],
                category=row["category"],
                true_label=row["label"],
                predicted_label=_label_from_score(score),
                latency_ms=elapsed_ms,
            )
        )
    return predictions


def _run_combined(rows: list[dict]) -> list[LabeledPrediction]:
    predictions = []
    for row in rows:
        fragment = Fragment(text=row["text"], span=(0, len(row["text"])), index=0)
        start = time.perf_counter()
        verdict = classify(fragment)
        elapsed_ms = (time.perf_counter() - start) * 1000
        predictions.append(
            LabeledPrediction(
                id=row["id"],
                category=row["category"],
                true_label=row["label"],
                predicted_label=verdict.risk,
                latency_ms=elapsed_ms,
            )
        )
    return predictions


VARIANTS = {
    "regex_only": _run_regex_only,
    "semantic_only": _run_semantic_only,
    "combined": _run_combined,
}


def _format_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _render_report(all_metrics: dict[str, dict]) -> str:
    lines = ["# IFS-R Fragment Classifier — Phase 12 Evaluation Results", ""]
    lines.append(
        f"Dataset: `backend/datasets/ifsr_eval/fragments.jsonl`, "
        f"{all_metrics['combined']['n']} hand-labeled fragments "
        "(50 malicious / 15 suspicious / 35 safe, including 15 "
        "over-defense-trap benign fragments and 5 isolated "
        "split-injection payload halves)."
    )
    lines.append("")
    lines.append(
        "`malicious_strict_recall` is the operationally relevant recall "
        "number — `reconstructor.py` only drops fragments verdicted "
        "exactly `malicious`; a fragment scored `suspicious` when it "
        "should have been `malicious` still reaches the target LLM in "
        "production. `malicious_catch_rate` is the softer number "
        "(flagged as suspicious *or* malicious — still visible in the "
        "explainability log even if not blocked)."
    )
    lines.append("")
    lines.append(
        "| Variant | Exact acc. | Malicious strict recall | Malicious catch rate "
        "| Benign FP rate | Binary P / R / F1 | Mean latency (ms) | p95 latency (ms) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for variant, m in all_metrics.items():
        lines.append(
            f"| {variant} | {_format_pct(m['exact_accuracy'])} "
            f"| {_format_pct(m['malicious_strict_recall'])} "
            f"| {_format_pct(m['malicious_catch_rate'])} "
            f"| {_format_pct(m['benign_false_positive_rate'])} "
            f"| {_format_pct(m['binary_precision'])} / {_format_pct(m['binary_recall'])} "
            f"/ {_format_pct(m['binary_f1'])} "
            f"| {m['mean_latency_ms']:.3f} "
            f"| {m['p95_latency_ms']:.3f} |"
        )
    lines.append("")

    lines.append("## Per-category breakdown (combined variant)")
    lines.append("")
    lines.append("| Category | n | True label | Metric | Value | Predicted-label counts |")
    lines.append("|---|---|---|---|---|---|")
    for category, breakdown in all_metrics["combined"]["by_category"].items():
        metric_name = "false_positive_rate" if "false_positive_rate" in breakdown else "recall"
        lines.append(
            f"| {category} | {breakdown['n']} | {breakdown['true_label']} "
            f"| {metric_name} | {_format_pct(breakdown[metric_name])} "
            f"| {breakdown['predicted_label_counts']} |"
        )
    lines.append("")

    lines.append(
        "**Known, expected gap:** `split_injection_payload` recall under "
        "`regex_only`/`semantic_only`/`combined` is measured here at the "
        "single-fragment level deliberately — it is expected to stay low "
        "for all three variants, since the payload half of a split "
        "injection carries no trigger words of its own by construction. "
        "That gap is closed by `classify_all()`'s adjacency-based "
        "contextual escalation (see `subintent_classifier.py`'s module "
        "docstring), which this harness does not exercise since it "
        "operates across fragments, not on one fragment in isolation. "
        "Not a regression — a reminder to eval `classify_all()` "
        "separately if this harness is extended."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    rows = _load_dataset()
    print(f"Loaded {len(rows)} labeled fragments from {DATASET_PATH.name}")

    all_metrics = {}
    for name, run_fn in VARIANTS.items():
        print(f"Running variant: {name} ...")
        predictions = run_fn(rows)
        all_metrics[name] = compute_metrics(predictions)

    report = _render_report(all_metrics)
    print("\n" + report)

    RESULTS_PATH.write_text(report, encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
