"""Sub-intent risk classifier — judges one fragment at a time.

v1 is rule-based, deliberately: reuses Phase 4's role-escalation rule
set (a fragment saying "ignore previous instructions" is exactly as
risky standing alone as it was as part of a whole prompt) plus a
second, IFS-R-specific rule file (rules/subintent_rules.yaml) covering
three categories role-escalation doesn't: data exfiltration, sandbox/
code-execution escape, and PII solicitation about third parties.

Extension point for v2: swap or blend in a learned classifier by
replacing this module's `classify()` body — callers everywhere else
only depend on the `classify(fragment) -> RiskVerdict` signature, not on
it being rule-based. A natural v2 would train on labeled fragments (not
whole prompts) using the same embeddings SWCSA already computes.
"""

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from app.ifsr.fragmenter import Fragment
from app.swcsa.role_escalation import role_escalation_score

_RULES_PATH = Path(__file__).parent / "rules" / "subintent_rules.yaml"

RiskLevel = Literal["safe", "suspicious", "malicious"]

# Deliberately separate from SWCSA's DRIFT_THRESHOLD — this operates on
# a single clause's risk score, not a whole-conversation drift score,
# so the two aren't comparable and shouldn't share a constant.
SUSPICIOUS_THRESHOLD = 0.3
MALICIOUS_THRESHOLD = 0.6


class SubintentRule(BaseModel):
    id: str
    pattern: str
    weight: float
    category: str
    case_sensitive: bool = False
    description: str = ""


def _load_rules(path: Path) -> list[SubintentRule]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [SubintentRule(**rule) for rule in data["rules"]]


# Loaded once at import time — same rationale as role_escalation.py's
# _RULES: the file rarely changes at runtime.
_RULES = _load_rules(_RULES_PATH)


class RiskVerdict(BaseModel):
    """Risk judgment for one fragment.

    Attributes:
        risk: "safe" / "suspicious" / "malicious", derived from `score`
            against SUSPICIOUS_THRESHOLD/MALICIOUS_THRESHOLD.
        reason: Human-readable explanation — which rules fired, or that
            none did. For the explainability log (Phase 10).
        score: The raw combined score (role-escalation + subintent
            rules, clipped to [0, 1]) that `risk` was derived from.
        matched_patterns: Rule ids that fired, from either rule set.
    """

    risk: RiskLevel
    reason: str
    score: float
    matched_patterns: list[str]


def _score_subintent_rules(text: str) -> tuple[float, list[str]]:
    matched: list[str] = []
    total_weight = 0.0

    for rule in _RULES:
        flags = 0 if rule.case_sensitive else re.IGNORECASE
        if re.search(rule.pattern, text, flags):
            matched.append(rule.id)
            total_weight += rule.weight

    return min(1.0, total_weight), matched


def classify(fragment: Fragment) -> RiskVerdict:
    """Judge a single fragment's risk level."""
    role_score, role_matched = role_escalation_score(fragment.text)
    subintent_score, subintent_matched = _score_subintent_rules(fragment.text)

    score = min(1.0, role_score + subintent_score)
    matched = [*role_matched, *subintent_matched]

    if score >= MALICIOUS_THRESHOLD:
        risk: RiskLevel = "malicious"
    elif score >= SUSPICIOUS_THRESHOLD:
        risk = "suspicious"
    else:
        risk = "safe"

    reason = f"matched: {', '.join(matched)}" if matched else "no risk patterns matched"

    return RiskVerdict(risk=risk, reason=reason, score=score, matched_patterns=matched)
