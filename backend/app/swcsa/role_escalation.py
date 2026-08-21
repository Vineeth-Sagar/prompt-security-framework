"""Role-escalation heuristic detector.

Matches a prompt against a data-driven rule set (`rules/role_escalation_rules.yaml`
— not hardcoded here, so the panel/guide can audit and tune it without
reading Python) covering common instruction-override phrasing and
role-prefix spoofing ("System:"/"Assistant:" appearing inside user text).

Deliberately a keyword/regex layer, not a learned classifier — see the
rule file's header comment for the honest limitations (paraphrase
evasion, occasional false positives on prompts that discuss these
phrases rather than attempt them).
"""

import re
from pathlib import Path

import yaml
from pydantic import BaseModel

_RULES_PATH = Path(__file__).parent / "rules" / "role_escalation_rules.yaml"


class RoleEscalationRule(BaseModel):
    id: str
    pattern: str
    weight: float
    case_sensitive: bool = False
    description: str = ""


def _load_rules(path: Path) -> list[RoleEscalationRule]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [RoleEscalationRule(**rule) for rule in data["rules"]]


# Loaded once at import time — the rule file rarely changes at runtime,
# and re-parsing YAML + recompiling regex on every call would be wasted
# work. Restart the process (or call `_load_rules` directly) to pick up
# edits during rule tuning.
_RULES = _load_rules(_RULES_PATH)


def role_escalation_score(text: str) -> tuple[float, list[str]]:
    """Score `text` against the role-escalation rule set.

    `text` should be the *cased* variant of the input (see
    `preprocessing.normalizer.NormalizedText.text_cased` /
    `InputResult.metadata["normalized"]["text_cased"]`), not the
    lowercased default — several rules here specifically depend on
    casing (e.g. spoofed "System:" vs incidental "system:").

    Returns:
        (score, matched_rule_ids) — score is the sum of matched rules'
        weights, clipped to [0, 1]; matched_rule_ids lists which rules
        fired, for the explainability log (Phase 10).
    """
    matched: list[str] = []
    total_weight = 0.0

    for rule in _RULES:
        flags = 0 if rule.case_sensitive else re.IGNORECASE
        if re.search(rule.pattern, text, flags):
            matched.append(rule.id)
            total_weight += rule.weight

    return min(1.0, total_weight), matched


def window_role_escalation_score(window_texts: list[str]) -> tuple[float, list[str]]:
    """Average role-escalation exposure across the *prior* turns in a window.

    `role_escalation_score` only ever looks at the newest prompt. A
    slow-burn attack often spreads weak/borderline escalation phrasing
    across several earlier turns instead of concentrating it in one —
    "we've built trust", "you already agreed", "as previously agreed" —
    none of which alone is alarming, but their *sustained* presence
    across a short window is itself a signal this function exists to
    surface. It's a background-pressure signal, reported separately
    from (and combined additively with, not replacing) the newest-turn
    score in the aggregator.

    Note: window turns' text comes from `TurnRecord.text`, which — per
    the current input-pipeline wiring (Phase 2/3) — is the *lowercased*
    normalized form, not the cased variant `role_escalation_score`
    otherwise expects. This means the case-sensitive role-prefix-
    spoofing rules can't fire against historical turns (only the
    case-insensitive rules can), which is an accepted, documented gap:
    catching a spoofed "SYSTEM:" specifically in an *earlier* turn is a
    narrower risk than catching sustained escalation pressure generally.

    Returns:
        (score, matched_rule_ids) — score is the mean per-turn score,
        clipped to [0, 1]; matched_rule_ids is the de-duplicated union
        of every rule that fired in any turn (for explainability).
        Empty window scores 0.0 with no matches.
    """
    if not window_texts:
        return 0.0, []

    scores: list[float] = []
    matched_all: set[str] = set()

    for text in window_texts:
        score, matched = role_escalation_score(text)
        scores.append(score)
        matched_all.update(matched)

    average = sum(scores) / len(scores)
    return min(1.0, average), sorted(matched_all)
