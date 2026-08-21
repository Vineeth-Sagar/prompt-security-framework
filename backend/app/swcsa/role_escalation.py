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
