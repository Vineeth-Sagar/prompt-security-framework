"""Policy rule engine — the single, auditable decision point.

Turns a SWCSA drift breakdown and an IFS-R reconstruction result into
one of BLOCK / SAFE_REWRITE / PASS, by evaluating a data-driven,
human-readable rule list (rules/default_policy.yaml) top to bottom and
taking the first full match. Every decision reports which named rule
fired, so the explainability dashboard (Phase 10) can always answer
"why did this get blocked?" with an actual rule name and rationale, not
"some code path decided so".

Condition strings use a small, deliberately constrained expression
language ("drift.aggregate >= 0.8") rather than arbitrary Python eval —
even though the YAML is admin-authored, not user-input-controlled, a
restricted grammar keeps every possible condition auditable by reading
the rule file, and keeps `_evaluate_condition` a total function that
can only ever compare a known field against a literal, never execute
anything.
"""

import operator
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ValidationError

from app.ifsr.reconstructor import ReconstructionResult
from app.swcsa.drift_score import DriftBreakdown

Action = Literal["BLOCK", "SAFE_REWRITE", "PASS"]

_DEFAULT_RULES_PATH = Path(__file__).parent / "rules" / "default_policy.yaml"

_CONDITION_RE = re.compile(r"^(\S+)\s*(>=|<=|==|!=|>|<)\s*(.+)$")
_OPERATORS = {
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    "<": operator.lt,
}


class PolicyConfigError(Exception):
    """Raised for a malformed policy file — a startup-time config error, not silently ignored."""


class PolicyRule(BaseModel):
    name: str
    when: list[str]
    action: Action
    rationale: str = ""


class PolicyDecision(BaseModel):
    """Result of `PolicyEngine.decide`.

    Attributes:
        action: BLOCK / SAFE_REWRITE / PASS.
        matched_rule: Name of the rule that fired — always populated,
            including for the built-in ifsr.blocked safety floor
            (reported as "ifsr_blocked_fallback").
        final_text: The text to actually forward, or None for BLOCK.
            For both PASS and SAFE_REWRITE this is `ifsr.safe_text` —
            the two actions differ in what they mean for audit/logging
            (did we have to intervene?), not in what text they produce.
    """

    action: Action
    matched_rule: str
    final_text: str | None


def _resolve_field(path: str, context: dict[str, Any]) -> Any:
    if "." not in path:
        raise PolicyConfigError(
            f"Malformed policy condition field {path!r} — expected 'object.field'."
        )
    obj_name, field_name = path.split(".", 1)
    if obj_name not in context:
        raise PolicyConfigError(
            f"Unknown object {obj_name!r} in policy condition — expected one of {sorted(context)}."
        )
    obj = context[obj_name]
    if not hasattr(obj, field_name):
        raise PolicyConfigError(
            f"Unknown field {field_name!r} on {obj_name!r} in policy condition."
        )
    return getattr(obj, field_name)


def _parse_value(raw: str) -> Any:
    raw = raw.strip()
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    try:
        return float(raw)
    except ValueError:
        return raw.strip("\"'")


def _evaluate_condition(condition: str, context: dict[str, Any]) -> bool:
    match = _CONDITION_RE.match(condition.strip())
    if not match:
        raise PolicyConfigError(f"Malformed policy condition: {condition!r}")

    field_path, op, raw_value = match.groups()
    actual = _resolve_field(field_path, context)
    expected = _parse_value(raw_value)
    return _OPERATORS[op](actual, expected)


class PolicyEngine:
    """Evaluates a loaded list of PolicyRule objects, first match wins."""

    def __init__(self, rules: list[PolicyRule]):
        self._rules = rules

    @classmethod
    def load(cls, path: str | Path = _DEFAULT_RULES_PATH) -> "PolicyEngine":
        """Load and validate a policy YAML file.

        Raises:
            PolicyConfigError: for malformed YAML syntax, a missing
                top-level 'rules' key, or a rule that fails schema
                validation (missing/wrong-typed field, invalid action) —
                always at load time, never silently ignored or deferred
                to first use.
        """
        path = Path(path)

        try:
            with path.open(encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise PolicyConfigError(f"Malformed policy YAML at {path}: {exc}") from exc
        except OSError as exc:
            raise PolicyConfigError(f"Could not read policy file at {path}: {exc}") from exc

        if not isinstance(data, dict) or "rules" not in data:
            raise PolicyConfigError(f"Policy file {path} is missing a top-level 'rules' key.")

        try:
            rules = [PolicyRule(**rule) for rule in data["rules"]]
        except (ValidationError, TypeError) as exc:
            raise PolicyConfigError(f"Malformed policy rule in {path}: {exc}") from exc

        if not rules:
            raise PolicyConfigError(f"Policy file {path} defines no rules.")

        return cls(rules)

    def decide(self, drift: DriftBreakdown, ifsr: ReconstructionResult) -> PolicyDecision:
        """Turn (drift, ifsr) into a PolicyDecision.

        Independent of the loaded rules: if `ifsr.blocked` is True (IFS-R
        itself couldn't produce any usable safe text), this always
        returns BLOCK immediately — a hard safety floor no YAML rule can
        override, since there's no final_text to forward regardless of
        what the rules say.
        """
        if ifsr.blocked:
            return PolicyDecision(
                action="BLOCK", matched_rule="ifsr_blocked_fallback", final_text=None
            )

        context = {"drift": drift, "ifsr": ifsr}

        for rule in self._rules:
            if all(_evaluate_condition(cond, context) for cond in rule.when):
                final_text = None if rule.action == "BLOCK" else ifsr.safe_text
                return PolicyDecision(
                    action=rule.action, matched_rule=rule.name, final_text=final_text
                )

        raise PolicyConfigError(
            "No policy rule matched and no unconditional default rule (when: []) was defined."
        )


_engine: PolicyEngine | None = None


def get_policy_engine() -> PolicyEngine:
    """Process-wide default PolicyEngine (lazy singleton), loaded from
    rules/default_policy.yaml. A plain function (not @lru_cache) so it
    stays a valid, overridable FastAPI dependency, same pattern as
    get_context_buffer()."""
    global _engine
    if _engine is None:
        _engine = PolicyEngine.load(_DEFAULT_RULES_PATH)
    return _engine
