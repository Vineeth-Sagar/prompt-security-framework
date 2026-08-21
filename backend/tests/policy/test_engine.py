import pytest

from app.ifsr.reconstructor import ReconstructionResult
from app.policy.engine import PolicyConfigError, PolicyEngine, PolicyRule
from app.swcsa.drift_score import DriftBreakdown


def _drift(aggregate: float) -> DriftBreakdown:
    return DriftBreakdown(
        semantic=0.0,
        role_escalation=0.0,
        topic_entropy=0.0,
        window_role_escalation=0.0,
        drift_trend=0.0,
        matched_patterns=[],
        aggregate=aggregate,
    )


def _ifsr(
    safe_text: str = "some safe reconstructed text",
    malicious: bool = False,
    suspicious: bool = False,
    blocked: bool = False,
) -> ReconstructionResult:
    return ReconstructionResult(
        safe_text="" if blocked else safe_text,
        removed=["bad fragment"] if malicious else [],
        suspicious=["ambiguous fragment"] if suspicious else [],
        modified=malicious,
        blocked=blocked,
    )


@pytest.fixture
def engine() -> PolicyEngine:
    return PolicyEngine.load()  # the real default_policy.yaml


# --- one test per rule in default_policy.yaml, table-driven ---

TABLE = [
    # (label, drift_aggregate, malicious, suspicious, blocked, expected_action, expected_rule)
    (
        "high drift + malicious -> BLOCK",
        0.9,
        True,
        False,
        False,
        "BLOCK",
        "block_high_drift_with_malicious_content",
    ),
    (
        "high drift + only suspicious -> SAFE_REWRITE",
        0.9,
        False,
        True,
        False,
        "SAFE_REWRITE",
        "rewrite_high_drift_suspicious_only",
    ),
    (
        "high drift + no ifsr signal -> SAFE_REWRITE",
        0.9,
        False,
        False,
        False,
        "SAFE_REWRITE",
        "rewrite_high_drift_no_ifsr_signal",
    ),
    (
        "low drift + clean -> PASS",
        0.3,
        False,
        False,
        False,
        "PASS",
        "pass_low_drift_clean",
    ),
    (
        "mid-range drift, uncovered -> default SAFE_REWRITE",
        0.65,
        False,
        False,
        False,
        "SAFE_REWRITE",
        "default_safe_rewrite",
    ),
    (
        "ifsr.blocked forces BLOCK regardless of drift",
        0.1,
        False,
        False,
        True,
        "BLOCK",
        "ifsr_blocked_fallback",
    ),
]


@pytest.mark.parametrize(
    "label,aggregate,malicious,suspicious,blocked,expected_action,expected_rule",
    TABLE,
    ids=[row[0] for row in TABLE],
)
def test_policy_decision_table(
    engine: PolicyEngine,
    label: str,
    aggregate: float,
    malicious: bool,
    suspicious: bool,
    blocked: bool,
    expected_action: str,
    expected_rule: str,
):
    decision = engine.decide(
        _drift(aggregate), _ifsr(malicious=malicious, suspicious=suspicious, blocked=blocked)
    )

    assert decision.action == expected_action
    assert decision.matched_rule == expected_rule


def test_block_action_has_no_final_text(engine: PolicyEngine):
    decision = engine.decide(_drift(0.9), _ifsr(malicious=True))

    assert decision.final_text is None


def test_pass_and_rewrite_actions_carry_ifsr_safe_text(engine: PolicyEngine):
    decision = engine.decide(_drift(0.2), _ifsr(safe_text="help me plan a trip"))

    assert decision.final_text == "help me plan a trip"


# --- rule order: first match wins ---


def test_first_matching_rule_wins_not_a_later_broader_match():
    rules = [
        PolicyRule(name="specific_block", when=["drift.aggregate >= 0.5"], action="BLOCK"),
        PolicyRule(name="broad_pass", when=["drift.aggregate >= 0.0"], action="PASS"),
    ]
    engine = PolicyEngine(rules)

    # Both rules' conditions are true for aggregate=0.9 — the first one
    # in the list must win, not the (also-matching) second.
    decision = engine.decide(_drift(0.9), _ifsr())

    assert decision.action == "BLOCK"
    assert decision.matched_rule == "specific_block"


def test_reordering_rules_changes_which_one_wins():
    rules = [
        PolicyRule(name="broad_pass", when=["drift.aggregate >= 0.0"], action="PASS"),
        PolicyRule(name="specific_block", when=["drift.aggregate >= 0.5"], action="BLOCK"),
    ]
    engine = PolicyEngine(rules)

    decision = engine.decide(_drift(0.9), _ifsr())

    assert decision.action == "PASS"
    assert decision.matched_rule == "broad_pass"


def test_no_matching_rule_and_no_default_raises_config_error():
    rules = [PolicyRule(name="only_high", when=["drift.aggregate >= 0.9"], action="BLOCK")]
    engine = PolicyEngine(rules)

    with pytest.raises(PolicyConfigError, match="No policy rule matched"):
        engine.decide(_drift(0.1), _ifsr())


# --- malformed policy files raise a clear config error at load time ---


def test_malformed_yaml_syntax_raises_config_error(tmp_path):
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("rules:\n  - name: bad\n    when: [\n", encoding="utf-8")

    with pytest.raises(PolicyConfigError, match="Malformed policy YAML"):
        PolicyEngine.load(bad_file)


def test_missing_rules_key_raises_config_error(tmp_path):
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("foo: bar\n", encoding="utf-8")

    with pytest.raises(PolicyConfigError, match="missing a top-level 'rules' key"):
        PolicyEngine.load(bad_file)


def test_invalid_action_raises_config_error(tmp_path):
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text(
        "rules:\n  - name: bad\n    when: []\n    action: NOT_A_REAL_ACTION\n",
        encoding="utf-8",
    )

    with pytest.raises(PolicyConfigError, match="Malformed policy rule"):
        PolicyEngine.load(bad_file)


def test_empty_rules_list_raises_config_error(tmp_path):
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("rules: []\n", encoding="utf-8")

    with pytest.raises(PolicyConfigError, match="defines no rules"):
        PolicyEngine.load(bad_file)


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(PolicyConfigError, match="Could not read policy file"):
        PolicyEngine.load(tmp_path / "does_not_exist.yaml")


def test_malformed_condition_string_raises_config_error(tmp_path):
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text(
        "rules:\n  - name: bad\n    when: ['this is not a valid condition']\n    action: PASS\n",
        encoding="utf-8",
    )
    engine = PolicyEngine.load(bad_file)

    with pytest.raises(PolicyConfigError, match="Malformed policy condition"):
        engine.decide(_drift(0.5), _ifsr())


def test_default_policy_file_loads_successfully():
    # Smoke test: the real shipped policy file is well-formed.
    engine = PolicyEngine.load()

    assert len(engine._rules) >= 1
