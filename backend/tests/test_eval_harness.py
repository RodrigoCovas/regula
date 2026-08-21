from src.eval_harness import (
    CURATED_SCENARIOS,
    EvalScenario,
    run_eval,
    score_scenario,
    ExpectedFinding,
    ProducedFinding,
    Strength,
)


def test_perfect_match_scores_one():
    expected = [
        ExpectedFinding(statement="Loan scoring is high-risk", strength=Strength.strong),
        ExpectedFinding(statement="Definitions apply", strength=Strength.weak),
    ]
    produced = [
        ProducedFinding(statement="Loan scoring is high-risk", strength=Strength.strong),
        ProducedFinding(statement="Definitions apply", strength=Strength.weak),
    ]
    result = score_scenario(expected, produced)
    assert result["f1"] == 1.0


def test_missed_finding_lowers_recall_by_weight():
    """Getting the weak Finding right but missing the strong one: recall = 1/4."""
    expected = [
        ExpectedFinding(statement="Loan scoring is high-risk", strength=Strength.strong),
        ExpectedFinding(statement="Definitions apply", strength=Strength.weak),
    ]
    produced = [ProducedFinding(statement="Definitions apply", strength=Strength.weak)]
    result = score_scenario(expected, produced)
    assert result["recall"] == 0.25


def test_missing_strong_finding_hurts_more_than_missing_weak():
    expected = [
        ExpectedFinding(statement="Loan scoring is high-risk", strength=Strength.strong),
        ExpectedFinding(statement="Definitions apply", strength=Strength.weak),
    ]
    miss_strong = score_scenario(expected, [ProducedFinding(statement="Definitions apply", strength=Strength.weak)])
    miss_weak = score_scenario(expected, [ProducedFinding(statement="Loan scoring is high-risk", strength=Strength.strong)])
    assert miss_strong["recall"] < miss_weak["recall"]


def test_spurious_finding_subtracts_full_produced_weight():
    """A fabricated strong claim hurts more than a fabricated weak one; the
    subtraction is always the FULL weight of the produced Strength — never discounted."""
    expected = [ExpectedFinding(statement="Loan scoring is high-risk", strength=Strength.strong)]
    correct = ProducedFinding(statement="Loan scoring is high-risk", strength=Strength.strong)

    spurious_weak = score_scenario(expected, [correct, ProducedFinding(statement="Made up", strength=Strength.weak)])
    assert spurious_weak["precision"] == 0.5

    spurious_moderate = score_scenario(expected, [correct, ProducedFinding(statement="Made up", strength=Strength.moderate)])
    assert spurious_moderate["precision"] == 0.2

    spurious_strong = score_scenario(expected, [correct, ProducedFinding(statement="Made up", strength=Strength.strong)])
    assert spurious_strong["precision"] == 0.0

    assert spurious_strong["recall"] == 1.0


def test_mistag_off_by_one_costs_half():
    """A matched Finding tagged one step off retains half its credit."""
    expected = [ExpectedFinding(statement="Loan scoring is high-risk", strength=Strength.strong)]
    produced = [ProducedFinding(statement="Loan scoring is high-risk", strength=Strength.moderate)]
    result = score_scenario(expected, produced)
    assert result["recall"] == 0.5


def test_mistag_weak_strong_reversal_costs_whole_finding():
    expected = [ExpectedFinding(statement="Framing definitions", strength=Strength.weak)]
    produced = [ProducedFinding(statement="Framing definitions", strength=Strength.strong)]
    result = score_scenario(expected, produced)
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_empty_expected_nothing_produced_scores_one():
    """A non-canonical Scenario expecting no Findings and producing none is a perfect run."""
    result = score_scenario([], [])
    assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_empty_expected_with_leakage_scores_zero():
    """Demo Findings leaking into a foreign Scenario must score zero, not perfect."""
    leaked = [
        ProducedFinding(statement="Loan scoring is high-risk", strength=Strength.strong),
        ProducedFinding(statement="Definitions apply", strength=Strength.weak),
    ]
    result = score_scenario([], leaked)
    assert result["f1"] == 0.0


def test_curated_scenario_count_within_spec():
    assert 6 <= len(CURATED_SCENARIOS) <= 8


def test_deterministic_demo_scores_perfect_on_all_curated_scenarios():
    report = run_eval()
    assert len(report.scenarios) == len(CURATED_SCENARIOS)
    for scenario in report.scenarios:
        assert scenario.f1 == 1.0, f"scenario {scenario.id} scored {scenario.f1}"
    assert report.mean_f1 == 1.0
