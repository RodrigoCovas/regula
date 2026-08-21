from src.eval_harness import CURATED_CASES, run_eval, score_case, ExpectedFinding, ProducedFinding, Strength


def test_perfect_match_scores_one():
    expected = [
        ExpectedFinding(statement="Loan scoring is high-risk", strength=Strength.strong),
        ExpectedFinding(statement="Definitions apply", strength=Strength.weak),
    ]
    produced = [
        ProducedFinding(statement="Loan scoring is high-risk", strength=Strength.strong),
        ProducedFinding(statement="Definitions apply", strength=Strength.weak),
    ]
    result = score_case(expected, produced)
    assert result["f1"] == 1.0


def test_missed_finding_lowers_recall_by_weight():
    """Getting the weak Finding right but missing the strong one: recall = 1/4."""
    expected = [
        ExpectedFinding(statement="Loan scoring is high-risk", strength=Strength.strong),
        ExpectedFinding(statement="Definitions apply", strength=Strength.weak),
    ]
    produced = [ProducedFinding(statement="Definitions apply", strength=Strength.weak)]
    result = score_case(expected, produced)
    assert result["recall"] == 0.25


def test_missing_strong_finding_hurts_more_than_missing_weak():
    expected = [
        ExpectedFinding(statement="Loan scoring is high-risk", strength=Strength.strong),
        ExpectedFinding(statement="Definitions apply", strength=Strength.weak),
    ]
    miss_strong = score_case(expected, [ProducedFinding(statement="Definitions apply", strength=Strength.weak)])
    miss_weak = score_case(expected, [ProducedFinding(statement="Loan scoring is high-risk", strength=Strength.strong)])
    assert miss_strong["recall"] < miss_weak["recall"]


def test_spurious_finding_subtracts_by_produced_strength():
    """A fabricated strong claim hurts more than a fabricated weak one."""
    expected = [ExpectedFinding(statement="Loan scoring is high-risk", strength=Strength.strong)]
    correct = ProducedFinding(statement="Loan scoring is high-risk", strength=Strength.strong)

    spurious_weak = score_case(expected, [correct, ProducedFinding(statement="Made up", strength=Strength.weak)])
    assert spurious_weak["precision"] == 0.5

    spurious_strong = score_case(expected, [correct, ProducedFinding(statement="Made up", strength=Strength.strong)])
    assert spurious_strong["precision"] == 0.0

    assert spurious_strong["recall"] == 1.0


def test_mistag_off_by_one_costs_half():
    """A matched Finding tagged one step off retains half its credit."""
    expected = [ExpectedFinding(statement="Loan scoring is high-risk", strength=Strength.strong)]
    produced = [ProducedFinding(statement="Loan scoring is high-risk", strength=Strength.moderate)]
    result = score_case(expected, produced)
    assert result["recall"] == 0.5


def test_mistag_weak_strong_reversal_costs_whole_finding():
    expected = [ExpectedFinding(statement="Framing definitions", strength=Strength.weak)]
    produced = [ProducedFinding(statement="Framing definitions", strength=Strength.strong)]
    result = score_case(expected, produced)
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0


def test_curated_case_count_within_spec():
    assert 6 <= len(CURATED_CASES) <= 8


def test_deterministic_demo_scores_perfect_on_all_curated_cases():
    report = run_eval()
    assert len(report.cases) == len(CURATED_CASES)
    for case in report.cases:
        assert case.f1 == 1.0, f"case {case.id} scored {case.f1}"
    assert report.mean_f1 == 1.0
