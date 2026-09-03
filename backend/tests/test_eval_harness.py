"""Suite-level pins for the eval harness: the Demo tripwire, the
per-request mode contract, and the audit dump that rides each case's
result. The scoring arithmetic itself is covered at the pure seam in
test_provision_coverage.py, the rubric judge in test_eval_judge.py
(ADR-0010); here the two components meet the evaluation loop — scripted
judge in, per-case and aggregate numbers out, never blended."""

from src.eval_harness import (
    CURATED_SCENARIOS,
    EvalScenario,
    ExpectedCitation,
    ExpectedFinding,
    evaluate_scenarios,
    run_eval,
)
from src.models import AnalyzeResponse, Answer, Citation, Finding, GroundedSummary, Mode, ProvisionTarget, Strength, Trace, answer_citations

import pytest

from fakes import (
    CONTRADICTION_VERDICT,
    FULL_AGREEMENT_VERDICT,
    POLARITY_FLIP_VERDICT,
    ROLE_MISMATCH_VERDICT,
    ScriptedJudge,
)


def test_curated_scenario_count_within_spec():
    assert 6 <= len(CURATED_SCENARIOS) <= 8


def test_deterministic_demo_scores_perfect_on_all_curated_scenarios():
    report = run_eval()
    assert len(report.scenarios) == len(CURATED_SCENARIOS)
    for scenario in report.scenarios:
        assert scenario.f1 == 1.0, f"scenario {scenario.id} scored {scenario.f1}"
    assert report.mean_f1 == 1.0


def test_demo_report_keeps_the_components_separate():
    """The Demo tripwire runs judgeless (no provider in CI), so summary
    fidelity is unmeasured on every case — the authored relevance summaries
    have no judge to face. Coverage, needing no judge, still scores the
    tripwire (test_deterministic_demo_scores_perfect_on_all_curated_scenarios)."""
    report = run_eval()
    for scenario in report.scenarios:
        assert scenario.summary_fidelity is None
    assert report.mean_summary_fidelity is None


def test_eval_pins_demo_mode_per_request_even_when_app_boots_live(monkeypatch):
    """ADR-0008: mean F1 below 1.0 must signal a regression, never the boot mode.

    The harness pins Demo mode on every request — the app's settings (pinned
    to Live, keyless, here) are never touched and never consulted for the
    dispatched mode.
    """
    import src.main as main
    from src.config import Settings
    from src.models import Mode

    monkeypatch.setattr(main, "settings", Settings(regula_mode=Mode.live, openrouter_api_key=None))
    report = run_eval()
    assert report.mean_f1 == 1.0
    assert main.settings.regula_mode == Mode.live


def test_scenario_description_and_mode_reach_the_analysis_request():
    """Live cases describe their own Scenario; the shared scoring loop must
    pass the description through so Live mode can ground the Planner on it,
    and pin the run's mode onto every request (ADR-0008)."""
    from src.models import Mode

    captured = []

    def respond(request):
        captured.append(request)
        return AnalyzeResponse(
            answer=Answer(findings=[], actions=[], citations=[]),
            trace=Trace(workflow="test", summary="nothing produced"),
        )

    evaluate_scenarios(
        [
            EvalScenario(
                id="described",
                scenario_id="some-scenario",
                description="A bank reliant on a cloud provider.",
                question="What applies?",
                expected=[],
            )
        ],
        respond,
        mode=Mode.live,
    )

    assert captured[0].scenario.id == "some-scenario"
    assert captured[0].scenario.description == "A bank reliant on a cloud provider."
    assert captured[0].mode == Mode.live


# --- The audit dump ---


def _summary(target: ProvisionTarget, relevance: str, strength: Strength | None = None) -> GroundedSummary:
    """One gate-shaped summaries entry: a rated target always arrives with
    its relevance, as it does through the Summarizer's gate."""
    return GroundedSummary(target=target, relevance=relevance, strength=strength)


def _respond_with(*findings: Finding, summaries_by_target=None):
    def respond(_request):
        return AnalyzeResponse(
            answer=Answer(
                findings=list(findings),
                actions=[],
                citations=answer_citations(list(findings), summaries_by_target),
            ),
            trace=Trace(workflow="fake", summary="canned"),
        )

    return respond


def test_per_case_result_carries_the_produced_versus_expected_dump():
    """The report rides the diagnostic material for the human audit: both
    sides' statements, the authored labels on the expected side with any
    authored relevance and the operator's rating, and each produced
    Citation's structural target with its quote, its Provision relevance, and
    the rated strength the Answer badges. Finding-level Strengths ride the
    produced side's Findings only (#58); the rated centrality rides the
    Citations on both sides (ADR-0011)."""
    scenario = EvalScenario(
        id="case",
        scenario_id="some-scenario",
        question="What applies?",
        expected=[ExpectedFinding(
            statement="expected statement",
            citations=[{
                "source_id": "gdpr",
                "provision": "Recital 71",
                "relevance": "Recital 71 frames the profiling the loan scoring performs.",
                "strength": Strength.strong,
            }],
        )],
    )
    report = evaluate_scenarios(
        [scenario],
        _respond_with(
            Finding(
                statement="produced wording",
                strength=Strength.weak,
                citations=[Citation.model_validate({"source_id": "gdpr", "recital_number": 71, "quote": "snip"})],
            ),
            summaries_by_target={
                _target(71, kind="recital"): _summary(
                    _target(71, kind="recital"), "produced relevance", Strength.weak
                ),
            },
        ),
        mode=Mode.demo,
    )

    result = report.scenarios[0]
    assert result.id == "case"
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0
    assert result.expected == [{
        "statement": "expected statement",
        "citations": [{
            "label": "gdpr Recital 71",
            "relevance": "Recital 71 frames the profiling the loan scoring performs.",
            "strength": "strong",
        }],
    }]
    assert result.produced == [{
        "statement": "produced wording",
        "strength": "weak",
        "citations": [{
            "target": "gdpr Recital 71",
            "quote": "snip",
            "relevance": "produced relevance",
            "strength": "weak",
        }],
    }]


def test_an_unrated_provision_dumps_no_strength():
    """A provision the Summarizer left unrated carries ``None`` in the dump —
    the audit sees the gap, never a default (ADR-0011)."""
    scenario = EvalScenario(
        id="case",
        scenario_id="some-scenario",
        question="What applies?",
        expected=[_expected_finding("Article 22")],
    )
    report = evaluate_scenarios(
        [scenario],
        _respond_with(_produced_finding(22)),
        mode=Mode.demo,
    )
    (produced,) = report.scenarios[0].produced
    assert produced["citations"][0]["strength"] is None


# --- The two components at the evaluation loop (ADR-0010, issue #50) ---


def _target(number: int, kind: str = "article", source_id: str = "gdpr") -> ProvisionTarget:
    return Citation.model_validate({"source_id": source_id, f"{kind}_number": number}).provision_target


def _expected_finding(label: str, relevance: str | None = None, strength: Strength = Strength.strong):
    citation: ExpectedCitation = {"source_id": "gdpr", "provision": label, "strength": strength}
    if relevance is not None:
        citation["relevance"] = relevance
    return ExpectedFinding(statement=f"expected: {label}", citations=[citation])


def _produced_finding(number: int, strength: Strength = Strength.strong, source_id: str = "gdpr", kind: str = "article"):
    return Finding(
        statement=f"produced: {source_id} {kind} {number}",
        strength=strength,
        citations=[Citation.model_validate({"source_id": source_id, f"{kind}_number": number})],
    )


def test_the_judge_scores_one_batched_pair_per_aligned_provision():
    """Expected and produced relevance align by provision target; the judge
    reads one pair per shared target with both statements verbatim."""
    target22 = _target(22)
    judge = ScriptedJudge({"P1": FULL_AGREEMENT_VERDICT})
    evaluate_scenarios(
        [EvalScenario(
            id="case",
            scenario_id="s",
            question="What applies?",
            expected=[_expected_finding("Article 22", relevance="expected relevance")],
        )],
        _respond_with(
            _produced_finding(22),
            summaries_by_target={target22: _summary(target22, "produced relevance")},
        ),
        mode=Mode.demo,
        judge=judge,
    )
    (call,) = judge.calls
    (pair,) = call
    assert pair.ref == "P1"
    assert pair.provision == "gdpr Article 22"
    assert pair.expected == "expected relevance"
    assert pair.produced == "produced relevance"


@pytest.mark.parametrize("verdict,expected_fidelity", [
    (FULL_AGREEMENT_VERDICT, 1.0),
    (ROLE_MISMATCH_VERDICT, 0.5),
    (POLARITY_FLIP_VERDICT, 0.5),
    (CONTRADICTION_VERDICT, 0.0),
])
def test_rubric_edges_land_in_the_case_score(verdict, expected_fidelity):
    """The rubric edges the acceptance criteria name, end to end through the
    evaluation loop: full agreement, role mismatch, polarity flip, and the
    contradiction floor."""
    target22 = _target(22)
    report = evaluate_scenarios(
        [EvalScenario(
            id="case",
            scenario_id="s",
            question="What applies?",
            expected=[_expected_finding("Article 22", relevance="expected relevance")],
        )],
        _respond_with(
            _produced_finding(22),
            summaries_by_target={target22: _summary(target22, "produced relevance")},
        ),
        mode=Mode.demo,
        judge=ScriptedJudge({"P1": verdict}),
    )
    assert report.scenarios[0].summary_fidelity == expected_fidelity


def test_summary_fidelity_means_over_all_judged_pairs():
    target22 = _target(22)
    target25 = _target(25)
    report = evaluate_scenarios(
        [EvalScenario(
            id="case",
            scenario_id="s",
            question="What applies?",
            expected=[
                _expected_finding("Article 22", relevance="expected one"),
                _expected_finding("Article 25", relevance="expected two"),
            ],
        )],
        _respond_with(
            _produced_finding(22),
            _produced_finding(25),
            summaries_by_target={
                target22: _summary(target22, "produced one"),
                target25: _summary(target25, "produced two"),
            },
        ),
        mode=Mode.demo,
        judge=ScriptedJudge({
            "P1": FULL_AGREEMENT_VERDICT,
            "P2": CONTRADICTION_VERDICT.model_copy(update={"ref": "P2"}),
        }),
    )
    assert report.scenarios[0].summary_fidelity == pytest.approx(0.5)


def test_a_cited_provision_without_a_summary_scores_zero_without_a_judge_call():
    """Ground truth summarizes a provision the Answer cites but the
    Summarizer left bare: maximally infidelitous, and nothing for a judge to
    compare — the zero is deterministic application code."""
    judge = ScriptedJudge({})
    report = evaluate_scenarios(
        [EvalScenario(
            id="case",
            scenario_id="s",
            question="What applies?",
            expected=[_expected_finding("Article 22", relevance="expected relevance")],
        )],
        _respond_with(_produced_finding(22)),
        mode=Mode.demo,
        judge=judge,
    )
    assert report.scenarios[0].summary_fidelity == 0.0
    assert judge.calls == []


def test_summary_fidelity_is_unmeasured_without_expected_summaries():
    """No authored relevance (the state until #51 lands) — no fidelity number,
    and the judge is never woken for nothing."""
    judge = ScriptedJudge({})
    report = evaluate_scenarios(
        [EvalScenario(
            id="case",
            scenario_id="s",
            question="What applies?",
            expected=[_expected_finding("Article 22")],
        )],
        _respond_with(
            _produced_finding(22),
            summaries_by_target={_target(22): _summary(_target(22), "produced")},
        ),
        mode=Mode.demo,
        judge=judge,
    )
    assert report.scenarios[0].summary_fidelity is None
    assert judge.calls == []


def test_summary_fidelity_is_unmeasured_without_a_judge():
    report = evaluate_scenarios(
        [EvalScenario(
            id="case",
            scenario_id="s",
            question="What applies?",
            expected=[_expected_finding("Article 22", relevance="expected relevance")],
        )],
        _respond_with(
            _produced_finding(22),
            summaries_by_target={_target(22): _summary(_target(22), "produced")},
        ),
        mode=Mode.demo,
    )
    assert report.scenarios[0].summary_fidelity is None


def test_fidelity_pairs_only_provisions_both_sides_touch():
    """An off-target produced citation is coverage's pessimism, never a
    fidelity pair; an expected target the Answer never cites is a coverage
    miss, not a fidelity one."""
    target22 = _target(22)
    judge = ScriptedJudge({"P1": FULL_AGREEMENT_VERDICT})
    report = evaluate_scenarios(
        [EvalScenario(
            id="case",
            scenario_id="s",
            question="What applies?",
            expected=[
                _expected_finding("Article 22", relevance="expected one"),
                _expected_finding("Article 25", relevance="never produced"),
            ],
        )],
        _respond_with(
            _produced_finding(22),
            _produced_finding(99),
            summaries_by_target={target22: _summary(target22, "produced one")},
        ),
        mode=Mode.demo,
        judge=judge,
    )
    (call,) = judge.calls
    assert [pair.ref for pair in call] == ["P1"]
    assert report.scenarios[0].summary_fidelity == 1.0


def test_the_report_carries_the_components_separately_and_never_blends_them():
    """ADR-0010: per-case numbers and an aggregate mean per component — no
    field blends them into one opaque score, and no strength-agreement
    metric rides the report at all."""
    import dataclasses

    report = evaluate_scenarios(
        [EvalScenario(
            id="case",
            scenario_id="s",
            question="What applies?",
            expected=[_expected_finding("Article 22", relevance="expected relevance")],
        )],
        _respond_with(
            _produced_finding(22),
            summaries_by_target={_target(22): _summary(_target(22), "produced", Strength.strong)},
        ),
        mode=Mode.demo,
        judge=ScriptedJudge({"P1": FULL_AGREEMENT_VERDICT}),
    )
    assert set(f.name for f in dataclasses.fields(report)) == {
        "scenarios", "mean_f1", "mean_summary_fidelity",
    }
    assert set(f.name for f in dataclasses.fields(report.scenarios[0])) == {
        "id", "precision", "recall", "f1", "expected", "produced",
        "summary_fidelity",
    }
    assert report.mean_f1 == 1.0
    assert report.mean_summary_fidelity == 1.0


def test_aggregate_component_means_skip_unmeasured_cases():
    """A mean over the cases where a component was measured; a component
    measured nowhere aggregates to no number, never a fake zero."""
    target22 = _target(22)
    report = evaluate_scenarios(
        [
            EvalScenario(
                id="measured",
                scenario_id="s",
                question="What applies?",
                expected=[_expected_finding("Article 22", relevance="expected relevance")],
            ),
            EvalScenario(
                id="unmeasured",
                scenario_id="s2",
                question="What applies?",
                expected=[_expected_finding("Article 22")],
            ),
        ],
        _respond_with(
            _produced_finding(22),
            summaries_by_target={target22: _summary(target22, "produced")},
        ),
        mode=Mode.demo,
        judge=ScriptedJudge({"P1": FULL_AGREEMENT_VERDICT}),
    )
    assert report.scenarios[0].summary_fidelity == 1.0
    assert report.scenarios[1].summary_fidelity is None
    assert report.mean_summary_fidelity == 1.0
