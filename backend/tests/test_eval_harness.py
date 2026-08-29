"""Suite-level pins for the eval harness: the Demo tripwire, the
per-request mode contract, and the audit dump that rides each case's
result. The scoring arithmetic itself is covered at the pure seam in
test_provision_coverage.py (ADR-0010)."""

from src.eval_harness import (
    CURATED_SCENARIOS,
    EvalScenario,
    ExpectedFinding,
    evaluate_scenarios,
    run_eval,
)
from src.models import AnalyzeResponse, Answer, Citation, Finding, Mode, Strength, Trace


def test_curated_scenario_count_within_spec():
    assert 6 <= len(CURATED_SCENARIOS) <= 8


def test_deterministic_demo_scores_perfect_on_all_curated_scenarios():
    report = run_eval()
    assert len(report.scenarios) == len(CURATED_SCENARIOS)
    for scenario in report.scenarios:
        assert scenario.f1 == 1.0, f"scenario {scenario.id} scored {scenario.f1}"
    assert report.mean_f1 == 1.0


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


def _respond_with(*findings: Finding):
    def respond(_request):
        return AnalyzeResponse(
            answer=Answer(findings=list(findings), actions=[], citations=[]),
            trace=Trace(workflow="fake", summary="canned"),
        )

    return respond


def test_per_case_result_carries_the_produced_versus_expected_dump():
    """The report rides the diagnostic material for the human audit: both
    sides' statements and Strengths, the authored labels on the expected side,
    and each produced Citation's structural target with its quote."""
    scenario = EvalScenario(
        id="case",
        scenario_id="some-scenario",
        question="What applies?",
        expected=[ExpectedFinding(
            statement="expected statement",
            strength=Strength.strong,
            citations=[{"source_id": "gdpr", "provision": "Recital 71"}],
        )],
    )
    report = evaluate_scenarios(
        [scenario],
        _respond_with(Finding(
            statement="produced wording",
            strength=Strength.weak,
            citations=[Citation.model_validate({"source_id": "gdpr", "recital_number": 71, "quote": "snip"})],
        )),
        mode=Mode.demo,
    )

    result = report.scenarios[0]
    assert result.id == "case"
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0
    assert result.expected == [{
        "statement": "expected statement",
        "strength": "strong",
        "citations": ["gdpr Recital 71"],
    }]
    assert result.produced == [{
        "statement": "produced wording",
        "strength": "weak",
        "citations": [{"target": "gdpr Recital 71", "quote": "snip"}],
    }]
