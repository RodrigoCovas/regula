"""Live eval runner (spec #9, ticket #20): the operator quality command.

The runner drives the curated Live cases through /api/analyze in Live mode
and scores the Answers with provision coverage (ADR-0010). With --output it
writes a per-case JSON artifact: component scores plus the
produced-versus-expected dump for the human audit. These tests wire the
established seams — provider fakes at the composition root, the patched
availability probe — so the suite stays fully offline: no OpenRouter, no
Ollama, no PostgreSQL. The command itself is operator-run and never part
of CI.
"""

import json

import pytest
from pydantic import SecretStr

from src.config import Settings
from src.eval_harness import (
    LIVE_EVAL_SCENARIOS,
    STRENGTH_WEIGHTS,
    EvalReport,
    EvalScenario,
    EvalScenarioResult,
    ExpectedFinding,
    expected_target_weights,
)
from src.llm import LlmError, LlmUnreachableError, OpenRouterClient
from src.live_eval import LiveEvalRefused, main, report_artifact, run_live_eval
from src.live_workflow import DraftClaim, DraftClaims, Plan, ProvisionSummary, ResearchTarget, Summaries, Verdict, Verdicts
from src.models import Mode, ProvisionKind, Strength

from conftest import install_fake_pipeline
from fakes import (
    CONTRADICTION_VERDICT,
    FULL_AGREEMENT_VERDICT,
    FakeRetriever,
    ScriptedJudge,
    ScriptedLlm,
    make_chunk,
)


def live_settings(query_log_path) -> Settings:
    """Operator settings for one test run; the query log stays hermetic."""
    return Settings(
        regula_mode=Mode.demo,
        openrouter_api_key=SecretStr("sk-or-test"),
        query_log_path=str(query_log_path),
    )


@pytest.fixture()
def ingested_store(monkeypatch):
    """The precondition probe reports an ingested store for this test."""
    import src.availability as availability

    monkeypatch.setattr(availability, "stored_chunk_count", lambda _database_url: 42)


def _on_target_llm() -> ScriptedLlm:
    """One strong Finding citing exactly the provisions its ground truth
    names. Wording is irrelevant to coverage scoring (ADR-0010), so the
    statement differs from the expectation's — an audit would still pair
    them by eye through the dump."""
    statement = (
        "AI systems that evaluate the creditworthiness of natural persons or "
        "establish their credit score are high-risk AI systems under the AI Act."
    )
    return ScriptedLlm(
        plan=Plan(targets=[ResearchTarget(query="creditworthiness evaluation high-risk")]),
        claims=DraftClaims(claims=[DraftClaim(statement=statement, evidence_refs=["E1", "E2"])]),
        verdicts=Verdicts(verdicts=[
            Verdict(statement=statement, supported=True, strength=Strength.strong, evidence_refs=["E1", "E2"])
        ]),
    )


def _on_target_retriever() -> FakeRetriever:
    """Chunks whose provision metadata matches the ground truth's targets."""
    return FakeRetriever(chunks=[
        make_chunk(source_id="ai-act", kind=ProvisionKind.article, number=6),
        make_chunk(source_id="ai-act", kind=ProvisionKind.annex, number=3),
    ])


# --- Preconditions: the command refuses clearly instead of measuring garbage ---


def test_missing_key_refuses_naming_the_variable(query_log_path):
    settings = Settings(regula_mode=Mode.demo, openrouter_api_key=None, query_log_path=str(query_log_path))
    with pytest.raises(LiveEvalRefused, match="OPENROUTER_API_KEY"):
        run_live_eval(settings)


def test_empty_store_refuses_naming_the_ingest_command(monkeypatch, query_log_path):
    import src.availability as availability

    monkeypatch.setattr(availability, "stored_chunk_count", lambda _database_url: 0)
    with pytest.raises(LiveEvalRefused, match=r"backend\.src\.ingest"):
        run_live_eval(live_settings(query_log_path))


def test_unreachable_store_refuses_with_the_start_hint(monkeypatch, query_log_path):
    import src.availability as availability

    def unreachable(_database_url):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(availability, "stored_chunk_count", unreachable)
    with pytest.raises(LiveEvalRefused, match=r"docker compose up -d postgres"):
        run_live_eval(live_settings(query_log_path))


# --- The measured run ---


def test_scores_every_live_case_through_the_live_pipeline(ingested_store, query_log_path):
    llm, retriever = _on_target_llm(), _on_target_retriever()
    install_fake_pipeline(llm, retriever)

    report = run_live_eval(live_settings(query_log_path))

    assert [result.id for result in report.scenarios] == [case.id for case in LIVE_EVAL_SCENARIOS]
    first = report.scenarios[0]
    assert isinstance(first, EvalScenarioResult)
    # On the canonical case the produced Citations hit exactly the two expected
    # targets and nothing else: full precision, partial recall. The other
    # cases' expectations describe different scenarios, so this one canned
    # Finding simply goes uncovered there — per-case scoring is what matters.
    assert first.precision == 1.0
    assert 0.0 < first.recall < 1.0
    assert 0.0 < first.f1 < 1.0
    assert report.mean_f1 == pytest.approx(sum(r.f1 for r in report.scenarios) / len(report.scenarios))
    # The audit dump rides the report for the human review.
    assert first.expected and first.produced
    # The retrieval tool really ran — Demo mode never touches the Retriever.
    assert retriever.queries
    assert llm.calls


def test_every_live_case_expectation_targets_a_known_provision():
    """Hand-authored labels are parsed at scoring time, so a typo would fail
    mid-run; this makes the same authoring bug fail loudly here instead."""
    from src.eval_harness import parse_provision

    for case in LIVE_EVAL_SCENARIOS:
        assert case.expected, f"live case {case.id} ships no expectations"
        for finding in case.expected:
            assert finding.citations, f"{case.id}: expectation without Citations"
            for citation in finding.citations:
                target = parse_provision(citation["source_id"], citation["provision"])
                assert target.source_id in {"gdpr", "ai-act", "dora"}
                assert target.number > 0


def test_live_case_ids_are_unique():
    ids = [case.id for case in LIVE_EVAL_SCENARIOS]
    assert len(ids) == len(set(ids))


def test_coverage_recall_is_strength_weighted_over_expected_targets(ingested_store, query_log_path):
    """The produced Citations hit the two targets of the first (strong)
    expectation: recall is their combined weight over the max-rule weighted
    total of the case's expected targets."""
    install_fake_pipeline(_on_target_llm(), _on_target_retriever())

    report = run_live_eval(live_settings(query_log_path))

    first = report.scenarios[0]
    hit_weight = STRENGTH_WEIGHTS[Strength.strong] * 2  # Article 6(2) + Annex III point 4(a)
    total_weight = sum(expected_target_weights(LIVE_EVAL_SCENARIOS[0].expected).values())
    assert first.recall == pytest.approx(hit_weight / total_weight)


def test_pins_live_mode_per_request_even_when_the_app_boots_demo(ingested_store, query_log_path):
    """ADR-0008: the run pins Live mode per request — every query-log record
    carries mode 'live' even though the app booted Demo — and the app's
    settings come back exactly as they were afterwards."""
    import src.main as main_module

    install_fake_pipeline(_on_target_llm(), _on_target_retriever())
    before = main_module.settings

    run_live_eval(live_settings(query_log_path))

    assert main_module.settings is before
    assert main_module.settings.regula_mode == Mode.demo
    records = [
        json.loads(line) for line in query_log_path.read_text().splitlines() if line.strip()
    ]
    assert records, "every request must append its observability record"
    assert all(record["mode"] == "live" for record in records)


def test_not_available_mid_run_aborts_instead_of_scoring_zeros(monkeypatch, query_log_path):
    """A Not-available response mid-run means infrastructure broke after the
    pre-run checks: scoring it would print silent zeros dressed as quality."""
    import src.availability as availability
    import src.main as main_module

    install_fake_pipeline(_on_target_llm(), _on_target_retriever())
    # The store passes the pre-run probe, then empties before the requests.
    monkeypatch.setattr(availability, "stored_chunk_count", lambda _database_url: 42)
    monkeypatch.setattr(main_module, "vector_store_is_empty", lambda _database_url: True)

    with pytest.raises(LiveEvalRefused, match="Not-available"):
        run_live_eval(live_settings(query_log_path))


# --- The judge components (ADR-0010, issue #50) ---


def _on_target_llm_with_summaries() -> ScriptedLlm:
    """The on-target pipeline plus the Summarizer's reply: one grounded
    relevance statement per cited provision (P1: Article 6, P2: Annex III)."""
    return ScriptedLlm(
        plan=Plan(targets=[ResearchTarget(query="creditworthiness evaluation high-risk")]),
        claims=DraftClaims(claims=[DraftClaim(
            statement="AI systems that evaluate the creditworthiness of natural persons are high-risk.",
            evidence_refs=["E1", "E2"],
        )]),
        verdicts=Verdicts(verdicts=[
            Verdict(
                statement="AI systems that evaluate the creditworthiness of natural persons are high-risk.",
                supported=True,
                strength=Strength.strong,
                evidence_refs=["E1", "E2"],
            )
        ]),
        summaries=Summaries(summaries=[
            ProvisionSummary(ref="P1", relevance="produced relevance for Article 6"),
            ProvisionSummary(ref="P2", relevance="produced relevance for Annex III"),
        ]),
    )


def _comparable_case() -> EvalScenario:
    """A Live case whose ground truth summarizes the provisions it cites —
    the shape #51 will author; here it is test data, never shipped labels."""
    return EvalScenario(
        id="judged-case",
        scenario_id="some-live-scenario",
        description="A company uses an AI system to score loan applicants.",
        question="What applies?",
        expected=[ExpectedFinding(
            statement="expected statement",
            strength=Strength.strong,
            citations=[
                {"source_id": "ai-act", "provision": "Article 6", "relevance": "expected relevance for Article 6"},
                {"source_id": "ai-act", "provision": "Annex III point 5(b)", "relevance": "expected relevance for Annex III"},
            ],
        )],
    )


def test_summary_fidelity_reaches_the_report_through_a_scripted_judge(ingested_store, query_log_path):
    judge = ScriptedJudge({
        "P1": FULL_AGREEMENT_VERDICT,
        "P2": CONTRADICTION_VERDICT.model_copy(update={"ref": "P2"}),
    })
    install_fake_pipeline(_on_target_llm_with_summaries(), _on_target_retriever())

    report = run_live_eval(live_settings(query_log_path), judge=judge, scenarios=[_comparable_case()])

    # One batched judge call for the case; both aligned pairs took part, and
    # the contradiction floored its half of the mean.
    assert len(judge.calls) == 1
    assert [pair.ref for pair in judge.calls[0]] == ["P1", "P2"]
    assert report.scenarios[0].summary_fidelity == pytest.approx((1.0 + 0.0) / 2)
    # The strength components ride the same case: max-rule on both sides.
    assert report.scenarios[0].strength_agreement == 1.0
    assert report.mean_summary_fidelity == pytest.approx(0.5)
    assert report.mean_strength_agreement == 1.0


def test_the_default_judge_is_built_from_the_configured_provider(monkeypatch, ingested_store, query_log_path):
    """ADR-0009: the judge reads the same environment configuration as the
    workflow — model, base URL, and the unwrapped key — through the one
    chat_client recipe both share."""
    import src.config as config

    install_fake_pipeline(_on_target_llm(), _on_target_retriever())
    captured: dict = {}
    real_client = OpenRouterClient

    def recording_client(*args, **kwargs):
        captured.update(kwargs)
        captured["args"] = args
        return real_client(*args, **kwargs)

    monkeypatch.setattr(config, "OpenRouterClient", recording_client)
    settings = live_settings(query_log_path)
    run_live_eval(settings, scenarios=[EvalScenario(
        id="probe", scenario_id="probe-scenario", question="What applies?", expected=[],
    )])
    assert captured["api_key"] == "sk-or-test"
    assert captured["model"] == settings.llm_model
    assert captured["base_url"] == settings.llm_base_url


def test_main_aborts_clearly_when_the_judge_cannot_be_reached(monkeypatch, capsys, tmp_path):
    """An unreachable judge is infrastructure failure, never measured zeros:
    the run aborts with the provider detail and writes no artifact."""
    import src.availability as availability
    import src.live_eval as live_eval

    monkeypatch.setattr(availability, "stored_chunk_count", lambda _database_url: 42)
    monkeypatch.setenv("REGULA_MODE", "demo")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    install_fake_pipeline(_on_target_llm_with_summaries(), _on_target_retriever())
    monkeypatch.setattr(live_eval, "LIVE_EVAL_SCENARIOS", [_comparable_case()])

    class UnreachableClient:
        def __init__(self, *args, **kwargs):
            pass

        def complete(self, *args, **kwargs):
            raise LlmUnreachableError("Could not reach the LLM provider at https://openrouter.ai: refused")

    import src.config as config

    monkeypatch.setattr(config, "OpenRouterClient", UnreachableClient)
    out_path = tmp_path / "live-eval.json"

    exit_code = main(["--output", str(out_path)])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "judge" in err.lower()
    assert "Could not reach the LLM provider" in err
    assert not out_path.exists()


def test_main_aborts_clearly_when_the_judge_reply_cannot_be_trusted(monkeypatch, capsys, tmp_path):
    """A schema-invalid or mis-keyed judge verdict aborts the run too — a
    verdict that cannot be validated must never dress silence as a score."""
    import src.availability as availability
    import src.live_eval as live_eval

    monkeypatch.setattr(availability, "stored_chunk_count", lambda _database_url: 42)
    monkeypatch.setenv("REGULA_MODE", "demo")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    install_fake_pipeline(_on_target_llm_with_summaries(), _on_target_retriever())
    monkeypatch.setattr(live_eval, "LIVE_EVAL_SCENARIOS", [_comparable_case()])

    class MisbehavingJudge:
        def __init__(self, llm):
            pass

        def compare(self, pairs):
            raise LlmError("The judge returned no verdict for ref(s) ['P1']")

    monkeypatch.setattr(live_eval, "SummaryFidelityJudge", MisbehavingJudge)

    exit_code = main([])

    assert exit_code == 1
    assert "judge" in capsys.readouterr().err.lower()


def test_main_prints_all_three_components_with_their_aggregates(monkeypatch, capsys):
    import src.availability as availability

    monkeypatch.setattr(availability, "stored_chunk_count", lambda _database_url: 42)
    monkeypatch.setenv("REGULA_MODE", "demo")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    install_fake_pipeline(_on_target_llm(), _on_target_retriever())

    exit_code = main([])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "coverage precision" in out
    assert "summary fidelity" in out
    assert "strength agreement" in out
    assert "mean coverage" in out.lower()
    assert "mean summary fidelity" in out.lower()
    assert "mean strength agreement" in out.lower()
    # No judge and no authored summaries: the unmeasured component reads as
    # n/a, never as a fake zero.
    assert "n/a" in out


# --- The operator CLI ---


def test_report_artifact_shape_at_the_pure_seam():
    """The CLI's JSON artifact is a pure function of the report: run metadata
    over the per-case component scores plus the produced-versus-expected dump."""
    report = EvalReport(
        scenarios=[EvalScenarioResult(
            id="case",
            precision=1.0,
            recall=0.5,
            f1=0.666666,
            expected=[{"statement": "expected", "strength": "strong", "citations": [{"label": "gdpr Article 22", "relevance": "expected relevance"}]}],
            produced=[{"statement": "produced", "strength": "weak", "citations": [{"target": "gdpr Article 22", "quote": None, "relevance": "produced relevance"}]}],
            summary_fidelity=0.75,
            strength_agreement=1.0,
        )],
        mean_f1=0.666666,
        mean_summary_fidelity=0.75,
        mean_strength_agreement=1.0,
    )

    artifact = report_artifact(report)

    assert artifact["mode"] == "live"
    assert artifact["generated_at"]
    assert artifact["mean_f1"] == pytest.approx(0.666666)
    assert artifact["mean_summary_fidelity"] == pytest.approx(0.75)
    assert artifact["mean_strength_agreement"] == pytest.approx(1.0)
    (case,) = artifact["scenarios"]
    assert set(case) == {
        "id", "precision", "recall", "f1", "expected", "produced",
        "summary_fidelity", "strength_agreement",
    }


def test_main_prints_per_case_scores_plus_aggregate(monkeypatch, capsys):
    import src.availability as availability

    monkeypatch.setattr(availability, "stored_chunk_count", lambda _database_url: 42)
    monkeypatch.setenv("REGULA_MODE", "demo")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    install_fake_pipeline(_on_target_llm(), _on_target_retriever())

    exit_code = main([])

    assert exit_code == 0
    out = capsys.readouterr().out
    for case in LIVE_EVAL_SCENARIOS:
        assert case.id in out
    assert "coverage precision" in out
    assert "mean coverage f1" in out.lower()


def test_main_writes_the_report_artifact_when_given_an_output_path(monkeypatch, capsys, tmp_path):
    """--output lands as JSON: the run's metadata, per-case component scores,
    and the produced-versus-expected dump — everything the audit quotes."""
    import src.availability as availability

    monkeypatch.setattr(availability, "stored_chunk_count", lambda _database_url: 42)
    monkeypatch.setenv("REGULA_MODE", "demo")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    install_fake_pipeline(_on_target_llm(), _on_target_retriever())
    out_path = tmp_path / "reports" / "live-eval.json"

    exit_code = main(["--output", str(out_path)])

    assert exit_code == 0
    assert "Report artifact written to" in capsys.readouterr().out
    artifact = json.loads(out_path.read_text())
    assert artifact["mode"] == "live"
    assert artifact["generated_at"]
    assert artifact["mean_f1"] == pytest.approx(
        sum(case["f1"] for case in artifact["scenarios"]) / len(artifact["scenarios"])
    )
    first = artifact["scenarios"][0]
    assert first["id"] == LIVE_EVAL_SCENARIOS[0].id
    assert set(first) == {
        "id", "precision", "recall", "f1", "expected", "produced",
        "summary_fidelity", "strength_agreement",
    }
    assert first["expected"] and first["produced"]
    # No judge ran and #51 has not authored the relevance labels yet: the
    # fidelity component reports unmeasured, never a fake zero.
    assert first["summary_fidelity"] is None
    assert artifact["mean_summary_fidelity"] is None
    # Both dump sides speak provisions: authored labels vs structural targets.
    assert all(
        "gdpr" in citation["label"] or "ai-act" in citation["label"] or "dora" in citation["label"]
        for case in artifact["scenarios"]
        for entry in case["expected"]
        for citation in entry["citations"]
    )
    assert all("target" in c for c in first["produced"][0]["citations"])
    assert all("relevance" in c for c in first["produced"][0]["citations"])


def test_refusal_writes_no_artifact(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    out_path = tmp_path / "live-eval.json"

    exit_code = main(["--output", str(out_path)])

    assert exit_code == 1
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err
    assert not out_path.exists()
