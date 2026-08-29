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
    EvalScenarioResult,
    expected_target_weights,
)
from src.live_eval import LiveEvalRefused, main, run_live_eval
from src.live_workflow import DraftClaim, DraftClaims, Plan, ResearchTarget, Verdict, Verdicts
from src.models import Mode, ProvisionKind, Strength

from conftest import install_fake_pipeline
from fakes import FakeRetriever, ScriptedLlm, make_chunk


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


# --- The operator CLI ---


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
    """--output lands as JSON: the run's metadata, per-case coverage scores,
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
    assert set(first) == {"id", "precision", "recall", "f1", "expected", "produced"}
    assert first["expected"] and first["produced"]
    # Both dump sides speak provisions: authored labels vs structural targets.
    assert all("gdpr" in c or "ai-act" in c or "dora" in c for c in first["expected"][0]["citations"])
    assert all("target" in c for c in first["produced"][0]["citations"])


def test_refusal_writes_no_artifact(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    out_path = tmp_path / "live-eval.json"

    exit_code = main(["--output", str(out_path)])

    assert exit_code == 1
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err
    assert not out_path.exists()
