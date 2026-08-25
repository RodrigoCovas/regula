"""Live eval runner (spec #9, ticket #20): the operator quality command.

The runner drives the curated Live cases through /api/analyze in Live mode
and scores the Answers with the semantic matcher plus citation fidelity.
These tests wire the established seams — provider fakes at the composition
root, the patched availability probe — so the suite stays fully offline:
no OpenRouter, no Ollama, no PostgreSQL. The command itself is
operator-run and never part of CI.
"""

import pytest
from pydantic import SecretStr

from src.config import Mode, Settings
from src.eval_harness import LIVE_EVAL_SCENARIOS, STRENGTH_WEIGHTS, EvalScenarioResult
from src.live_eval import LiveEvalRefused, main, run_live_eval
from src.live_workflow import DraftClaim, DraftClaims, Plan, ResearchTarget, Verdict, Verdicts
from src.models import ProvisionKind, Strength

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


def _paraphrasing_llm() -> ScriptedLlm:
    """One strong Finding restating the top expectation in different words,
    citing exactly the provisions its ground truth names."""
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
    llm, retriever = _paraphrasing_llm(), _on_target_retriever()
    install_fake_pipeline(llm, retriever)

    report = run_live_eval(live_settings(query_log_path))

    assert [result.id for result in report.scenarios] == [case.id for case in LIVE_EVAL_SCENARIOS]
    for result in report.scenarios:
        assert isinstance(result, EvalScenarioResult)
        # The paraphrase paired through the semantic matcher with on-target
        # Citations: full precision, partial recall against nine expectations.
        assert result.precision == 1.0
        assert 0.0 < result.recall < 1.0
        assert 0.0 < result.f1 < 1.0
    assert report.mean_f1 == pytest.approx(sum(r.f1 for r in report.scenarios) / len(report.scenarios))
    # The retrieval tool really ran — Demo mode never touches the Retriever.
    assert retriever.queries
    assert llm.calls


def test_semantic_matching_and_citation_fidelity_are_applied(ingested_store, query_log_path):
    """The produced Finding is a paraphrase (verbatim matching would pair
    nothing) whose Citations hit their expected structural targets."""
    install_fake_pipeline(_paraphrasing_llm(), _on_target_retriever())

    report = run_live_eval(live_settings(query_log_path))

    expected_weight_total = sum(STRENGTH_WEIGHTS[f.strength] for f in LIVE_EVAL_SCENARIOS[0].expected)
    first = report.scenarios[0]
    # One strong Finding matched with full citation fidelity: recall is its
    # share of the expected weight — nonzero only because the semantic
    # matcher paired the paraphrase.
    assert first.recall == pytest.approx(STRENGTH_WEIGHTS[Strength.strong] / expected_weight_total)


def test_pins_live_mode_and_restores_app_settings(ingested_store, query_log_path):
    """The run forces Live dispatch even when the app booted Demo, and the
    app's settings come back exactly as they were — the run_eval contract,
    mirrored."""
    import src.main as main_module

    install_fake_pipeline(_paraphrasing_llm(), _on_target_retriever())
    before = main_module.settings

    run_live_eval(live_settings(query_log_path))

    assert main_module.settings is before
    assert main_module.settings.regula_mode == Mode.demo


def test_not_available_mid_run_aborts_instead_of_scoring_zeros(monkeypatch, query_log_path):
    """A Not-available response mid-run means infrastructure broke after the
    pre-run checks: scoring it would print silent zeros dressed as quality."""
    import src.availability as availability
    import src.main as main_module

    install_fake_pipeline(_paraphrasing_llm(), _on_target_retriever())
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
    monkeypatch.setattr("src.live_eval._source_local_env", lambda: None)
    install_fake_pipeline(_paraphrasing_llm(), _on_target_retriever())

    exit_code = main()

    assert exit_code == 0
    out = capsys.readouterr().out
    for case in LIVE_EVAL_SCENARIOS:
        assert case.id in out
    assert "F1" in out
    assert "mean f1" in out.lower()


def test_main_refusal_exits_nonzero_with_cause_on_stderr(monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("src.live_eval._source_local_env", lambda: None)

    exit_code = main()

    assert exit_code == 1
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err
