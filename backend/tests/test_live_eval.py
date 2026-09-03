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

from conftest import ROOT, install_fake_pipeline
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


@pytest.fixture(autouse=True)
def _hermetic_checkpoints(tmp_path, monkeypatch):
    """The CLI always checkpoints (ADR-0013): redirect the checkpoint
    directory into tmp_path so tests never write into the repository."""
    import src.live_eval as live_eval

    monkeypatch.setattr(live_eval, "CHECKPOINT_DIR", tmp_path / "eval-runs")


# The one Finding the on-target pipeline produces, shared by every scripted
# variant: wording is irrelevant to coverage scoring (ADR-0010), so it need
# not mirror the expectation's text.
_ON_TARGET_STATEMENT = (
    "The company must notify the supervisory authority of the breach "
    "within seventy-two hours and inform affected customers where they "
    "face a high risk."
)


def _on_target_llm() -> ScriptedLlm:
    """One strong Finding citing exactly the provisions its ground truth
    names — an audit would still pair it by eye through the dump."""
    return ScriptedLlm(
        plan=Plan(targets=[ResearchTarget(query="breach notification duties")]),
        claims=DraftClaims(claims=[DraftClaim(statement=_ON_TARGET_STATEMENT, evidence_refs=["E1", "E2"])]),
        verdicts=Verdicts(verdicts=[
            Verdict(statement=_ON_TARGET_STATEMENT, supported=True, strength=Strength.strong, evidence_refs=["E1", "E2"])
        ]),
    )


def _on_target_retriever() -> FakeRetriever:
    """Chunks whose provision metadata matches the ground truth's targets."""
    return FakeRetriever(chunks=[
        make_chunk(source_id="gdpr", kind=ProvisionKind.article, number=33),
        make_chunk(source_id="gdpr", kind=ProvisionKind.article, number=34),
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
    # On the first case the produced Citations hit exactly the two expected
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


def test_shipped_ground_truth_transcribes_the_operators_citations_file():
    """#51's ground truth is transcribed verbatim from the operator's
    data/regulations/citations.json: every case's cited provisions with their
    relevance summaries and per-provision Strength ratings must match the
    file exactly (#56), so the operator's authored file stays the record and
    any transcription drift fails here instead of silently changing the
    judge's ground truth. Cases pair with scenario keys by order — the file
    numbers its scenarios the same way (scenario_1 first) — and a reorder or
    mismatch fails loudly.

    The projection is deliberately spelled out here instead of reusing a
    harness shape: the pin must not share an implementation with the ground
    truth it pins, or it could drift in step with the code it should
    contradict."""
    citations = json.loads((ROOT / "data" / "regulations" / "citations.json").read_text())
    assert len(LIVE_EVAL_SCENARIOS) == len(citations), "one case per citations.json scenario"

    for index, case in enumerate(LIVE_EVAL_SCENARIOS, start=1):
        key = f"scenario_{index}"
        shipped = [
            (c["source_id"], c["provision"], c["relevance"], c["strength"].value)
            for finding in case.expected
            for c in finding.citations
        ]
        authored = [
            (entry["source_id"], entry["provision"], entry["relevance"], entry["strength"])
            for entry in citations[key]
        ]
        assert sorted(shipped) == sorted(authored), f"{case.id} does not transcribe {key} verbatim"
        # Every shipped citation carries its authored relevance — the shape
        # the fidelity judge reads — with none left bare.
        assert all(relevance for _, _, relevance, _ in shipped), f"{case.id}: bare expected citation"


def test_coverage_recall_is_strength_weighted_over_expected_targets(ingested_store, query_log_path):
    """The produced Citations hit two strongly-weighted expected targets:
    recall is their combined weight over the max-rule weighted total of the
    case's expected targets."""
    install_fake_pipeline(_on_target_llm(), _on_target_retriever())

    report = run_live_eval(live_settings(query_log_path))

    first = report.scenarios[0]
    hit_weight = STRENGTH_WEIGHTS[Strength.strong] * 2  # gdpr Article 33 + Article 34
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
    relevance statement per cited provision (P1: Article 33, P2: Article 34)."""
    return ScriptedLlm(
        plan=Plan(targets=[ResearchTarget(query="breach notification duties")]),
        claims=DraftClaims(claims=[DraftClaim(
            statement=_ON_TARGET_STATEMENT,
            evidence_refs=["E1", "E2"],
        )]),
        verdicts=Verdicts(verdicts=[
            Verdict(
                statement=_ON_TARGET_STATEMENT,
                supported=True,
                strength=Strength.strong,
                evidence_refs=["E1", "E2"],
            )
        ]),
        summaries=Summaries(summaries=[
            ProvisionSummary(ref="P1", relevance="produced relevance for Article 33", strength=Strength.strong),
            ProvisionSummary(ref="P2", relevance="produced relevance for Article 34", strength=Strength.strong),
        ]),
    )


def _comparable_case() -> EvalScenario:
    """A Live case whose ground truth summarizes the provisions it cites —
    the shape #51 authored; here it is test data, never shipped labels."""
    return EvalScenario(
        id="judged-case",
        scenario_id="some-live-scenario",
        description="A company uses an AI system to score loan applicants.",
        question="What applies?",
        expected=[ExpectedFinding(
            statement="expected statement",
            citations=[
                {"source_id": "gdpr", "provision": "Article 33", "relevance": "expected relevance for Article 33", "strength": Strength.strong},
                {"source_id": "gdpr", "provision": "Article 34", "relevance": "expected relevance for Article 34", "strength": Strength.strong},
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
    # The strength component rides the same case: the operator's ratings
    # against the Summarizer's (ADR-0011) — here they agree.
    assert report.scenarios[0].strength_agreement == 1.0
    assert report.mean_summary_fidelity == pytest.approx(0.5)
    assert report.mean_strength_agreement == 1.0


def test_the_default_judge_is_built_from_the_configured_provider(monkeypatch, ingested_store, query_log_path):
    """ADR-0009: the judge reads the same environment configuration as the
    workflow — model, base URL, provider routing preference, and the
    unwrapped key — through the one chat_client recipe both share."""
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
    assert captured["provider"] == settings.llm_provider


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
    assert "llm call failed" in err.lower()
    # The judge names itself: the operator can tell which LLM call failed.
    assert "summary-fidelity judge" in err
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


def test_main_aborts_clearly_when_a_workflow_llm_call_fails(monkeypatch, capsys):
    """A provider glitch mid-run — a malformed planner reply, say — aborts the
    run like any infrastructure failure, and the message must not blame the
    summary-fidelity judge: the abort names the LLM call failure itself, with
    the provider detail verbatim, so the operator fixes the right thing."""
    import src.availability as availability

    monkeypatch.setattr(availability, "stored_chunk_count", lambda _database_url: 42)
    monkeypatch.setenv("REGULA_MODE", "demo")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    class GlitchingPlanner:
        def complete(self, system, user, schema):
            raise LlmError(
                "Model 'test-model-9' did not return parseable JSON for schema Plan: '{\"targets\":[]}]'"
            )

    install_fake_pipeline(GlitchingPlanner(), _on_target_retriever())

    exit_code = main([])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "Live eval aborted" in err
    assert "LLM call failed" in err
    assert "schema Plan" in err
    assert "judge" not in err.lower()


def test_main_prints_all_three_components_with_their_aggregates(monkeypatch, capsys):
    import src.availability as availability

    monkeypatch.setattr(availability, "stored_chunk_count", lambda _database_url: 42)
    monkeypatch.setenv("REGULA_MODE", "demo")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("LLM_MODEL", "test-model-9")
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
    # The operator output names the model that produced the numbers —
    # results are model-sensitive by design (ADR-0009).
    assert "Model: test-model-9" in out
    # No judge and no authored summaries: the unmeasured component reads as
    # n/a, never as a fake zero.
    assert "n/a" in out


# --- The operator CLI ---


def test_report_artifact_shape_at_the_pure_seam():
    """The CLI's JSON artifact is a pure function of the report: run metadata
    over the per-case component scores plus the produced-versus-expected dump.
    The metadata names the model that produced the numbers — eval results are
    model-sensitive by design (ADR-0009), so the artifact must say whose they
    are or a later reader cannot judge them."""
    report = EvalReport(
        scenarios=[EvalScenarioResult(
            id="case",
            precision=1.0,
            recall=0.5,
            f1=0.666666,
            expected=[{"statement": "expected", "citations": [{"label": "gdpr Article 22", "relevance": "expected relevance", "strength": "strong"}]}],
            produced=[{"statement": "produced", "strength": "weak", "citations": [{"target": "gdpr Article 22", "quote": None, "relevance": "produced relevance", "strength": None}]}],
            summary_fidelity=0.75,
            strength_agreement=1.0,
        )],
        mean_f1=0.666666,
        mean_summary_fidelity=0.75,
        mean_strength_agreement=1.0,
    )

    artifact = report_artifact(report, llm_model="upstage/solar-pro4")

    assert artifact["mode"] == "live"
    assert artifact["generated_at"]
    assert artifact["llm_model"] == "upstage/solar-pro4"
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
    monkeypatch.setenv("LLM_MODEL", "test-model-9")
    install_fake_pipeline(_on_target_llm(), _on_target_retriever())
    out_path = tmp_path / "reports" / "live-eval.json"

    exit_code = main(["--output", str(out_path)])

    assert exit_code == 0
    assert "Report artifact written to" in capsys.readouterr().out
    artifact = json.loads(out_path.read_text())
    assert artifact["mode"] == "live"
    assert artifact["generated_at"]
    # The artifact records the model that produced the numbers (ADR-0009):
    # the configured LLM_MODEL, not a hardcoded pin.
    assert artifact["llm_model"] == "test-model-9"
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
    # No judge ran, and the canned pipeline ships no relevance summaries for
    # the provisions it cites: the fidelity component scores the
    # deterministic floor — a bare citation is infidelitous by construction —
    # while every unmeasured case reports None, never a fake zero.
    assert first["summary_fidelity"] == pytest.approx(0.0)
    assert artifact["mean_summary_fidelity"] == pytest.approx(0.0)
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


# --- Checkpointed runs: resume by run id (ADR-0013) --------------------------------


def _bare_case(case_id: str) -> EvalScenario:
    """A minimal Live case whose ground truth carries no relevance summaries —
    the fidelity judge is never woken for it, so tests need no scripted judge."""
    return EvalScenario(
        id=case_id,
        scenario_id="some-live-scenario",
        description="A company uses an AI system to score loan applicants.",
        question="What applies?",
        expected=[ExpectedFinding(
            statement="expected statement",
            citations=[{"source_id": "gdpr", "provision": "Article 33", "strength": Strength.strong}],
        )],
    )


def test_checkpoint_records_each_completed_case(ingested_store, query_log_path, tmp_path):
    """With a run id the runner checkpoints: every completed case's full
    EvalScenarioResult lands on disk before the next case starts — a run that
    dies mid-flight never loses measured work (ADR-0013)."""
    install_fake_pipeline(_on_target_llm(), _on_target_retriever())
    checkpoint_dir = tmp_path / "eval-runs"

    run_live_eval(
        live_settings(query_log_path),
        run_id="test-run",
        checkpoint_dir=checkpoint_dir,
        scenarios=[_bare_case("first-case"), _bare_case("second-case")],
    )

    data = json.loads((checkpoint_dir / "test-run.json").read_text())
    assert data["run_id"] == "test-run"
    assert data["mode"] == "live"
    assert data["started_at"]
    assert data["llm_model"] == live_settings(query_log_path).llm_model
    assert [case["id"] for case in data["scenarios"]] == ["first-case", "second-case"]
    assert set(data["scenarios"][0]) == {
        "id", "precision", "recall", "f1", "expected", "produced",
        "summary_fidelity", "strength_agreement",
    }
    # Each completed case carries the hash of the definition that produced it —
    # the staleness guard a later resume reads (ADR-0013).
    assert set(data["scenario_hashes"]) == {"first-case", "second-case"}


def _checkpoint_document(tmp_path, cases, llm_model, hashes):
    """A hand-written checkpoint document for one run id, as the runner saves it."""
    document = {
        "run_id": "resumed-run",
        "mode": "live",
        "started_at": "2026-09-03T10:00:00+00:00",
        "llm_model": llm_model,
        "scenario_hashes": hashes,
        "scenarios": cases,
    }
    checkpoint_dir = tmp_path / "eval-runs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "resumed-run.json").write_text(json.dumps(document, indent=2) + "\n")
    return checkpoint_dir


def _completed_result(case_id):
    """One obviously-checkpointed result: numbers the on-target pipeline can
    never produce, so a substitution reads unambiguously in the report."""
    return {
        "id": case_id,
        "precision": 0.25,
        "recall": 0.5,
        "f1": 0.333333,
        "expected": [{"statement": "expected statement", "citations": [
            {"label": "gdpr Article 33", "relevance": None, "strength": "strong"},
        ]}],
        "produced": [],
        "summary_fidelity": None,
        "strength_agreement": None,
    }


def test_resume_skips_checkpointed_cases_and_stitches_the_report(ingested_store, query_log_path, tmp_path):
    """Resuming a run id re-runs only the pending cases: checkpointed results
    are substituted verbatim (never re-measured), fresh ones fill the rest,
    and the report keeps today's case-list order throughout."""
    from src.eval_harness import scenario_definition_hash

    cases = [_bare_case("first-case"), _bare_case("second-case")]
    checkpoint_dir = _checkpoint_document(
        tmp_path,
        cases=[_completed_result("first-case")],
        llm_model=live_settings(query_log_path).llm_model,
        hashes={"first-case": scenario_definition_hash(cases[0])},
    )
    install_fake_pipeline(_on_target_llm(), _on_target_retriever())

    report = run_live_eval(
        live_settings(query_log_path),
        run_id="resumed-run",
        checkpoint_dir=checkpoint_dir,
        scenarios=cases,
    )

    # The checkpointed case was substituted, not re-run: its impossible-from-
    # -the-pipeline numbers survive into the report, in list order.
    assert [result.id for result in report.scenarios] == ["first-case", "second-case"]
    assert report.scenarios[0].precision == 0.25
    assert report.scenarios[0].f1 == pytest.approx(0.333333)
    # Fresh: the on-target pipeline ran, hitting the expected target plus one
    # spurious one — precision 0.5, recall 1.0.
    assert report.scenarios[1].precision == 0.5
    # Only the pending case crossed the endpoint — the checkpointed one
    # cost no LLM work at all.
    records = [
        json.loads(line) for line in query_log_path.read_text().splitlines() if line.strip()
    ]
    assert len(records) == 1
    # The resume re-persisted the full set: both cases now checkpointed.
    data = json.loads((checkpoint_dir / "resumed-run.json").read_text())
    assert [case["id"] for case in data["scenarios"]] == ["first-case", "second-case"]


def test_resume_refuses_when_a_completed_case_changed_under_the_run_id(ingested_store, query_log_path, tmp_path):
    """A case edited since its result was checkpointed makes the record stale:
    the resume refuses instead of quietly substituting old numbers (ADR-0013)."""
    cases = [_bare_case("first-case")]
    checkpoint_dir = _checkpoint_document(
        tmp_path,
        cases=[_completed_result("first-case")],
        llm_model=live_settings(query_log_path).llm_model,
        hashes={"first-case": "stale-hash"},
    )
    install_fake_pipeline(_on_target_llm(), _on_target_retriever())

    with pytest.raises(LiveEvalRefused, match="changed since run"):
        run_live_eval(
            live_settings(query_log_path),
            run_id="resumed-run",
            checkpoint_dir=checkpoint_dir,
            scenarios=cases,
        )


def test_resume_refuses_when_the_configured_model_differs(ingested_store, query_log_path, tmp_path):
    """Results are model-sensitive (ADR-0009): a checkpoint produced by one
    model must never be stitched into a run configured for another."""
    cases = [_bare_case("first-case")]
    checkpoint_dir = _checkpoint_document(
        tmp_path,
        cases=[_completed_result("first-case")],
        llm_model="some-other-model",
        hashes={"first-case": "whatever"},
    )
    install_fake_pipeline(_on_target_llm(), _on_target_retriever())

    with pytest.raises(LiveEvalRefused, match="model-sensitive"):
        run_live_eval(
            live_settings(query_log_path),
            run_id="resumed-run",
            checkpoint_dir=checkpoint_dir,
            scenarios=cases,
        )


def test_resume_refuses_when_the_checkpoint_file_is_unreadable(ingested_store, query_log_path, tmp_path):
    """A corrupt checkpoint aborts with a clear message instead of silently
    re-running ten paid cases (ADR-0013): when it happens, something else is
    wrong and the operator decides."""
    checkpoint_dir = tmp_path / "eval-runs"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "resumed-run.json").write_text('{"llm_model": "test-model-9", "scen')

    with pytest.raises(LiveEvalRefused, match="unreadable"):
        run_live_eval(
            live_settings(query_log_path),
            run_id="resumed-run",
            checkpoint_dir=checkpoint_dir,
            scenarios=[_bare_case("first-case")],
        )


def test_resume_refuses_when_a_checkpointed_record_is_malformed(ingested_store, query_log_path, tmp_path):
    """Parseable JSON with a garbage case record is as unreadable as truncated
    JSON: the resume refuses with the same message, never a traceback."""
    from src.eval_harness import scenario_definition_hash

    checkpoint_dir = _checkpoint_document(
        tmp_path,
        cases=[{"id": "first-case", "precision": "not-a-number"}],
        llm_model=live_settings(query_log_path).llm_model,
        hashes={"first-case": scenario_definition_hash(_bare_case("first-case"))},
    )
    install_fake_pipeline(_on_target_llm(), _on_target_retriever())

    with pytest.raises(LiveEvalRefused, match="unreadable"):
        run_live_eval(
            live_settings(query_log_path),
            run_id="resumed-run",
            checkpoint_dir=checkpoint_dir,
            scenarios=[_bare_case("first-case")],
        )


def test_resume_keeps_the_original_run_start_in_the_checkpoint(ingested_store, query_log_path, tmp_path):
    """The checkpoint's metadata describes the run that measured the cases:
    resuming adopts the stored started_at instead of re-stamping the resume
    time over it (ADR-0013)."""
    from src.eval_harness import scenario_definition_hash

    case = _bare_case("first-case")
    checkpoint_dir = _checkpoint_document(
        tmp_path,
        cases=[_completed_result("first-case")],
        llm_model=live_settings(query_log_path).llm_model,
        hashes={"first-case": scenario_definition_hash(case)},
    )
    install_fake_pipeline(_on_target_llm(), _on_target_retriever())

    run_live_eval(
        live_settings(query_log_path),
        run_id="resumed-run",
        checkpoint_dir=checkpoint_dir,
        scenarios=[case, _bare_case("second-case")],
    )

    data = json.loads((checkpoint_dir / "resumed-run.json").read_text())
    assert data["started_at"] == "2026-09-03T10:00:00+00:00"


def test_resume_refuses_when_a_completed_record_has_no_hash(ingested_store, query_log_path, tmp_path):
    """A completed record without its definition hash is corruption, not an
    edited case: the refusal must say the file is unreadable, never blame the
    ground truth (ADR-0013)."""
    checkpoint_dir = _checkpoint_document(
        tmp_path,
        cases=[_completed_result("first-case")],
        llm_model=live_settings(query_log_path).llm_model,
        hashes={},
    )
    install_fake_pipeline(_on_target_llm(), _on_target_retriever())

    with pytest.raises(LiveEvalRefused, match="unreadable"):
        run_live_eval(
            live_settings(query_log_path),
            run_id="resumed-run",
            checkpoint_dir=checkpoint_dir,
            scenarios=[_bare_case("first-case")],
        )


def test_resume_refuses_when_a_checkpointed_score_is_not_a_number(ingested_store, query_log_path, tmp_path):
    """Dataclasses validate nothing, so a record with the right keys but a
    wrong-typed value must be caught at load: refusing as unreadable, never
    crashing later mid-print (ADR-0013)."""
    from src.eval_harness import scenario_definition_hash

    case = _completed_result("first-case")
    case["precision"] = "not-a-number"
    checkpoint_dir = _checkpoint_document(
        tmp_path,
        cases=[case],
        llm_model=live_settings(query_log_path).llm_model,
        hashes={"first-case": scenario_definition_hash(_bare_case("first-case"))},
    )
    install_fake_pipeline(_on_target_llm(), _on_target_retriever())

    with pytest.raises(LiveEvalRefused, match="unreadable"):
        run_live_eval(
            live_settings(query_log_path),
            run_id="resumed-run",
            checkpoint_dir=checkpoint_dir,
            scenarios=[_bare_case("first-case")],
        )


def test_resume_serves_a_fully_checkpointed_run_without_preconditions(query_log_path, tmp_path):
    """A checkpoint covering every case is a finished measurement: the report
    is served from it alone — no key check, no store probe, no provider call.
    The operator can re-print numbers from a machine that could no longer run
    a single case."""
    from src.eval_harness import scenario_definition_hash

    case = _bare_case("first-case")
    checkpoint_dir = _checkpoint_document(
        tmp_path,
        cases=[_completed_result("first-case")],
        llm_model=live_settings(query_log_path).llm_model,
        hashes={"first-case": scenario_definition_hash(case)},
    )
    # No key, no ingested store, no pipeline installed: a probe or request
    # would refuse or crash — the serve must never reach for either.
    settings = Settings(regula_mode=Mode.demo, openrouter_api_key=None, query_log_path=str(query_log_path))

    report = run_live_eval(settings, run_id="resumed-run", checkpoint_dir=checkpoint_dir, scenarios=[case])

    assert [result.id for result in report.scenarios] == ["first-case"]
    assert report.scenarios[0].precision == 0.25
    assert not query_log_path.exists()


def test_resume_ignores_records_for_cases_no_longer_in_the_list(ingested_store, query_log_path, tmp_path):
    """The current case list wins (ADR-0013): a record for a case that left
    the list goes unused — it neither fails the resume nor reaches the report."""
    from src.eval_harness import scenario_definition_hash

    cases = [_bare_case("first-case")]
    checkpoint_dir = _checkpoint_document(
        tmp_path,
        cases=[_completed_result("ghost-case")],
        llm_model=live_settings(query_log_path).llm_model,
        hashes={"ghost-case": scenario_definition_hash(_bare_case("ghost-case"))},
    )
    install_fake_pipeline(_on_target_llm(), _on_target_retriever())

    report = run_live_eval(
        live_settings(query_log_path),
        run_id="resumed-run",
        checkpoint_dir=checkpoint_dir,
        scenarios=cases,
    )

    assert [result.id for result in report.scenarios] == ["first-case"]
    data = json.loads((checkpoint_dir / "resumed-run.json").read_text())
    assert [case["id"] for case in data["scenarios"]] == ["first-case"]


# --- The checkpointed CLI: every command is resumable (ADR-0013) --------------------


class _dies_on_second_plan:
    """The provider seam that serves the first request whole, then dies on the
    second — with an LlmError (infrastructure failure) or a KeyboardInterrupt
    (the operator pausing the run). One Plan call per request, so the second
    Plan is the second case."""
    def __init__(self, inner, error):
        self._inner = inner
        self._error = error
        self._plans = 0

    def complete(self, system, user, schema):
        if schema is Plan:
            self._plans += 1
            if self._plans == 2:
                raise self._error
        return self._inner.complete(system, user, schema)


def _wire_live_cli(monkeypatch, llm=None, retriever=None):
    """The operator-CLI preamble every checkpointed-command test shares:
    ingested-store probe, operator env, fakes at the composition root."""
    import src.availability as availability

    monkeypatch.setattr(availability, "stored_chunk_count", lambda _database_url: 42)
    monkeypatch.setenv("REGULA_MODE", "demo")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    install_fake_pipeline(llm if llm is not None else _on_target_llm(), retriever if retriever is not None else _on_target_retriever())


def _two_cli_cases(monkeypatch):
    """Point the CLI at two bare cases so a provider that dies on the second
    Plan call aborts mid-run."""
    import src.live_eval as live_eval

    monkeypatch.setattr(live_eval, "LIVE_EVAL_SCENARIOS", [_bare_case("first-case"), _bare_case("second-case")])


def test_main_run_id_prints_the_checkpoint_and_keeps_it_without_output(monkeypatch, capsys, tmp_path):
    _wire_live_cli(monkeypatch)
    checkpoint_dir = tmp_path / "eval-runs"

    exit_code = main(["--run-id", "cli-run"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Run id: cli-run" in out
    assert str(checkpoint_dir / "cli-run.json") in out
    # No --output: the checkpoint file is the run's only record, so it stays.
    assert (checkpoint_dir / "cli-run.json").exists()


def test_main_deletes_the_checkpoint_when_the_artifact_supersedes_it(monkeypatch, capsys, tmp_path):
    _wire_live_cli(monkeypatch)
    checkpoint_dir = tmp_path / "eval-runs"
    out_path = tmp_path / "live-eval.json"

    exit_code = main(["--run-id", "cli-run", "--output", str(out_path)])

    assert exit_code == 0
    assert out_path.exists()
    assert not (checkpoint_dir / "cli-run.json").exists()


def test_main_mints_a_run_id_when_none_is_given(monkeypatch, capsys, tmp_path):
    import re

    _wire_live_cli(monkeypatch)
    checkpoint_dir = tmp_path / "eval-runs"

    exit_code = main([])

    assert exit_code == 0
    out = capsys.readouterr().out
    # UTC timestamp plus four random hex characters: sortable, collision-free
    # even when two runs start in the same second.
    minted = re.search(r"Run id: (\d{8}-\d{6}-[0-9a-f]{4})", out)
    assert minted, f"no minted run id in: {out}"
    assert (checkpoint_dir / f"{minted.group(1)}.json").exists()


def test_main_refuses_a_run_id_that_is_not_a_plain_filename(monkeypatch, capsys, tmp_path):
    """The run id names a file under data/eval-runs/: anything that could
    traverse out of it — slashes, leading dots — is refused before any work,
    never written (ADR-0013)."""
    _wire_live_cli(monkeypatch)

    exit_code = main(["--run-id", "../escape"])

    assert exit_code == 1
    assert "invalid --run-id" in capsys.readouterr().err
    assert not (tmp_path / "escape.json").exists()


def test_main_aborts_mid_run_with_completed_numbers_and_a_resume_hint(monkeypatch, capsys, tmp_path):
    """The provider dies on the second case: exit 1, the completed first
    case's numbers are printed and checkpointed, the artifact is not written,
    and stderr tells the operator exactly how to resume (ADR-0013)."""
    _two_cli_cases(monkeypatch)
    _wire_live_cli(monkeypatch, llm=_dies_on_second_plan(_on_target_llm(), LlmError("provider died mid-run")))
    checkpoint_dir = tmp_path / "eval-runs"
    out_path = tmp_path / "live-eval.json"

    exit_code = main(["--run-id", "cli-run", "--output", str(out_path)])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Live eval aborted" in captured.err
    assert "provider died mid-run" in captured.err
    assert "Resume with: python -m backend.src.live_eval --run-id cli-run" in captured.err
    # The measured work survives — visible and on disk.
    assert "first-case" in captured.out
    assert "second-case" not in captured.out
    assert not out_path.exists()
    data = json.loads((checkpoint_dir / "cli-run.json").read_text())
    assert [case["id"] for case in data["scenarios"]] == ["first-case"]


def test_main_interrupted_mid_run_pauses_with_a_resume_hint_and_exits_130(monkeypatch, capsys, tmp_path):
    """Ctrl-C mid-run is a first-class pause, not a crash: the checkpoint
    holds the completed cases and stderr says how to pick the run back up."""
    _two_cli_cases(monkeypatch)
    _wire_live_cli(monkeypatch, llm=_dies_on_second_plan(_on_target_llm(), KeyboardInterrupt()))
    checkpoint_dir = tmp_path / "eval-runs"
    out_path = tmp_path / "live-eval.json"

    exit_code = main(["--run-id", "cli-run", "--output", str(out_path)])

    assert exit_code == 130
    captured = capsys.readouterr()
    assert "paused" in captured.err
    assert "Resume with: python -m backend.src.live_eval --run-id cli-run" in captured.err
    assert "first-case" in captured.out
    assert not out_path.exists()
    data = json.loads((checkpoint_dir / "cli-run.json").read_text())
    assert [case["id"] for case in data["scenarios"]] == ["first-case"]


def test_main_checkpoints_beside_the_logs_when_data_is_read_only(monkeypatch, capsys, tmp_path):
    """The stack mounts data/ read-only (the Corpus lives there): the CLI
    then checkpoints under logs/eval-runs/ — beside the artifacts — instead
    of dying on the first completed case, and the printed line names the
    resolved location (ADR-0013)."""
    import src.live_eval as live_eval

    _wire_live_cli(monkeypatch)
    blocker = tmp_path / "data"
    blocker.write_text("not a directory, so eval-runs cannot be created here")
    monkeypatch.setattr(live_eval, "CHECKPOINT_DIR", blocker / "eval-runs")
    fallback = tmp_path / "logs" / "eval-runs"

    exit_code = main(["--run-id", "ro-run"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert str(fallback / "ro-run.json") in out
    assert (fallback / "ro-run.json").exists()


def test_failed_run_resumes_end_to_end_from_the_checkpoint(monkeypatch, capsys, tmp_path, query_log_path):
    """The operator flow ADR-0013 exists for, end to end: a run dies on the
    last case, the printed command picks it back up under the same id, only
    the pending case crosses the provider, and the artifact supersedes the
    checkpoint."""
    _two_cli_cases(monkeypatch)
    _wire_live_cli(monkeypatch, llm=_dies_on_second_plan(_on_target_llm(), LlmError("provider died mid-run")))
    checkpoint_dir = tmp_path / "eval-runs"
    out_path = tmp_path / "final-report.json"

    assert main(["--run-id", "chain-run", "--output", str(out_path)]) == 1
    assert not out_path.exists()

    # Same command, same id, healthy provider: the resume.
    query_log_path.unlink(missing_ok=True)
    _wire_live_cli(monkeypatch)
    exit_code = main(["--run-id", "chain-run", "--output", str(out_path)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "first-case" in out and "second-case" in out
    records = [
        json.loads(line) for line in query_log_path.read_text().splitlines() if line.strip()
    ]
    # The resume crosses the provider for the pending case only: the
    # checkpointed first case never re-ran.
    assert len(records) == 1
    artifact = json.loads(out_path.read_text())
    assert [case["id"] for case in artifact["scenarios"]] == ["first-case", "second-case"]
    assert not (checkpoint_dir / "chain-run.json").exists()
