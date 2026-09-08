"""Baseline-eval (spec #90, ticket #91): the parametric variant scored by
the shared eval harness and compared to the pipeline report.

Every test drives the run seam — fixture directories, a synthetic pipeline
report, and a scripted judge — so the suite stays fully offline: no
provider calls, no real logs, no writes into the repository. The CLI tests
pin the refusal and abort exit codes and the printed table.
"""

import json
import re
from pathlib import Path
from typing import Callable, Optional

import pytest
from pydantic import BaseModel, SecretStr

import src.baseline_eval as baseline_eval
from src.baseline_eval import (
    PARAMETRIC_MARKER,
    BaselineEvalRefused,
    DropRecords,
    main,
    run_baseline_eval,
)
from src.config import Settings
from src.eval_harness import (
    LIVE_EVAL_SCENARIOS,
    EvalScenario,
    ExpectedCitation,
    format_provision_target,
    parse_provision,
)
from src.eval_judge import (
    JudgeVerdict,
    PairFidelityVerdict,
    PairVerdict,
    RelevancePair,
    pair_fidelity_verdict,
)
from src.llm import LlmUnreachableError
from src.live_workflow import SEEK_COUNSEL_ACTION
from src.models import Mode, ProvisionKind, ProvisionTarget, Strength

from fakes import ROLE_MISMATCH_VERDICT, ScriptedJudge


def _settings(query_log_path) -> Settings:
    return Settings(
        regula_mode=Mode.demo,
        openrouter_api_key=SecretStr("sk-or-test"),
        query_log_path=str(query_log_path),
    )


class AgreeingJudge:
    """Ref-generic scripted judge: full agreement for whatever pairs arrive,
    every compared pair list recorded for the batched-call assertions."""

    def __init__(self) -> None:
        self.calls: list[list[RelevancePair]] = []

    def compare(self, pairs: list[RelevancePair]) -> dict[str, PairFidelityVerdict]:
        self.calls.append(list(pairs))
        return {
            pair.ref: pair_fidelity_verdict(
                pair,
                PairVerdict(ref=pair.ref, same_role=True, same_direction=True, contradiction=False),
            )
            for pair in pairs
        }


class UnreachableJudge:
    """The scripted judge whose provider call fails mid-run."""

    def compare(self, pairs: list[RelevancePair]) -> dict[str, PairFidelityVerdict]:
        raise LlmUnreachableError("the provider is unreachable")


class _AllAgreeJudgeClient:
    """The provider seam the CLI's default judge runs on: one full-agreement
    verdict per pair block in the prompt, so the CLI succeeds offline."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def complete(self, system: str, user: str, schema: type[BaseModel]) -> JudgeVerdict:
        refs = re.findall(r"\[(P\d+)\]", user)
        return JudgeVerdict(
            verdicts=[
                PairVerdict(ref=ref, same_role=True, same_direction=True, contradiction=False)
                for ref in refs
            ]
        )


class _DyingJudgeClient:
    """The provider seam whose judge call dies — infrastructure failure."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def complete(self, system: str, user: str, schema: type[BaseModel]) -> JudgeVerdict:
        raise LlmUnreachableError("provider down mid-run")


# --- Fixture builders ---------------------------------------------------------


def _unique_expected_citations(case: EvalScenario) -> list[ExpectedCitation]:
    """The case's expected citations, deduplicated by structural target in
    first-mention order — the produced side that covers the case exactly."""
    seen: dict[ProvisionTarget, ExpectedCitation] = {}
    for finding in case.expected:
        for citation in finding.citations:
            target = parse_provision(citation["source_id"], citation["provision"])
            seen.setdefault(target, citation)
    return list(seen.values())


def _payload(case: EvalScenario, *, citations: Optional[int] = None, summaries: bool = True) -> dict:
    """A valid stored-shape file for the case: one finding citing ``citations``
    (all unique expected targets when None) with a relevance summary per
    cited provision."""
    chosen = _unique_expected_citations(case)
    if citations is not None:
        chosen = chosen[:citations]
    return {
        "findings": [
            {
                "statement": f"Baseline finding for {case.scenario_id}.",
                "strength": "strong",
                "citations": [
                    {"source_id": c["source_id"], "provision": c["provision"]} for c in chosen
                ],
            }
        ],
        "summaries": [
            {
                "source_id": c["source_id"],
                "provision": c["provision"],
                "relevance": f"Baseline relevance for {c['provision']}.",
                "strength": "moderate",
            }
            for c in chosen
        ]
        if summaries
        else [],
        "actions": ["Confirm the company's facts with the compliance officer."],
    }


def _write_dir(
    tmp_path: Path,
    overrides: Optional[dict[str, object]] = None,
    *,
    minimal: bool = False,
) -> Path:
    """A fixture baseline directory: one valid file per live Scenario id,
    overrides applied by filename (None deletes, str writes raw bytes)."""
    directory = tmp_path / "parametric"
    directory.mkdir(parents=True)
    for case in LIVE_EVAL_SCENARIOS:
        payload = _payload(case, citations=1 if minimal else None)
        (directory / f"{case.scenario_id}.json").write_text(json.dumps(payload))
    for name, content in (overrides or {}).items():
        target = directory / name
        if content is None:
            target.unlink()
        elif isinstance(content, str):
            target.write_text(content)
        else:
            target.write_text(json.dumps(content))
    return directory


def _write_report(
    path: Path,
    *,
    llm_model: str = "z-ai/test-model",
    entries: Optional[dict[str, tuple[float, float, float, Optional[float]]]] = None,
) -> Path:
    """A synthetic pipeline report in the live-eval artifact shape; cases
    absent from ``entries`` score zeros with unmeasured fidelity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    by_id = entries or {}
    scenarios = []
    for case in LIVE_EVAL_SCENARIOS:
        precision, recall, f1, fidelity = by_id.get(case.id, (0.0, 0.0, 0.0, None))
        scenarios.append({
            "id": case.id,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "summary_fidelity": fidelity,
            "expected": [],
            "produced": [],
            "fidelity_verdicts": [],
            "summarizer_limitation": None,
        })
    path.write_text(json.dumps({
        "mode": "live",
        "generated_at": "2026-09-07T20:15:37+00:00",
        "llm_model": llm_model,
        "scenarios": scenarios,
        "mean_f1": 0.0,
        "mean_summary_fidelity": None,
    }))
    return path


def _case_file(case: EvalScenario) -> str:
    return f"{case.scenario_id}.json"


_FIRST_CASE = LIVE_EVAL_SCENARIOS[0]
_FIRST_FILE = _case_file(_FIRST_CASE)
_REPORT = "live-eval-report-fixture.json"


# --- The measured run ----------------------------------------------------------


def test_scores_all_ten_cases_with_the_scripted_judge(tmp_path, query_log_path):
    directory = _write_dir(tmp_path)
    report = _write_report(
        tmp_path / _REPORT,
        entries={_FIRST_CASE.id: (0.6, 0.5, 0.55, 0.8)},
    )
    judge = AgreeingJudge()

    comparison = run_baseline_eval(
        _settings(query_log_path), judge=judge, parametric_dir=directory, report_path=report
    )

    assert [case.case_id for case in comparison.cases] == [case.id for case in LIVE_EVAL_SCENARIOS]
    # One batched judge call per case, every call carrying pairs.
    assert len(judge.calls) == len(LIVE_EVAL_SCENARIOS)
    assert all(call for call in judge.calls)
    # Every file cites exactly its case's expected targets: full coverage.
    for case in comparison.cases:
        assert case.parametric.precision == 1.0
        assert case.parametric.recall == 1.0
        assert case.parametric.f1 == 1.0
        assert case.parametric.summary_fidelity == 1.0
    assert comparison.parametric_report.mean_f1 == pytest.approx(1.0)
    assert comparison.parametric_report.mean_summary_fidelity == pytest.approx(1.0)
    # The pipeline side joins per case, never re-run.
    assert comparison.report_path == report
    assert comparison.pipeline_model == "z-ai/test-model"
    assert comparison.cases[0].pipeline.precision == 0.6
    assert comparison.cases[0].pipeline.recall == 0.5
    assert comparison.cases[0].pipeline.f1 == 0.55
    assert comparison.cases[0].pipeline.summary_fidelity == 0.8
    assert comparison.cases[1].pipeline.f1 == 0.0
    # No drops in the clean run.
    assert comparison.drops.duplicate_summaries == []
    assert comparison.drops.unparseable_labels == []
    # No provider answered and no real log was touched.
    assert not Path(str(query_log_path)).exists()


def test_produced_dump_carries_the_baseline_shape(tmp_path, query_log_path):
    directory = _write_dir(tmp_path, minimal=True)
    report = _write_report(tmp_path / _REPORT)

    comparison = run_baseline_eval(
        _settings(query_log_path),
        judge=AgreeingJudge(),
        parametric_dir=directory,
        report_path=report,
    )

    first = comparison.cases[0].parametric
    assert first.id == _FIRST_CASE.id
    produced = first.produced[0]
    assert produced["statement"] == f"Baseline finding for {_FIRST_CASE.scenario_id}."
    assert produced["strength"] == "strong"
    citation = produced["citations"][0]
    expected = _unique_expected_citations(_FIRST_CASE)[0]
    assert citation["target"] == format_provision_target(
        parse_provision(expected["source_id"], expected["provision"])
    )
    assert citation["relevance"] == f"Baseline relevance for {expected['provision']}."
    assert citation["strength"] == "moderate"
    # No quote is ever fabricated for a baseline answer.
    assert citation["quote"] is None


def test_scripted_verdicts_ride_the_result_records(tmp_path, query_log_path):
    directory = _write_dir(tmp_path, minimal=True)
    judge = ScriptedJudge({"P1": ROLE_MISMATCH_VERDICT})
    report = _write_report(tmp_path / _REPORT)

    comparison = run_baseline_eval(
        _settings(query_log_path), judge=judge, parametric_dir=directory, report_path=report
    )

    assert len(judge.calls) == len(LIVE_EVAL_SCENARIOS)
    assert all(len(call) == 1 for call in judge.calls)
    first = comparison.cases[0].parametric
    assert first.summary_fidelity == 0.5
    verdict = first.fidelity_verdicts[0]
    expected = _unique_expected_citations(_FIRST_CASE)[0]
    assert verdict["provision"] == format_provision_target(
        parse_provision(expected["source_id"], expected["provision"])
    )
    assert verdict["score"] == 0.5
    assert verdict["role_match"] is False
    assert verdict["direction_match"] is True
    assert verdict["contradiction"] is False


def test_unsummarized_cited_provision_floors_without_a_judge_call(tmp_path, query_log_path):
    directory = _write_dir(
        tmp_path,
        overrides={_case_file(LIVE_EVAL_SCENARIOS[3]): _payload(LIVE_EVAL_SCENARIOS[3], citations=1, summaries=False)},
    )
    judge = AgreeingJudge()
    report = _write_report(tmp_path / _REPORT)

    comparison = run_baseline_eval(
        _settings(query_log_path), judge=judge, parametric_dir=directory, report_path=report
    )

    # Nine cases judge one pair each; the bare one floors deterministically.
    assert len(judge.calls) == len(LIVE_EVAL_SCENARIOS) - 1
    floored = comparison.cases[3].parametric
    assert floored.summary_fidelity == 0.0
    assert floored.fidelity_verdicts == []
    assert floored.summarizer_limitation is None


# --- The adapter: stored shape -> response shape -------------------------------


def test_adapter_maps_the_stored_shape_through_the_answer_citations_builder(tmp_path):
    path = tmp_path / "online-retailer-breach.json"
    path.write_text(json.dumps({
        "findings": [
            {
                "statement": "Notify the authority within 72 hours.",
                "strength": "strong",
                "citations": [{"source_id": "gdpr", "provision": "Article 33(1)"}],
            },
        ],
        "summaries": [
            {
                "source_id": "gdpr",
                "provision": "Article 33(1)",
                "relevance": "Sets the 72-hour clock.",
                "strength": "moderate",
            },
        ],
        "actions": ["Log the breach timeline."],
    }))

    response = baseline_eval.load_parametric_case(path, DropRecords())

    finding = response.answer.findings[0]
    assert finding.statement == "Notify the authority within 72 hours."
    assert finding.strength is Strength.strong
    # The per-paragraph label folds onto the Article it refines.
    citation = response.answer.citations[0]
    assert citation.provision_target == ProvisionTarget("gdpr", ProvisionKind.article, 33)
    assert citation.relevance == "Sets the 72-hour clock."
    assert citation.strength is Strength.moderate
    # The standing seek-counsel hand-off rides last.
    assert response.answer.actions == ["Log the breach timeline.", SEEK_COUNSEL_ACTION]
    assert response.trace.workflow == PARAMETRIC_MARKER
    # Nothing is fabricated: no limitation, no trace detail, no quote.
    assert response.known_limitations == []
    assert response.detailed_trace == []
    assert citation.quote is None


# --- The two recorded exceptions (ADR-0017) -------------------------------------


def test_duplicate_summaries_resolve_first_wins_and_are_counted(tmp_path, query_log_path):
    override = _payload(_FIRST_CASE, citations=1)
    label = _unique_expected_citations(_FIRST_CASE)[0]["provision"]
    override["summaries"] = [
        {"source_id": "gdpr", "provision": label, "relevance": "First summary stands.", "strength": "moderate"},
        {"source_id": "gdpr", "provision": f"{label}(12)", "relevance": "Later duplicate falls.", "strength": "weak"},
    ]
    directory = _write_dir(tmp_path, overrides={_FIRST_FILE: override})
    judge = AgreeingJudge()
    report = _write_report(tmp_path / _REPORT)

    comparison = run_baseline_eval(
        _settings(query_log_path), judge=judge, parametric_dir=directory, report_path=report
    )

    produced = comparison.cases[0].parametric.produced[0]["citations"][0]
    assert produced["relevance"] == "First summary stands."
    assert comparison.drops.duplicate_summaries == [
        baseline_eval.DroppedDuplicateSummary(file=directory / _FIRST_FILE, label=f"{label}(12)")
    ]
    assert comparison.drops.unparseable_labels == []


def test_unparseable_citation_label_is_dropped_recorded_and_counted(tmp_path, query_log_path):
    override = _payload(_FIRST_CASE, citations=1)
    override["findings"][0]["citations"].append({"source_id": "gdpr", "provision": "Section 12"})
    directory = _write_dir(tmp_path, overrides={_FIRST_FILE: override})
    judge = AgreeingJudge()
    report = _write_report(tmp_path / _REPORT)

    comparison = run_baseline_eval(
        _settings(query_log_path), judge=judge, parametric_dir=directory, report_path=report
    )

    # Only the parseable citation produces; the junk label never flatters
    # precision and never voids the run.
    produced_citations = comparison.cases[0].parametric.produced[0]["citations"]
    assert len(produced_citations) == 1
    assert comparison.cases[0].parametric.precision == 1.0
    drops = comparison.drops.unparseable_labels
    assert len(drops) == 1
    assert drops[0].file == directory / _FIRST_FILE
    assert drops[0].label == "Section 12"
    assert "parse" in drops[0].reason
    assert comparison.drops.duplicate_summaries == []


def test_unparseable_summary_label_is_dropped_and_counted(tmp_path, query_log_path):
    override = _payload(_FIRST_CASE, citations=1)
    override["summaries"] = [
        {"source_id": "gdpr", "provision": "Section 12", "relevance": "Junk label.", "strength": "moderate"},
    ]
    directory = _write_dir(tmp_path, overrides={_FIRST_FILE: override})
    judge = AgreeingJudge()
    report = _write_report(tmp_path / _REPORT)

    comparison = run_baseline_eval(
        _settings(query_log_path), judge=judge, parametric_dir=directory, report_path=report
    )

    # The dropped summary leaves the cited provision bare: deterministic
    # floor, no judge call for that case.
    assert len(judge.calls) == len(LIVE_EVAL_SCENARIOS) - 1
    assert comparison.cases[0].parametric.summary_fidelity == 0.0
    drops = comparison.drops.unparseable_labels
    assert len(drops) == 1
    assert drops[0].label == "Section 12"


# --- Strict validation: the run refuses, naming file and problem ----------------


_MALFORMED: list[tuple[str, Callable[[dict], None]]] = [
    ("extra top-level key", lambda payload: payload.update({"extra": []})),
    ("missing summaries key", lambda payload: payload.pop("summaries")),
    ("findings not a list", lambda payload: payload.update({"findings": {"statement": "x"}})),
    ("finding missing citations key", lambda payload: payload["findings"][0].pop("citations")),
    ("finding empty statement", lambda payload: payload["findings"][0].update({"statement": ""})),
    ("finding non-string statement", lambda payload: payload["findings"][0].update({"statement": 7})),
    ("finding invalid strength", lambda payload: payload["findings"][0].update({"strength": "critical"})),
    (
        "finding non-list citations",
        lambda payload: payload["findings"][0].update({"citations": "gdpr Article 4"}),
    ),
    ("finding empty citations", lambda payload: payload["findings"][0].update({"citations": []})),
    ("citation missing provision", lambda payload: payload["findings"][0]["citations"][0].pop("provision")),
    ("citation extra key", lambda payload: payload["findings"][0]["citations"][0].update({"quote": "x"})),
    (
        "citation unknown source id",
        lambda payload: payload["findings"][0]["citations"][0].update({"source_id": "iso-27001"}),
    ),
    ("summary invalid strength", lambda payload: payload["summaries"][0].update({"strength": "extreme"})),
    ("summary empty relevance", lambda payload: payload["summaries"][0].update({"relevance": ""})),
    ("summary missing relevance key", lambda payload: payload["summaries"][0].pop("relevance")),
    (
        "summary unknown source id",
        lambda payload: payload["summaries"][0].update({"source_id": "eidas"}),
    ),
    ("actions not a list", lambda payload: payload.update({"actions": "confirm with counsel"})),
    ("action non-string", lambda payload: payload.update({"actions": [7]})),
]


@pytest.mark.parametrize(
    ["label", "mutate"],
    _MALFORMED,
    ids=[name for name, _ in _MALFORMED],
)
def test_malformed_case_file_refuses_naming_file_and_problem(
    tmp_path, query_log_path, label: str, mutate: Callable[[dict], None]
):
    payload = _payload(_FIRST_CASE)
    mutate(payload)
    directory = _write_dir(tmp_path, overrides={_FIRST_FILE: payload})
    judge = AgreeingJudge()
    report = _write_report(tmp_path / _REPORT)

    with pytest.raises(BaselineEvalRefused, match=_FIRST_FILE) as excinfo:
        run_baseline_eval(
            _settings(query_log_path), judge=judge, parametric_dir=directory, report_path=report
        )

    # The refusal names the problem, not just the file, and no judge call
    # was ever spent on a run that could not start.
    assert str(excinfo.value).strip() != str(directory / _FIRST_FILE)
    assert judge.calls == []


def test_stray_filename_outside_the_live_scenarios_refuses(tmp_path, query_log_path):
    directory = _write_dir(tmp_path, overrides={"not-a-scenario.json": _payload(_FIRST_CASE)})
    report = _write_report(tmp_path / _REPORT)

    with pytest.raises(BaselineEvalRefused, match="not-a-scenario"):
        run_baseline_eval(
            _settings(query_log_path),
            judge=AgreeingJudge(),
            parametric_dir=directory,
            report_path=report,
        )


def test_stray_non_json_file_is_never_silently_skipped(tmp_path, query_log_path):
    directory = _write_dir(tmp_path, overrides={"stray-notes.txt": "leftover export"})
    report = _write_report(tmp_path / _REPORT)

    with pytest.raises(BaselineEvalRefused, match="stray-notes"):
        run_baseline_eval(
            _settings(query_log_path),
            judge=AgreeingJudge(),
            parametric_dir=directory,
            report_path=report,
        )


def test_stray_subdirectory_is_never_silently_skipped(tmp_path, query_log_path):
    directory = _write_dir(tmp_path)
    (directory / "nested-export").mkdir()
    report = _write_report(tmp_path / _REPORT)

    with pytest.raises(BaselineEvalRefused, match="nested-export"):
        run_baseline_eval(
            _settings(query_log_path),
            judge=AgreeingJudge(),
            parametric_dir=directory,
            report_path=report,
        )


def test_missing_case_file_refuses_naming_it(tmp_path, query_log_path):
    directory = _write_dir(tmp_path, overrides={_FIRST_FILE: None})
    report = _write_report(tmp_path / _REPORT)

    with pytest.raises(BaselineEvalRefused, match=_FIRST_CASE.scenario_id):
        run_baseline_eval(
            _settings(query_log_path),
            judge=AgreeingJudge(),
            parametric_dir=directory,
            report_path=report,
        )


def test_unreadable_case_file_refuses(tmp_path, query_log_path):
    directory = _write_dir(tmp_path, overrides={_FIRST_FILE: "{not json"})
    report = _write_report(tmp_path / _REPORT)

    with pytest.raises(BaselineEvalRefused, match=_FIRST_FILE):
        run_baseline_eval(
            _settings(query_log_path),
            judge=AgreeingJudge(),
            parametric_dir=directory,
            report_path=report,
        )


def test_missing_baseline_directory_refuses(tmp_path, query_log_path):
    report = _write_report(tmp_path / _REPORT)

    with pytest.raises(BaselineEvalRefused, match="does not exist"):
        run_baseline_eval(
            _settings(query_log_path),
            judge=AgreeingJudge(),
            parametric_dir=tmp_path / "nope",
            report_path=report,
        )


def test_missing_api_key_refuses_when_the_default_judge_is_needed(tmp_path, query_log_path):
    directory = _write_dir(tmp_path)
    report = _write_report(tmp_path / _REPORT)
    settings = Settings(
        regula_mode=Mode.demo, openrouter_api_key=None, query_log_path=str(query_log_path)
    )

    with pytest.raises(BaselineEvalRefused, match="OPENROUTER_API_KEY"):
        run_baseline_eval(settings, parametric_dir=directory, report_path=report)


def test_judge_failure_aborts_the_run(tmp_path, query_log_path):
    directory = _write_dir(tmp_path)
    report = _write_report(tmp_path / _REPORT)

    with pytest.raises(LlmUnreachableError, match="the provider is unreachable"):
        run_baseline_eval(
            _settings(query_log_path),
            judge=UnreachableJudge(),
            parametric_dir=directory,
            report_path=report,
        )


# --- The pipeline report side of the join ----------------------------------------


def test_report_missing_a_live_case_refuses(tmp_path, query_log_path):
    directory = _write_dir(tmp_path)
    report = _write_report(tmp_path / _REPORT)
    data = json.loads(report.read_text())
    data["scenarios"] = data["scenarios"][:-1]
    report.write_text(json.dumps(data))
    last_case = LIVE_EVAL_SCENARIOS[-1]

    with pytest.raises(BaselineEvalRefused, match=last_case.id):
        run_baseline_eval(
            _settings(query_log_path),
            judge=AgreeingJudge(),
            parametric_dir=directory,
            report_path=report,
        )


@pytest.mark.parametrize(
    ["label", "content", "fragment"],
    [
        ("unreadable json", "{not json", "cannot be read"),
        ("not an object", "[]", "JSON object"),
        ("scenarios missing", json.dumps({"llm_model": "m"}), "scenarios"),
        ("scenarios not a list", json.dumps({"scenarios": {}}), "scenarios"),
        (
            "case without id",
            json.dumps({"scenarios": [{"precision": 1, "recall": 1, "f1": 1}]}),
            "id",
        ),
        (
            "non-numeric f1",
            json.dumps({"scenarios": [{"id": "live-retailer-breach", "precision": 1, "recall": 1, "f1": "high"}]}),
            "f1",
        ),
        (
            "boolean precision",
            json.dumps({"scenarios": [{"id": "live-retailer-breach", "precision": True, "recall": 1, "f1": 1}]}),
            "precision",
        ),
    ],
    ids=[
        "unreadable json",
        "not an object",
        "scenarios missing",
        "scenarios not a list",
        "case without id",
        "non-numeric f1",
        "boolean precision",
    ],
)
def test_malformed_pipeline_report_refuses_naming_file_and_problem(
    tmp_path, query_log_path, label: str, content: str, fragment: str
):
    directory = _write_dir(tmp_path)
    report = tmp_path / _REPORT
    report.write_text(content)

    with pytest.raises(BaselineEvalRefused, match=_REPORT) as excinfo:
        run_baseline_eval(
            _settings(query_log_path),
            judge=AgreeingJudge(),
            parametric_dir=directory,
            report_path=report,
        )

    assert fragment in str(excinfo.value)


# --- File-location defaults -------------------------------------------------------


def test_default_locations_resolve_to_the_canonical_dirs(tmp_path, monkeypatch, query_log_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    report = _write_report(logs / "live-eval-report-only.json")
    directory = _write_dir(tmp_path)
    monkeypatch.setattr(baseline_eval, "LOGS_DIR", logs)
    monkeypatch.setattr(baseline_eval, "BASELINE_DIR", directory)

    comparison = run_baseline_eval(_settings(query_log_path), judge=AgreeingJudge())

    assert comparison.report_path == report
    assert [case.parametric.id for case in comparison.cases] == [
        case.id for case in LIVE_EVAL_SCENARIOS
    ]


def test_newest_report_wins_by_default(tmp_path, monkeypatch, query_log_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_report(logs / "live-eval-report-2026-09-01-a.json")
    newest = _write_report(logs / "live-eval-report-2026-09-08-b.json")
    directory = _write_dir(tmp_path)
    monkeypatch.setattr(baseline_eval, "LOGS_DIR", logs)
    monkeypatch.setattr(baseline_eval, "BASELINE_DIR", directory)

    comparison = run_baseline_eval(_settings(query_log_path), judge=AgreeingJudge())

    assert comparison.report_path == newest


def test_no_report_found_refuses_with_the_flag_hint(tmp_path, monkeypatch, query_log_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr(baseline_eval, "LOGS_DIR", logs)

    with pytest.raises(BaselineEvalRefused, match="--report"):
        run_baseline_eval(_settings(query_log_path), judge=AgreeingJudge())


# --- The printer -------------------------------------------------------------------


def test_printer_prints_table_means_and_delta_footer(tmp_path, query_log_path, capsys):
    directory = _write_dir(tmp_path)
    entries: dict[str, tuple[float, float, float, Optional[float]]] = {
        LIVE_EVAL_SCENARIOS[2].id: (0.5, 0.5, 0.5, 0.5)
    }
    report = _write_report(tmp_path / _REPORT, entries=entries)

    comparison = run_baseline_eval(
        _settings(query_log_path),
        judge=AgreeingJudge(),
        parametric_dir=directory,
        report_path=report,
    )
    baseline_eval.print_comparison(comparison)
    out = capsys.readouterr().out

    assert f"Pipeline report: {report} (model: z-ai/test-model)" in out
    assert _FIRST_CASE.id in out
    assert LIVE_EVAL_SCENARIOS[2].id in out
    # Pipeline row for the one scored case; parametric rows are all full marks.
    assert "0.500 / 0.500 / 0.500 / 0.500" in out
    assert "1.000 / 1.000 / 1.000 / 1.000" in out
    # Means and the Δ(pipeline − parametric) footer: coverage means over all
    # ten cases (0.5/10 = 0.050 against 1.000); fidelity means over the
    # measured cases only (one measured pipeline case: 0.500 against 1.000).
    assert (
        "Mean coverage F1: pipeline 0.050 | parametric 1.000 | Δ(pipeline − parametric) -0.950"
        in out
    )
    assert (
        "Mean summary fidelity: pipeline 0.500 | parametric 1.000 | Δ(pipeline − parametric) -0.500"
        in out
    )


def test_printer_handles_unmeasured_fidelity(tmp_path, query_log_path, capsys):
    directory = _write_dir(tmp_path, minimal=True)
    for case in LIVE_EVAL_SCENARIOS:
        (directory / _case_file(case)).write_text(
            json.dumps(_payload(case, citations=1, summaries=False))
        )
    report = _write_report(tmp_path / _REPORT)

    comparison = run_baseline_eval(
        _settings(query_log_path),
        judge=AgreeingJudge(),
        parametric_dir=directory,
        report_path=report,
    )
    baseline_eval.print_comparison(comparison)
    out = capsys.readouterr().out

    assert "n/a" in out
    assert "Δ(pipeline − parametric) n/a" in out


# --- The CLI ------------------------------------------------------------------------


def _wire_cli(monkeypatch, tmp_path, logs_dir: Path) -> None:
    monkeypatch.setenv("REGULA_MODE", "demo")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(baseline_eval, "LOGS_DIR", logs_dir)


def test_cli_success_prints_the_table_and_exits_zero(tmp_path, monkeypatch, capsys):
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_report(logs / "live-eval-report-2026-09-01-a.json")
    newest = _write_report(logs / "live-eval-report-2026-09-08-b.json")
    directory = _write_dir(tmp_path)
    _wire_cli(monkeypatch, tmp_path, logs)
    monkeypatch.setattr("src.config.OpenRouterClient", _AllAgreeJudgeClient)

    exit_code = main(["--parametric-dir", str(directory)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Baseline comparison: pipeline vs parametric (10 case(s))" in out
    assert str(newest) in out
    assert "z-ai/test-model" in out
    assert _FIRST_CASE.id in out


def test_cli_refusal_exits_1_naming_the_file(tmp_path, monkeypatch, capsys):
    logs = tmp_path / "logs"
    logs.mkdir()
    report = _write_report(logs / _REPORT)
    directory = _write_dir(tmp_path, overrides={_FIRST_FILE: None})
    _wire_cli(monkeypatch, tmp_path, logs)

    exit_code = main(["--parametric-dir", str(directory), "--report", str(report)])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Baseline eval refused" in captured.err
    assert _FIRST_CASE.scenario_id in captured.err
    assert captured.out == ""


def test_cli_judge_failure_aborts_with_exit_1(tmp_path, monkeypatch, capsys):
    logs = tmp_path / "logs"
    logs.mkdir()
    report = _write_report(logs / _REPORT)
    directory = _write_dir(tmp_path)
    _wire_cli(monkeypatch, tmp_path, logs)
    monkeypatch.setattr("src.config.OpenRouterClient", _DyingJudgeClient)

    exit_code = main(["--parametric-dir", str(directory), "--report", str(report)])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Baseline eval aborted" in captured.err
    assert "summary-fidelity judge" in captured.err
    assert "provider down mid-run" in captured.err
