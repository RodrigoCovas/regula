"""Baseline-eval (spec #90, tickets #91–#92): both baseline variants scored
by the shared eval harness and compared to the pipeline report, with the
combined artifact.

Every test drives the run seam — fixture directories, a synthetic pipeline
report, and a scripted judge — so the suite stays fully offline: no
provider calls, no real logs, no writes into the repository. The CLI tests
pin the refusal and abort exit codes, the printed table, and the artifact
write.
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional

import pytest
from pydantic import BaseModel, SecretStr

import src.baseline_eval as baseline_eval
from src.baseline_eval import (
    FULL_CORPUS_MARKER,
    FULL_CORPUS_VARIANT,
    PARAMETRIC_MARKER,
    PARAMETRIC_VARIANT,
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
    SummaryJudge,
    pair_fidelity_verdict,
)
from src.llm import LlmUnreachableError
from src.live_workflow import SEEK_COUNSEL_ACTION
from src.models import Mode, ProvisionKind, ProvisionTarget, Strength

from fakes import ROLE_MISMATCH_VERDICT, ScriptedJudge

VARIANTS = (PARAMETRIC_VARIANT, FULL_CORPUS_VARIANT)


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


def _duplicate_summaries_override(case: EvalScenario) -> dict:
    """A valid case file whose summaries repeat the same structural target —
    the second under a sub-reference label: the first-wins gate drops the
    later duplicate and counts it (ADR-0017)."""
    override = _payload(case, citations=1)
    label = _unique_expected_citations(case)[0]["provision"]
    override["summaries"] = [
        {"source_id": "gdpr", "provision": label, "relevance": "First summary stands.", "strength": "moderate"},
        {"source_id": "gdpr", "provision": f"{label}(12)", "relevance": "Later duplicate falls.", "strength": "weak"},
    ]
    return override


def _write_variant_dir(
    tmp_path: Path,
    variant: str,
    overrides: Optional[Mapping[str, object]] = None,
    *,
    minimal: bool = False,
) -> Path:
    """A fixture baseline directory for one variant: one valid file per live
    Scenario id, overrides applied by filename (None deletes, str writes raw
    bytes)."""
    directory = tmp_path / variant
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


def _write_dirs(
    tmp_path: Path,
    overrides_by_variant: Optional[Mapping[str, Mapping[str, object]]] = None,
    *,
    minimal: bool = False,
) -> dict[str, Path]:
    """Both variant directories, keyed by variant name: one valid file per
    live Scenario id each. ``overrides_by_variant`` maps a variant name to
    its per-file overrides (None deletes, str writes raw bytes); a variant
    absent from the map stays all-valid."""
    overrides_by_variant = overrides_by_variant or {}
    return {
        variant: _write_variant_dir(
            tmp_path, variant, overrides_by_variant.get(variant), minimal=minimal
        )
        for variant in (PARAMETRIC_VARIANT, FULL_CORPUS_VARIANT)
    }


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


def _run(
    settings: Settings,
    judge: Optional[SummaryJudge],
    tmp_path: Path,
    *,
    overrides_by_variant: Optional[Mapping[str, Mapping[str, object]]] = None,
    minimal: bool = False,
    report: Optional[Path] = None,
):
    """The one run-seam invocation most tests share: both fixture variant
    directories (overrides keyed by variant) against the synthetic report."""
    dirs = _write_dirs(tmp_path, overrides_by_variant, minimal=minimal)
    return run_baseline_eval(
        settings,
        judge=judge,
        variant_dirs=dirs,
        report_path=report if report is not None else tmp_path / _REPORT,
    )


# --- The measured run ----------------------------------------------------------


def test_scores_all_ten_cases_for_both_variants_with_the_scripted_judge(tmp_path, query_log_path):
    report = _write_report(
        tmp_path / _REPORT,
        entries={_FIRST_CASE.id: (0.6, 0.5, 0.55, 0.8)},
    )
    judge = AgreeingJudge()

    comparison = _run(_settings(query_log_path), judge, tmp_path, report=report)

    assert [case.case_id for case in comparison.cases] == [case.id for case in LIVE_EVAL_SCENARIOS]
    # One batched judge call per case per variant: twenty calls, every call
    # carrying pairs.
    assert len(judge.calls) == 2 * len(LIVE_EVAL_SCENARIOS)
    assert all(call for call in judge.calls)
    # Every file cites exactly its case's expected targets: full coverage on
    # both answering paths.
    for case in comparison.cases:
        assert case.baselines[PARAMETRIC_VARIANT].precision == 1.0
        assert case.baselines[PARAMETRIC_VARIANT].recall == 1.0
        assert case.baselines[PARAMETRIC_VARIANT].f1 == 1.0
        assert case.baselines[PARAMETRIC_VARIANT].summary_fidelity == 1.0
        assert case.baselines[FULL_CORPUS_VARIANT].precision == 1.0
        assert case.baselines[FULL_CORPUS_VARIANT].recall == 1.0
        assert case.baselines[FULL_CORPUS_VARIANT].f1 == 1.0
        assert case.baselines[FULL_CORPUS_VARIANT].summary_fidelity == 1.0
    assert comparison.outcomes[PARAMETRIC_VARIANT].report.mean_f1 == pytest.approx(1.0)
    assert comparison.outcomes[PARAMETRIC_VARIANT].report.mean_summary_fidelity == pytest.approx(1.0)
    assert comparison.outcomes[FULL_CORPUS_VARIANT].report.mean_f1 == pytest.approx(1.0)
    assert comparison.outcomes[FULL_CORPUS_VARIANT].report.mean_summary_fidelity == pytest.approx(1.0)
    # The pipeline side joins per case, never re-run.
    assert comparison.report_path == report
    assert comparison.pipeline_model == "z-ai/test-model"
    assert comparison.cases[0].pipeline.precision == 0.6
    assert comparison.cases[0].pipeline.recall == 0.5
    assert comparison.cases[0].pipeline.f1 == 0.55
    assert comparison.cases[0].pipeline.summary_fidelity == 0.8
    assert comparison.cases[1].pipeline.f1 == 0.0
    # No drops in the clean run, on either variant.
    assert comparison.outcomes[PARAMETRIC_VARIANT].drops.duplicate_summaries == []
    assert comparison.outcomes[PARAMETRIC_VARIANT].drops.unparseable_labels == []
    assert comparison.outcomes[FULL_CORPUS_VARIANT].drops.duplicate_summaries == []
    assert comparison.outcomes[FULL_CORPUS_VARIANT].drops.unparseable_labels == []
    # No provider answered and no real log was touched.
    assert not Path(str(query_log_path)).exists()


def test_full_corpus_variant_is_scored_independently_of_the_parametric(tmp_path, query_log_path):
    dirs = _write_dirs(
        tmp_path,
        minimal=False,
        overrides_by_variant={FULL_CORPUS_VARIANT: {_case_file(_FIRST_CASE): _payload(_FIRST_CASE, citations=1)}},
    )
    report = _write_report(tmp_path / _REPORT)

    comparison = run_baseline_eval(
        _settings(query_log_path),
        judge=AgreeingJudge(),
        variant_dirs=dirs,
        report_path=report,
    )

    # The full-corpus file cites only one of the case's expected targets: its
    # recall falls while the parametric side stays at full marks.
    first = comparison.cases[0]
    assert first.baselines[PARAMETRIC_VARIANT].f1 == 1.0
    assert first.baselines[FULL_CORPUS_VARIANT].precision == 1.0
    assert first.baselines[FULL_CORPUS_VARIANT].recall < 1.0
    assert comparison.outcomes[PARAMETRIC_VARIANT].report.mean_f1 == pytest.approx(1.0)
    assert comparison.outcomes[FULL_CORPUS_VARIANT].report.mean_f1 < 1.0


def test_produced_dump_carries_the_baseline_shape(tmp_path, query_log_path):
    judge = AgreeingJudge()
    dirs = _write_dirs(tmp_path, minimal=True)
    report = _write_report(tmp_path / _REPORT)

    comparison = run_baseline_eval(
        _settings(query_log_path),
        judge=judge,
        variant_dirs=dirs,
        report_path=report,
    )

    first = comparison.cases[0].baselines[PARAMETRIC_VARIANT]
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
    judge = ScriptedJudge({"P1": ROLE_MISMATCH_VERDICT})
    report = _write_report(tmp_path / _REPORT)

    comparison = _run(_settings(query_log_path), judge, tmp_path, minimal=True, report=report)

    assert len(judge.calls) == 2 * len(LIVE_EVAL_SCENARIOS)
    assert all(len(call) == 1 for call in judge.calls)
    first = comparison.cases[0].baselines[PARAMETRIC_VARIANT]
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
    bare = _payload(LIVE_EVAL_SCENARIOS[3], citations=1, summaries=False)
    judge = AgreeingJudge()
    report = _write_report(tmp_path / _REPORT)

    comparison = _run(
        _settings(query_log_path),
        judge,
        tmp_path,
        overrides_by_variant={
            PARAMETRIC_VARIANT: {_case_file(LIVE_EVAL_SCENARIOS[3]): bare},
            FULL_CORPUS_VARIANT: {_case_file(LIVE_EVAL_SCENARIOS[3]): bare},
        },
        report=report,
    )

    # Nine cases judge one pair each per variant; the bare ones floor
    # deterministically, never waking the judge for nothing.
    assert len(judge.calls) == 2 * (len(LIVE_EVAL_SCENARIOS) - 1)
    for side in (comparison.cases[3].baselines[PARAMETRIC_VARIANT], comparison.cases[3].baselines[FULL_CORPUS_VARIANT]):
        assert side.summary_fidelity == 0.0
        assert side.fidelity_verdicts == []
        assert side.summarizer_limitation is None


# --- The adapter: stored shape -> response shape -------------------------------


@pytest.mark.parametrize(
    ["marker"],
    [(PARAMETRIC_MARKER,), (FULL_CORPUS_MARKER,)],
    ids=list(VARIANTS),
)
def test_adapter_maps_the_stored_shape_through_the_answer_citations_builder(tmp_path, marker):
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

    response = baseline_eval.load_baseline_case(path, DropRecords(), marker)

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
    # The trace names the variant's provenance so artifacts never mix variants.
    assert response.trace.workflow == marker
    # Nothing is fabricated: no limitation, no trace detail, no quote.
    assert response.known_limitations == []
    assert response.detailed_trace == []
    assert citation.quote is None


# --- The two recorded exceptions (ADR-0017) -------------------------------------


def test_duplicate_summaries_resolve_first_wins_and_are_counted(tmp_path, query_log_path):
    override = _duplicate_summaries_override(_FIRST_CASE)
    label = _unique_expected_citations(_FIRST_CASE)[0]["provision"]
    dirs = _write_dirs(
        tmp_path, {PARAMETRIC_VARIANT: {_FIRST_FILE: override}}
    )
    judge = AgreeingJudge()
    report = _write_report(tmp_path / _REPORT)

    comparison = run_baseline_eval(
        _settings(query_log_path),
        judge=judge,
        variant_dirs=dirs,
        report_path=report,
    )

    produced = comparison.cases[0].baselines[PARAMETRIC_VARIANT].produced[0]["citations"][0]
    assert produced["relevance"] == "First summary stands."
    assert comparison.outcomes[PARAMETRIC_VARIANT].drops.duplicate_summaries == [
        baseline_eval.DroppedDuplicateSummary(file=dirs[PARAMETRIC_VARIANT] / _FIRST_FILE, label=f"{label}(12)")
    ]
    assert comparison.outcomes[PARAMETRIC_VARIANT].drops.unparseable_labels == []
    # The other variant's files are clean: no drops leak across variants.
    assert comparison.outcomes[FULL_CORPUS_VARIANT].drops.duplicate_summaries == []
    assert comparison.outcomes[FULL_CORPUS_VARIANT].drops.unparseable_labels == []


def test_unparseable_citation_label_is_dropped_recorded_and_counted(tmp_path, query_log_path):
    override = _payload(_FIRST_CASE, citations=1)
    override["findings"][0]["citations"].append({"source_id": "gdpr", "provision": "Section 12"})
    dirs = _write_dirs(
        tmp_path, {PARAMETRIC_VARIANT: {_FIRST_FILE: override}}
    )
    judge = AgreeingJudge()
    report = _write_report(tmp_path / _REPORT)

    comparison = run_baseline_eval(
        _settings(query_log_path),
        judge=judge,
        variant_dirs=dirs,
        report_path=report,
    )

    # Only the parseable citation produces; the junk label never flatters
    # precision and never voids the run.
    produced_citations = comparison.cases[0].baselines[PARAMETRIC_VARIANT].produced[0]["citations"]
    assert len(produced_citations) == 1
    assert comparison.cases[0].baselines[PARAMETRIC_VARIANT].precision == 1.0
    drops = comparison.outcomes[PARAMETRIC_VARIANT].drops.unparseable_labels
    assert len(drops) == 1
    assert drops[0].file == dirs[PARAMETRIC_VARIANT] / _FIRST_FILE
    assert drops[0].label == "Section 12"
    assert "parse" in drops[0].reason
    assert comparison.outcomes[PARAMETRIC_VARIANT].drops.duplicate_summaries == []


def test_unparseable_summary_label_is_dropped_and_counted(tmp_path, query_log_path):
    override = _payload(_FIRST_CASE, citations=1)
    override["summaries"] = [
        {"source_id": "gdpr", "provision": "Section 12", "relevance": "Junk label.", "strength": "moderate"},
    ]
    dirs = _write_dirs(
        tmp_path, {PARAMETRIC_VARIANT: {_FIRST_FILE: override}}
    )
    judge = AgreeingJudge()
    report = _write_report(tmp_path / _REPORT)

    comparison = run_baseline_eval(
        _settings(query_log_path),
        judge=judge,
        variant_dirs=dirs,
        report_path=report,
    )

    # The dropped summary leaves the cited provision bare: deterministic
    # floor, no judge call for that case on the parametric side.
    assert len(judge.calls) == 2 * len(LIVE_EVAL_SCENARIOS) - 1
    assert comparison.cases[0].baselines[PARAMETRIC_VARIANT].summary_fidelity == 0.0
    drops = comparison.outcomes[PARAMETRIC_VARIANT].drops.unparseable_labels
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
@pytest.mark.parametrize("variant_dir", VARIANTS)
def test_malformed_case_file_refuses_naming_file_and_problem(
    tmp_path, query_log_path, label: str, mutate: Callable[[dict], None], variant_dir: str
):
    payload = _payload(_FIRST_CASE)
    mutate(payload)
    judge = AgreeingJudge()
    report = _write_report(tmp_path / _REPORT)
    overrides = {_FIRST_FILE: payload}
    dirs = _write_dirs(tmp_path, {variant_dir: overrides})

    with pytest.raises(BaselineEvalRefused, match=_FIRST_FILE) as excinfo:
        run_baseline_eval(
            _settings(query_log_path),
            judge=judge,
            variant_dirs=dirs,
            report_path=report,
        )

    # The refusal names the problem, not just the file, and no judge call
    # was ever spent on a run that could not start.
    assert str(excinfo.value).strip() != str(dirs[PARAMETRIC_VARIANT] / _FIRST_FILE)
    assert judge.calls == []


@pytest.mark.parametrize("variant_dir", VARIANTS)
def test_stray_filename_outside_the_live_scenarios_refuses(tmp_path, query_log_path, variant_dir: str):
    overrides = {"not-a-scenario.json": _payload(_FIRST_CASE)}
    dirs = _write_dirs(tmp_path, {variant_dir: overrides})
    report = _write_report(tmp_path / _REPORT)

    with pytest.raises(BaselineEvalRefused, match="not-a-scenario"):
        run_baseline_eval(
            _settings(query_log_path),
            judge=AgreeingJudge(),
            variant_dirs=dirs,
            report_path=report,
        )


@pytest.mark.parametrize("variant_dir", VARIANTS)
def test_stray_non_json_file_is_never_silently_skipped(tmp_path, query_log_path, variant_dir: str):
    overrides = {"stray-notes.txt": "leftover export"}
    dirs = _write_dirs(tmp_path, {variant_dir: overrides})
    report = _write_report(tmp_path / _REPORT)

    with pytest.raises(BaselineEvalRefused, match="stray-notes"):
        run_baseline_eval(
            _settings(query_log_path),
            judge=AgreeingJudge(),
            variant_dirs=dirs,
            report_path=report,
        )


@pytest.mark.parametrize("variant_dir", VARIANTS)
def test_stray_subdirectory_is_never_silently_skipped(tmp_path, query_log_path, variant_dir: str):
    dirs = _write_dirs(tmp_path)
    dirs[variant_dir].joinpath("nested-export").mkdir()
    report = _write_report(tmp_path / _REPORT)

    with pytest.raises(BaselineEvalRefused, match="nested-export"):
        run_baseline_eval(
            _settings(query_log_path),
            judge=AgreeingJudge(),
            variant_dirs=dirs,
            report_path=report,
        )


@pytest.mark.parametrize("variant_dir", VARIANTS)
def test_missing_case_file_refuses_naming_it(tmp_path, query_log_path, variant_dir: str):
    overrides = {_FIRST_FILE: None}
    dirs = _write_dirs(tmp_path, {variant_dir: overrides})
    report = _write_report(tmp_path / _REPORT)

    with pytest.raises(BaselineEvalRefused, match=_FIRST_CASE.scenario_id):
        run_baseline_eval(
            _settings(query_log_path),
            judge=AgreeingJudge(),
            variant_dirs=dirs,
            report_path=report,
        )


@pytest.mark.parametrize("variant_dir", VARIANTS)
def test_unreadable_case_file_refuses(tmp_path, query_log_path, variant_dir: str):
    overrides = {_FIRST_FILE: "{not json"}
    dirs = _write_dirs(tmp_path, {variant_dir: overrides})
    report = _write_report(tmp_path / _REPORT)

    with pytest.raises(BaselineEvalRefused, match=_FIRST_FILE):
        run_baseline_eval(
            _settings(query_log_path),
            judge=AgreeingJudge(),
            variant_dirs=dirs,
            report_path=report,
        )


@pytest.mark.parametrize("variant_dir", VARIANTS)
def test_missing_baseline_directory_refuses(tmp_path, query_log_path, variant_dir: str):
    dirs = _write_dirs(tmp_path)
    dirs[variant_dir] = tmp_path / "nope"
    report = _write_report(tmp_path / _REPORT)

    with pytest.raises(BaselineEvalRefused, match="does not exist"):
        run_baseline_eval(
            _settings(query_log_path),
            judge=AgreeingJudge(),
            variant_dirs=dirs,
            report_path=report,
        )


def test_missing_api_key_refuses_when_the_default_judge_is_needed(tmp_path, query_log_path):
    dirs = _write_dirs(tmp_path)
    report = _write_report(tmp_path / _REPORT)
    settings = Settings(
        regula_mode=Mode.demo, openrouter_api_key=None, query_log_path=str(query_log_path)
    )

    with pytest.raises(BaselineEvalRefused, match="OPENROUTER_API_KEY"):
        run_baseline_eval(
            settings,
            variant_dirs=dirs,
            report_path=report,
        )


def test_judge_failure_aborts_the_run(tmp_path, query_log_path):
    report = _write_report(tmp_path / _REPORT)

    with pytest.raises(LlmUnreachableError, match="the provider is unreachable"):
        _run(_settings(query_log_path), UnreachableJudge(), tmp_path, report=report)


# --- The pipeline report side of the join ----------------------------------------


def test_report_missing_a_live_case_refuses(tmp_path, query_log_path):
    report = _write_report(tmp_path / _REPORT)
    data = json.loads(report.read_text())
    data["scenarios"] = data["scenarios"][:-1]
    report.write_text(json.dumps(data))
    last_case = LIVE_EVAL_SCENARIOS[-1]

    with pytest.raises(BaselineEvalRefused, match=last_case.id):
        _run(_settings(query_log_path), AgreeingJudge(), tmp_path, report=report)


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
    report = tmp_path / _REPORT
    report.write_text(content)

    with pytest.raises(BaselineEvalRefused, match=_REPORT) as excinfo:
        _run(_settings(query_log_path), AgreeingJudge(), tmp_path, report=report)

    assert fragment in str(excinfo.value)


# --- File-location defaults -------------------------------------------------------


def _point_defaults_at(monkeypatch, logs: Path, dirs: Mapping[str, Path]) -> None:
    monkeypatch.setattr(baseline_eval, "LOGS_DIR", logs)
    monkeypatch.setattr(baseline_eval, "PARAMETRIC_DIR", dirs[PARAMETRIC_VARIANT])
    monkeypatch.setattr(baseline_eval, "FULL_CORPUS_DIR", dirs[FULL_CORPUS_VARIANT])


def test_default_locations_resolve_to_the_canonical_dirs(tmp_path, monkeypatch, query_log_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    report = _write_report(logs / "live-eval-report-only.json")
    dirs = _write_dirs(tmp_path)
    _point_defaults_at(monkeypatch, logs, dirs)

    comparison = run_baseline_eval(_settings(query_log_path), judge=AgreeingJudge())

    assert comparison.report_path == report
    assert [case.baselines[PARAMETRIC_VARIANT].id for case in comparison.cases] == [
        case.id for case in LIVE_EVAL_SCENARIOS
    ]
    assert [case.baselines[FULL_CORPUS_VARIANT].id for case in comparison.cases] == [
        case.id for case in LIVE_EVAL_SCENARIOS
    ]


def test_newest_report_wins_by_default(tmp_path, monkeypatch, query_log_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_report(logs / "live-eval-report-2026-09-01-a.json")
    newest = _write_report(logs / "live-eval-report-2026-09-08-b.json")
    dirs = _write_dirs(tmp_path)
    _point_defaults_at(monkeypatch, logs, dirs)

    comparison = run_baseline_eval(_settings(query_log_path), judge=AgreeingJudge())

    assert comparison.report_path == newest


def test_newest_report_wins_by_mtime_not_alphabetical_order(tmp_path, monkeypatch, query_log_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    older = _write_report(logs / "live-eval-report-2026-09-10-glm53_v8.json")
    newest = _write_report(logs / "live-eval-report-2026-09-10-glm53_v10.json")
    # Pin the file times: `v10` sorts below `v8` lexicographically, so only
    # a newest-by-mtime rule can pick the file actually written last.
    os.utime(older, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newest, ns=(2_000_000_000, 2_000_000_000))
    dirs = _write_dirs(tmp_path)
    _point_defaults_at(monkeypatch, logs, dirs)

    comparison = run_baseline_eval(_settings(query_log_path), judge=AgreeingJudge())

    assert comparison.report_path == newest


def test_no_report_found_refuses_with_the_flag_hint(tmp_path, monkeypatch, query_log_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    dirs = _write_dirs(tmp_path)
    _point_defaults_at(monkeypatch, logs, dirs)

    with pytest.raises(BaselineEvalRefused, match="--report"):
        run_baseline_eval(_settings(query_log_path), judge=AgreeingJudge())


# --- The printer -------------------------------------------------------------------


def test_printer_prints_table_means_and_delta_footer(tmp_path, query_log_path, capsys):
    report = _write_report(
        tmp_path / _REPORT,
        entries={LIVE_EVAL_SCENARIOS[2].id: (0.5, 0.5, 0.5, 0.5)},
    )

    comparison = _run(_settings(query_log_path), AgreeingJudge(), tmp_path, report=report)
    baseline_eval.print_comparison(comparison)
    out = capsys.readouterr().out

    assert f"Pipeline report: {report} (model: z-ai/test-model)" in out
    assert _FIRST_CASE.id in out
    assert LIVE_EVAL_SCENARIOS[2].id in out
    # Pipeline row for the one scored case; both baseline rows are full marks.
    assert "0.500 / 0.500 / 0.500 / 0.500" in out
    assert "1.000 / 1.000 / 1.000 / 1.000" in out
    # All three columns are captioned, and the full-corpus column is the
    # distinct answering path it claims to be.
    assert "pipeline (precision / recall / F1 / fidelity)" in out
    assert f"{PARAMETRIC_VARIANT} (precision / recall / F1 / fidelity)" in out
    assert f"{FULL_CORPUS_VARIANT} (precision / recall / F1 / fidelity)" in out
    # Means and the Δ(pipeline − baseline) footer: coverage means over all
    # ten cases (0.5/10 = 0.050 against 1.000); fidelity means over the
    # measured cases only (one measured pipeline case: 0.500 against 1.000).
    assert (
        f"Mean coverage F1: pipeline 0.050 | {PARAMETRIC_VARIANT} 1.000 | "
        f"{FULL_CORPUS_VARIANT} 1.000 | Δ(pipeline − {PARAMETRIC_VARIANT}) -0.950 "
        f"| Δ(pipeline − {FULL_CORPUS_VARIANT}) -0.950"
        in out
    )
    assert (
        f"Mean summary fidelity: pipeline 0.500 | {PARAMETRIC_VARIANT} 1.000 | "
        f"{FULL_CORPUS_VARIANT} 1.000 | Δ(pipeline − {PARAMETRIC_VARIANT}) -0.500 "
        f"| Δ(pipeline − {FULL_CORPUS_VARIANT}) -0.500"
        in out
    )


def test_printer_keeps_the_variant_columns_apart(tmp_path, query_log_path, capsys):
    # The full-corpus fixture cites one target per case; the parametric one
    # covers everything: the two baseline columns must read differently.
    judge = AgreeingJudge()
    dirs = {
        PARAMETRIC_VARIANT: _write_variant_dir(tmp_path, PARAMETRIC_VARIANT),
        FULL_CORPUS_VARIANT: _write_variant_dir(tmp_path, FULL_CORPUS_VARIANT, minimal=True),
    }
    report = _write_report(tmp_path / _REPORT)

    comparison = run_baseline_eval(
        _settings(query_log_path),
        judge=judge,
        variant_dirs=dirs,
        report_path=report,
    )
    baseline_eval.print_comparison(comparison)
    out = capsys.readouterr().out

    first = comparison.cases[0]
    parametric_cell = (
        f"{first.baselines[PARAMETRIC_VARIANT].precision:.3f} / "
        f"{first.baselines[PARAMETRIC_VARIANT].recall:.3f} / "
        f"{first.baselines[PARAMETRIC_VARIANT].f1:.3f} / 1.000"
    )
    full_corpus_cell = (
        f"{first.baselines[FULL_CORPUS_VARIANT].precision:.3f} / "
        f"{first.baselines[FULL_CORPUS_VARIANT].recall:.3f} / "
        f"{first.baselines[FULL_CORPUS_VARIANT].f1:.3f} / 1.000"
    )
    assert parametric_cell in out
    assert full_corpus_cell in out
    assert parametric_cell != full_corpus_cell


def test_printer_handles_unmeasured_fidelity(tmp_path, query_log_path, capsys):
    judge = AgreeingJudge()
    dirs = _write_dirs(tmp_path)
    for directory in dirs.values():
        for case in LIVE_EVAL_SCENARIOS:
            (directory / _case_file(case)).write_text(
                json.dumps(_payload(case, citations=1, summaries=False))
            )
    report = _write_report(tmp_path / _REPORT)

    comparison = run_baseline_eval(
        _settings(query_log_path),
        judge=judge,
        variant_dirs=dirs,
        report_path=report,
    )
    baseline_eval.print_comparison(comparison)
    out = capsys.readouterr().out

    assert "n/a" in out
    assert f"Δ(pipeline − {PARAMETRIC_VARIANT}) n/a" in out
    assert f"Δ(pipeline − {FULL_CORPUS_VARIANT}) n/a" in out


# --- The combined artifact ----------------------------------------------------------


def _artifact(tmp_path, query_log_path, report=None):
    """A run plus its combined artifact, with the configured model named."""
    if report is None:
        report = _write_report(tmp_path / _REPORT)
    comparison = _run(
        _settings(query_log_path), AgreeingJudge(), tmp_path, report=report
    )
    return comparison, baseline_eval.combined_artifact(comparison, "z-ai/glm-5.3-flash")


def test_artifact_metadata_carries_both_models_note_and_case_inventory(tmp_path, query_log_path):
    report = _write_report(tmp_path / _REPORT, llm_model="z-ai/pipeline-model")
    _, artifact = _artifact(tmp_path, query_log_path, report=report)

    # The two models travel together with no consistency gate between them:
    # the configured model scored the baselines; the pipeline report's model
    # produced the left column.
    assert artifact["configured_model"] == "z-ai/glm-5.3-flash"
    assert artifact["pipeline_model"] == "z-ai/pipeline-model"
    # The two fidelity judges are named side by side: the pipeline report's
    # fidelity was judged by its own run's model, the baselines' by the
    # configured one — when they differ, the fidelity deltas compare across
    # judges, and the artifact says so.
    assert artifact["fidelity_judge_models"] == {
        "pipeline": "z-ai/pipeline-model",
        "baselines": "z-ai/glm-5.3-flash",
    }
    assert artifact["pipeline_report"] == str(report)
    # Generation time is a real UTC stamp.
    assert datetime.fromisoformat(artifact["generated_at"]).tzinfo is not None
    # The methodology note rides with the artifact: toolless production for
    # the scored runs, the tool-confounded predecessor pointed at ADR-0017.
    assert artifact["threat_note"] == baseline_eval.THREAT_NOTE
    assert "WebFetch" in artifact["threat_note"]
    assert "toolless" in artifact["threat_note"]
    # The case inventory is the ordered live case list the run joined on.
    assert artifact["case_inventory"] == [case.id for case in LIVE_EVAL_SCENARIOS]
    # The whole artifact serializes — paths and variants included.
    json.dumps(artifact)


def test_artifact_variants_carry_the_pipeline_report_shape(tmp_path, query_log_path):
    _, artifact = _artifact(tmp_path, query_log_path)

    for variant, marker in (
        (PARAMETRIC_VARIANT, PARAMETRIC_MARKER),
        (FULL_CORPUS_VARIANT, FULL_CORPUS_MARKER),
    ):
        variant_artifact = artifact["variants"][variant]
        assert variant_artifact["mode"] == "live"
        # The variant's Execution-trace marker rides the artifact, naming its
        # provenance, so artifacts never mix variants (issue #90, story 19).
        assert variant_artifact["execution_trace_marker"] == marker
        assert variant_artifact["mean_f1"] == pytest.approx(1.0)
        assert variant_artifact["mean_summary_fidelity"] == pytest.approx(1.0)
        assert len(variant_artifact["scenarios"]) == len(LIVE_EVAL_SCENARIOS)
        first = variant_artifact["scenarios"][0]
        assert first["id"] == _FIRST_CASE.id
        # The pipeline report's per-case shape: expected/produced dumps and
        # the per-pair fidelity verdicts, so the human audit works the same
        # for baselines.
        assert first["expected"]
        assert first["produced"][0]["statement"] == f"Baseline finding for {_FIRST_CASE.scenario_id}."
        assert first["fidelity_verdicts"][0]["score"] == 1.0
        assert first["summarizer_limitation"] is None


def test_artifact_comparison_table_data_matches_the_run(tmp_path, query_log_path):
    report = _write_report(
        tmp_path / _REPORT,
        entries={LIVE_EVAL_SCENARIOS[2].id: (0.5, 0.5, 0.5, 0.5)},
    )
    _, artifact = _artifact(tmp_path, query_log_path, report=report)

    table = artifact["comparison"]
    assert [row["case_id"] for row in table["cases"]] == [case.id for case in LIVE_EVAL_SCENARIOS]
    third = table["cases"][2]
    assert third["pipeline"] == {
        "precision": 0.5, "recall": 0.5, "f1": 0.5, "summary_fidelity": 0.5,
    }
    assert third[PARAMETRIC_VARIANT] == {
        "precision": 1.0, "recall": 1.0, "f1": 1.0, "summary_fidelity": 1.0,
    }
    assert third[FULL_CORPUS_VARIANT] == {
        "precision": 1.0, "recall": 1.0, "f1": 1.0, "summary_fidelity": 1.0,
    }
    # Means over all ten cases (coverage) and the measured ones (fidelity),
    # with the Δ(pipeline − baseline) footers as data.
    assert table["means"]["pipeline"] == {"f1": pytest.approx(0.05), "summary_fidelity": pytest.approx(0.5)}
    assert table["means"][PARAMETRIC_VARIANT] == {"f1": pytest.approx(1.0), "summary_fidelity": pytest.approx(1.0)}
    assert table["means"][FULL_CORPUS_VARIANT] == {"f1": pytest.approx(1.0), "summary_fidelity": pytest.approx(1.0)}
    assert table["delta_vs_pipeline"][PARAMETRIC_VARIANT] == {
        "f1": pytest.approx(-0.95), "summary_fidelity": pytest.approx(-0.5),
    }
    assert table["delta_vs_pipeline"][FULL_CORPUS_VARIANT] == {
        "f1": pytest.approx(-0.95), "summary_fidelity": pytest.approx(-0.5),
    }


def test_artifact_comparison_table_data_carries_unmeasured_fidelity_as_none(tmp_path, query_log_path):
    # The pipeline side's fidelity is unmeasured in the fixture report; the
    # baseline sides floor at zero deterministically. The unmeasured pipeline
    # mean must stay null — never dressed up as a zero — and the delta with it.
    _, artifact = _artifact(tmp_path, query_log_path)

    means = artifact["comparison"]["means"]
    assert means["pipeline"] == {"f1": pytest.approx(0.0), "summary_fidelity": None}
    assert means[PARAMETRIC_VARIANT]["summary_fidelity"] == pytest.approx(1.0)
    assert artifact["comparison"]["delta_vs_pipeline"][FULL_CORPUS_VARIANT]["summary_fidelity"] is None


def test_artifact_paths_travel_relative_to_the_repo_root(tmp_path, monkeypatch, query_log_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    report = _write_report(logs / _REPORT)
    dirs = _write_dirs(
        tmp_path, {PARAMETRIC_VARIANT: {_FIRST_FILE: _duplicate_summaries_override(_FIRST_CASE)}}
    )
    monkeypatch.setattr(baseline_eval, "REPO_ROOT", tmp_path.resolve())

    comparison = run_baseline_eval(
        _settings(query_log_path),
        judge=AgreeingJudge(),
        variant_dirs=dirs,
        report_path=report,
    )
    artifact = baseline_eval.combined_artifact(comparison, "z-ai/glm-5.3-flash")

    # Files inside the tree travel relative — the artifact stays portable —
    # and files outside it keep their absolute paths (every other test pins
    # that fallback through fixture directories outside the patched root).
    assert artifact["pipeline_report"] == f"logs/{_REPORT}"
    duplicate_record = artifact["drop_records"][PARAMETRIC_VARIANT]["duplicate_summaries"][0]
    assert duplicate_record["file"] == f"{PARAMETRIC_VARIANT}/{_FIRST_FILE}"


def test_artifact_surfaces_drop_counts_and_records_per_variant(tmp_path, query_log_path):
    duplicate = _duplicate_summaries_override(_FIRST_CASE)
    label = _unique_expected_citations(_FIRST_CASE)[0]["provision"]
    junk = _payload(_FIRST_CASE, citations=1)
    junk["findings"][0]["citations"].append({"source_id": "gdpr", "provision": "Section 12"})
    dirs = _write_dirs(
        tmp_path,
        {
            PARAMETRIC_VARIANT: {
                _FIRST_FILE: duplicate,
                _case_file(LIVE_EVAL_SCENARIOS[1]): junk,
            },
            FULL_CORPUS_VARIANT: {_FIRST_FILE: duplicate},
        },
    )
    report = _write_report(tmp_path / _REPORT)

    comparison = run_baseline_eval(
        _settings(query_log_path),
        judge=AgreeingJudge(),
        variant_dirs=dirs,
        report_path=report,
    )
    artifact = baseline_eval.combined_artifact(comparison, "z-ai/glm-5.3-flash")

    assert artifact["drops"][PARAMETRIC_VARIANT] == {"duplicate_summaries": 1, "unparseable_labels": 1}
    assert artifact["drops"][FULL_CORPUS_VARIANT] == {"duplicate_summaries": 1, "unparseable_labels": 0}
    parametric_records = artifact["drop_records"][PARAMETRIC_VARIANT]
    assert parametric_records["duplicate_summaries"][0] == {
        "file": str(dirs[PARAMETRIC_VARIANT] / _FIRST_FILE), "label": f"{label}(12)",
    }
    unparseable = parametric_records["unparseable_labels"][0]
    assert unparseable["label"] == "Section 12"
    assert unparseable["file"] == str(dirs[PARAMETRIC_VARIANT] / _case_file(LIVE_EVAL_SCENARIOS[1]))
    assert "parse" in unparseable["reason"]
    assert artifact["drop_records"][FULL_CORPUS_VARIANT]["unparseable_labels"] == []


def _default_artifact_path(logs: Path) -> Path:
    """The UTC-dated default artifact path the spec's convention names."""
    return logs / f"baseline-eval-report-{datetime.now(timezone.utc).date().isoformat()}.json"


def test_cli_success_writes_the_artifact_to_the_utc_dated_default(tmp_path, monkeypatch, capsys):
    logs = tmp_path / "logs"
    logs.mkdir()
    report = _write_report(logs / "live-eval-report-2026-09-08-b.json")
    dirs = _write_dirs(tmp_path)
    _wire_cli(monkeypatch, logs, dirs)
    monkeypatch.setattr("src.config.OpenRouterClient", _AllAgreeJudgeClient)

    exit_code = main([
        "--parametric-dir", str(dirs[PARAMETRIC_VARIANT]),
        "--full-corpus-dir", str(dirs[FULL_CORPUS_VARIANT]),
        "--report", str(report),
    ])

    assert exit_code == 0
    expected = _default_artifact_path(logs)
    assert expected.exists()
    artifact = json.loads(expected.read_text())
    assert artifact["configured_model"]
    assert artifact["pipeline_model"] == "z-ai/test-model"
    assert set(artifact["variants"]) == {PARAMETRIC_VARIANT, FULL_CORPUS_VARIANT}
    assert "comparison" in artifact
    out = capsys.readouterr().out
    assert f"Combined artifact written to {expected}" in out


def test_cli_output_flag_moves_the_artifact(tmp_path, monkeypatch, capsys):
    logs = tmp_path / "logs"
    logs.mkdir()
    report = _write_report(logs / _REPORT)
    dirs = _write_dirs(tmp_path)
    _wire_cli(monkeypatch, logs, dirs)
    monkeypatch.setattr("src.config.OpenRouterClient", _AllAgreeJudgeClient)
    output = tmp_path / "elsewhere" / "combined.json"

    exit_code = main([
        "--parametric-dir", str(dirs[PARAMETRIC_VARIANT]),
        "--full-corpus-dir", str(dirs[FULL_CORPUS_VARIANT]),
        "--report", str(report),
        "--output", str(output),
    ])

    assert exit_code == 0
    assert output.exists()
    artifact = json.loads(output.read_text())
    assert artifact["pipeline_report"] == str(report)


def test_cli_zero_arguments_uses_the_defaults(tmp_path, monkeypatch, capsys):
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_report(logs / "live-eval-report-2026-09-01-a.json")
    newest = _write_report(logs / "live-eval-report-2026-09-08-b.json")
    dirs = _write_dirs(tmp_path)
    _wire_cli(monkeypatch, logs, dirs)
    monkeypatch.setattr("src.config.OpenRouterClient", _AllAgreeJudgeClient)

    exit_code = main([])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert str(newest) in out
    expected = _default_artifact_path(logs)
    assert expected.exists()
    artifact = json.loads(expected.read_text())
    assert artifact["pipeline_report"] == str(newest)
    assert [row["case_id"] for row in artifact["comparison"]["cases"]] == [
        case.id for case in LIVE_EVAL_SCENARIOS
    ]


def test_cli_refusal_writes_no_artifact(tmp_path, monkeypatch, capsys):
    logs = tmp_path / "logs"
    logs.mkdir()
    report = _write_report(logs / _REPORT)
    dirs = _write_dirs(tmp_path, {PARAMETRIC_VARIANT: {_FIRST_FILE: None}})
    _wire_cli(monkeypatch, logs, dirs)

    exit_code = main([
        "--parametric-dir", str(dirs[PARAMETRIC_VARIANT]),
        "--full-corpus-dir", str(dirs[FULL_CORPUS_VARIANT]),
        "--report", str(report),
    ])

    assert exit_code == 1
    assert capsys.readouterr().out == ""
    assert not list(logs.glob("baseline-eval-report-*.json"))


def test_cli_judge_failure_writes_no_artifact(tmp_path, monkeypatch, capsys):
    logs = tmp_path / "logs"
    logs.mkdir()
    report = _write_report(logs / _REPORT)
    dirs = _write_dirs(tmp_path)
    _wire_cli(monkeypatch, logs, dirs)
    monkeypatch.setattr("src.config.OpenRouterClient", _DyingJudgeClient)

    exit_code = main([
        "--parametric-dir", str(dirs[PARAMETRIC_VARIANT]),
        "--full-corpus-dir", str(dirs[FULL_CORPUS_VARIANT]),
        "--report", str(report),
    ])

    assert exit_code == 1
    assert not list(logs.glob("baseline-eval-report-*.json"))


# --- The CLI ------------------------------------------------------------------------


def _wire_cli(monkeypatch, logs_dir: Path, dirs: Mapping[str, Path]) -> None:
    monkeypatch.setenv("REGULA_MODE", "demo")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    _point_defaults_at(monkeypatch, logs_dir, dirs)


def test_cli_success_prints_the_table_and_exits_zero(tmp_path, monkeypatch, capsys):
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_report(logs / "live-eval-report-2026-09-01-a.json")
    newest = _write_report(logs / "live-eval-report-2026-09-08-b.json")
    dirs = _write_dirs(tmp_path)
    _wire_cli(monkeypatch, logs, dirs)
    monkeypatch.setattr("src.config.OpenRouterClient", _AllAgreeJudgeClient)

    exit_code = main(["--parametric-dir", str(dirs[PARAMETRIC_VARIANT]), "--full-corpus-dir", str(dirs[FULL_CORPUS_VARIANT])])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert (
        f"Baseline comparison: pipeline vs {PARAMETRIC_VARIANT} vs {FULL_CORPUS_VARIANT} (10 case(s))"
        in out
    )
    assert str(newest) in out
    assert "z-ai/test-model" in out
    assert _FIRST_CASE.id in out


def test_cli_refusal_exits_1_naming_the_file(tmp_path, monkeypatch, capsys):
    logs = tmp_path / "logs"
    logs.mkdir()
    report = _write_report(logs / _REPORT)
    dirs = _write_dirs(tmp_path, {PARAMETRIC_VARIANT: {_FIRST_FILE: None}})
    _wire_cli(monkeypatch, logs, dirs)

    exit_code = main([
        "--parametric-dir", str(dirs[PARAMETRIC_VARIANT]),
        "--full-corpus-dir", str(dirs[FULL_CORPUS_VARIANT]),
        "--report", str(report),
    ])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Baseline eval refused" in captured.err
    assert _FIRST_CASE.scenario_id in captured.err
    assert captured.out == ""


def test_cli_judge_failure_aborts_with_exit_1(tmp_path, monkeypatch, capsys):
    logs = tmp_path / "logs"
    logs.mkdir()
    report = _write_report(logs / _REPORT)
    dirs = _write_dirs(tmp_path)
    _wire_cli(monkeypatch, logs, dirs)
    monkeypatch.setattr("src.config.OpenRouterClient", _DyingJudgeClient)

    exit_code = main([
        "--parametric-dir", str(dirs[PARAMETRIC_VARIANT]),
        "--full-corpus-dir", str(dirs[FULL_CORPUS_VARIANT]),
        "--report", str(report),
    ])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Baseline eval aborted" in captured.err
    assert "summary-fidelity judge" in captured.err
    assert "provider down mid-run" in captured.err
