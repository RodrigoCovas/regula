"""Baseline-eval operator command (spec #90, tickets #91–#92): score both
baseline variants with the shared eval harness and print the comparison
against the newest pipeline report.

The Baseline comparison measures what the workflow's machinery adds by
scoring answers produced without it. Both variants' answers are stored — one
structured completion per live case, generated in the OpenCode agent harness
— under ``logs/baseline-runs/<variant>/`` (``parametric/``,
``full-corpus/``) as ``<scenario-id>.json`` in the stored template shape
``{findings, summaries, actions}`` (ADR-0017). This command loads each
variant's file for the ten live cases
(``eval_harness.LIVE_EVAL_SCENARIOS``), validates it strictly — wrong keys,
wrong types, invalid strength, unknown source id, missing or unreadable
file, or a filename outside the ten live Scenario ids refuse the run naming
file and problem — and maps every validated file into the response shape the
shared evaluation harness consumes: Findings with Citations derived from the
parsed provision labels (sub-references fold onto the provision they
refine), Provision relevance and Citation strength attached through the
existing answer-citations builder, and the standing seek-counsel Action
appended last. Each variant's Execution-trace marker names its provenance
(``baseline-parametric: agent-produced (OpenCode harness)`` /
``baseline-full-corpus: agent-produced (OpenCode harness)``).

Scoring is the shared loop itself (``evaluate_scenarios``, mode pinned to
live): the same provision-coverage scorer and the same strict fidelity judge
a pipeline run pays — the judge constructed exactly like the live-eval
runner's, from the configured provider. A judge or provider failure aborts
the run: infrastructure failure is never measured quality. The pipeline
side of the comparison is never re-run — the per-case numbers are read from
the newest ``live-eval-report-*.json`` (or the path given via ``--report``)
and joined on the live case list.

Two domain rules are the recorded exceptions to strictness (ADR-0017):
duplicate summaries for the same structural target resolve first-wins — the
pipeline's own duplicate gate — with later duplicates recorded and counted;
a citation or summary whose provision label does not parse is dropped,
recorded, and counted, so one junk label neither voids the run nor quietly
flatters precision. The drop counts ride the result per variant and surface
in the combined artifact.

The two exceptions aside, nothing is normalized inside the pipeline:
malformed files are fixed by hand before the comparison runs.

Run from the repository root:

    python -m backend.src.baseline_eval [--report PATH] [--parametric-dir PATH]
        [--full-corpus-dir PATH] [--output PATH]

and inside the stack:

    docker compose exec backend python -m backend.src.baseline_eval

Zero arguments is the day-to-day invocation: the newest
``live-eval-report-*.json`` in the logs tree is compared against the
canonical variant directories, and the combined artifact is written to
``logs/baseline-eval-report-<UTC-date>.json``. ``--output`` moves the
artifact.

Refusals exit 1; a judge or provider failure mid-run aborts with exit 1;
Ctrl-C pauses with exit 130; success prints the comparison table — per-case
precision, recall, coverage F1, and summary fidelity for pipeline,
parametric, and full-corpus — with the per-variant means and the
Δ(pipeline − baseline) footer.
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from .config import ConfigurationError, Settings, chat_client, load_settings
from .corpus import load_documents
from .eval_harness import (
    LIVE_EVAL_SCENARIOS,
    EvalReport,
    EvalScenarioResult,
    evaluate_scenarios,
    format_score,
    parse_provision,
)
from .eval_judge import SummaryFidelityJudge, SummaryJudge
from .llm import LlmError
from .live_eval import _judge_naming_its_failures
from .live_workflow import SEEK_COUNSEL_ACTION
from .models import (
    PROVISION_NUMBER_FIELDS,
    AnalyzeRequest,
    AnalyzeResponse,
    Answer,
    Citation,
    Finding,
    GroundedSummary,
    Mode,
    ProvisionTarget,
    Strength,
    Trace,
    answer_citations,
)


class BaselineEvalRefused(RuntimeError):
    """The baseline comparison cannot start; the message names the file and the problem."""


# Canonical file locations (ADR-0017): module constants so tests can point
# them at fixture directories. The directory names carry the canonical
# spellings; the misspelled `parametrized/` was renamed before any artifact
# referenced it.
LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"
BASELINE_DIR = LOGS_DIR / "baseline-runs" / "parametric"
FULL_CORPUS_DIR = LOGS_DIR / "baseline-runs" / "full-corpus"
_REPORT_GLOB = "live-eval-report-*.json"

# The artifact's name follows the spec's convention: UTC-dated, under the
# logs tree beside the pipeline reports it compares against.
_ARTIFACT_STEM = "baseline-eval-report"

# The ADR-0017 threat note, verbatim: it must ride the artifact so the
# numbers are never quoted without their caveat.
THREAT_NOTE = (
    "The current runs were produced by agents with tools such as WebFetch "
    "available, which is a threat to validity: a Parametric baseline with "
    "web access is not strictly parametric. The runs are kept and scored "
    "anyway, the threat is recorded with the artifact, and toolless runs "
    "(an agent with every tool denied, or raw API calls) remain the planned "
    "follow-up before any external claim."
)

# Each variant's Execution-trace marker (spec #90): provenance travels with
# every answer so artifacts never mix variants. The names are the canonical
# directory spellings (ADR-0017).
PARAMETRIC_VARIANT = "parametric"
FULL_CORPUS_VARIANT = "full-corpus"
PARAMETRIC_MARKER = "baseline-parametric: agent-produced (OpenCode harness)"
FULL_CORPUS_MARKER = "baseline-full-corpus: agent-produced (OpenCode harness)"

_TRACE_SUMMARY = "Agent-produced baseline answer; no workflow stages ran."

# The stored template shape (ADR-0017), validated key-exact — nothing is
# normalized, so a shape mismatch refuses instead of repairing.
_STORED_KEYS = {"findings", "summaries", "actions"}
_FINDING_KEYS = {"statement", "strength", "citations"}
_CITATION_KEYS = {"source_id", "provision"}
_SUMMARY_KEYS = {"source_id", "provision", "relevance", "strength"}


@dataclass
class StoredCitation:
    """One validated stored citation: source id and provision label as written."""

    source_id: str
    provision: str


@dataclass
class StoredFinding:
    """One validated stored finding: a non-empty statement, a bare Strength,
    and a non-empty citation list."""

    statement: str
    strength: Strength
    citations: List[StoredCitation]


@dataclass
class StoredSummary:
    """One validated stored summary: the cited provision, its relevance
    statement, and the provision's rated Citation strength."""

    source_id: str
    provision: str
    relevance: str
    strength: Strength


@dataclass
class CaseFile:
    """One validated baseline case file, ready for the adapter."""

    findings: List[StoredFinding]
    summaries: List[StoredSummary]
    actions: List[str]


@dataclass
class DroppedDuplicateSummary:
    """One later duplicate summary dropped by the first-wins gate (ADR-0017)."""

    file: Path
    label: str


@dataclass
class DroppedLabel:
    """One citation or summary label that parsed to no provision (ADR-0017)."""

    file: Path
    label: str
    reason: str


@dataclass
class DropRecords:
    """The two deterministic drop accounts (ADR-0017), recorded per file so
    the run's resolutions stay visible after the fact."""

    duplicate_summaries: List[DroppedDuplicateSummary] = field(default_factory=list)
    unparseable_labels: List[DroppedLabel] = field(default_factory=list)


@dataclass
class PipelineCaseResult:
    """One case's numbers read from the pipeline report artifact — never re-run."""

    id: str
    precision: float
    recall: float
    f1: float
    summary_fidelity: Optional[float]


@dataclass
class ComparisonCase:
    """One live case joined across the three answering paths."""

    case_id: str
    pipeline: PipelineCaseResult
    parametric: EvalScenarioResult
    full_corpus: EvalScenarioResult


@dataclass
class BaselineComparison:
    """The run's outcome: the joined per-case comparison, each variant's
    report in the shared shape, and each variant's drop records (the
    combined artifact surfaces them in full)."""

    report_path: Path
    pipeline_model: Optional[str]
    cases: List[ComparisonCase]
    parametric_report: EvalReport
    full_corpus_report: EvalReport
    parametric_drops: DropRecords
    full_corpus_drops: DropRecords


def _refuse(path: Path, problem: str) -> BaselineEvalRefused:
    """The one refusal shape: every message names the file and the problem."""
    return BaselineEvalRefused(f"{path}: {problem}")


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise _refuse(path, f"cannot be read as JSON ({error})") from error


def _non_empty_str(path: Path, where: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise _refuse(path, f"{where} must be a non-empty string")
    return value


def _strength(path: Path, where: str, value: object) -> Strength:
    if not isinstance(value, str):
        raise _refuse(path, f"{where} must be a bare 'strong'/'moderate'/'weak' string")
    try:
        return Strength(value)
    except ValueError:
        raise _refuse(
            path,
            f"{where} carries invalid strength {value!r} — "
            "expected 'strong', 'moderate', or 'weak'",
        ) from None


def _source_id(path: Path, where: str, value: object, known: set[str]) -> str:
    source = _non_empty_str(path, f"{where} source_id", value)
    if source not in known:
        raise _refuse(
            path,
            f"{where} names unknown source id {source!r} "
            f"(known sources: {', '.join(sorted(known))})",
        )
    return source


def _exact_keys(path: Path, where: str, value: object, keys: set[str]) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise _refuse(path, f"{where} must carry exactly the keys {sorted(keys)}")
    return value


def _validate_case_file(path: Path, data: object, known_sources: set[str]) -> CaseFile:
    """Validate one stored case file against the template shape, key-exact
    and type-exact: a mismatch refuses the run naming file and problem —
    no normalization (ADR-0017)."""
    if not isinstance(data, dict):
        raise _refuse(path, "the file is not a JSON object")
    if set(data) != _STORED_KEYS:
        raise _refuse(path, f"expected exactly the keys {sorted(_STORED_KEYS)}, got {sorted(data)}")

    raw_findings = data["findings"]
    if not isinstance(raw_findings, list):
        raise _refuse(path, "'findings' must be a list")
    findings = []
    for index, raw_finding in enumerate(raw_findings):
        where = f"finding {index}"
        finding = _exact_keys(path, where, raw_finding, _FINDING_KEYS)
        statement = _non_empty_str(path, f"{where} statement", finding["statement"])
        strength = _strength(path, where, finding["strength"])
        raw_citations = finding["citations"]
        if not isinstance(raw_citations, list) or not raw_citations:
            raise _refuse(path, f"{where} must carry a non-empty citations list")
        citations = []
        for c_index, raw_citation in enumerate(raw_citations):
            c_where = f"{where} citation {c_index}"
            citation = _exact_keys(path, c_where, raw_citation, _CITATION_KEYS)
            citations.append(StoredCitation(
                source_id=_source_id(path, c_where, citation["source_id"], known_sources),
                provision=_non_empty_str(path, f"{c_where} provision", citation["provision"]),
            ))
        findings.append(StoredFinding(statement=statement, strength=strength, citations=citations))

    raw_summaries = data["summaries"]
    if not isinstance(raw_summaries, list):
        raise _refuse(path, "'summaries' must be a list")
    summaries = []
    for index, raw_summary in enumerate(raw_summaries):
        where = f"summary {index}"
        summary = _exact_keys(path, where, raw_summary, _SUMMARY_KEYS)
        summaries.append(StoredSummary(
            source_id=_source_id(path, where, summary["source_id"], known_sources),
            provision=_non_empty_str(path, f"{where} provision", summary["provision"]),
            relevance=_non_empty_str(path, f"{where} relevance", summary["relevance"]),
            strength=_strength(path, where, summary["strength"]),
        ))

    raw_actions = data["actions"]
    if not isinstance(raw_actions, list) or not all(isinstance(action, str) for action in raw_actions):
        raise _refuse(path, "'actions' must be a list of strings")

    return CaseFile(
        findings=findings,
        summaries=summaries,
        actions=list(raw_actions),
    )


def _citation_for(target: ProvisionTarget, label: str) -> Citation:
    """One derived Citation targeting the parsed provision: the exactly-one
    number field comes from the kind (``PROVISION_NUMBER_FIELDS``)."""
    return Citation.model_validate({
        "source_id": target.source_id,
        "provision": label,
        PROVISION_NUMBER_FIELDS[target.kind]: target.number,
    })


def load_baseline_case(
    path: Path,
    drops: DropRecords,
    marker: str,
    known_sources: Optional[set[str]] = None,
) -> AnalyzeResponse:
    """Load, validate, and adapt one stored baseline case file into the
    response shape the shared harness consumes (ADR-0017).

    ``marker`` is the variant's Execution-trace marker — the only thing that
    differs between the variants, so a case never loses its provenance.
    ``known_sources`` is the set of Corpus source ids a citation may name;
    it loads from the Corpus when omitted — a run hoists the set and pays
    for it once.

    Citations derive from the parsed provision labels — the same
    Article/Recital/Annex grammar the ground truth uses, with
    sub-references folding onto the provision they refine. A label that
    does not parse is dropped, recorded, and counted rather than voiding
    the run; duplicate summaries for one structural target resolve
    first-wins with later duplicates recorded and counted — the pipeline's
    own duplicate gate. Provision relevance and Citation strength attach
    through the existing answer-citations builder, the standing seek-counsel
    Action is appended last, and the trace names the variant's provenance.
    Nothing else is normalized: no quote, limitation, or trace detail is
    ever fabricated.
    """
    sources = known_sources if known_sources is not None else set(load_documents())
    case = _validate_case_file(path, _read_json(path), sources)

    findings: List[Finding] = []
    for stored in case.findings:
        citations: List[Citation] = []
        for stored_citation in stored.citations:
            try:
                target = parse_provision(stored_citation.source_id, stored_citation.provision)
            except ValueError as error:
                drops.unparseable_labels.append(
                    DroppedLabel(file=path, label=stored_citation.provision, reason=str(error))
                )
                continue
            citations.append(_citation_for(target, stored_citation.provision))
        findings.append(Finding(statement=stored.statement, strength=stored.strength, citations=citations))

    summaries: Dict[ProvisionTarget, GroundedSummary] = {}
    for stored_summary in case.summaries:
        try:
            target = parse_provision(stored_summary.source_id, stored_summary.provision)
        except ValueError as error:
            drops.unparseable_labels.append(
                DroppedLabel(file=path, label=stored_summary.provision, reason=str(error))
            )
            continue
        if target in summaries:
            drops.duplicate_summaries.append(
                DroppedDuplicateSummary(file=path, label=stored_summary.provision)
            )
            continue
        summaries[target] = GroundedSummary(
            target=target,
            relevance=stored_summary.relevance,
            strength=stored_summary.strength,
        )

    answer = Answer(
        findings=findings,
        actions=[*case.actions, SEEK_COUNSEL_ACTION],
        citations=answer_citations(findings, summaries),
    )
    return AnalyzeResponse(
        answer=answer,
        trace=Trace(workflow=marker, summary=_TRACE_SUMMARY),
    )


def _discover_case_files(directory: Path) -> Dict[str, Path]:
    """The ten case files keyed by Scenario id: every entry in the directory
    must be a case file named for a live Scenario id — a typo'd or stray
    file or directory refuses the run and can never be silently skipped —
    and all ten must be present."""
    if not directory.is_dir():
        raise BaselineEvalRefused(f"Baseline directory {directory} does not exist")
    scenario_ids = {case.scenario_id for case in LIVE_EVAL_SCENARIOS}
    files: Dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix != ".json" or path.stem not in scenario_ids:
            raise BaselineEvalRefused(
                f"{path} is not one of the ten live Scenario case files — rename or remove it "
                f"(expected exactly ten files named <scenario-id>.json: "
                f"{', '.join(sorted(scenario_ids))})"
            )
        files[path.stem] = path
    missing = [case.scenario_id for case in LIVE_EVAL_SCENARIOS if case.scenario_id not in files]
    if missing:
        raise BaselineEvalRefused(
            f"Missing baseline case file(s) in {directory} for: {', '.join(missing)}"
        )
    return files


def _newest_report(logs_dir: Path) -> Path:
    reports = sorted(logs_dir.glob(_REPORT_GLOB))
    if not reports:
        raise BaselineEvalRefused(
            f"No pipeline report found in {logs_dir} (no {_REPORT_GLOB} files). "
            "Run the live eval first or pass --report PATH."
        )
    return reports[-1]


def _numeric(path: Path, where: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _refuse(path, f"{where} must be a number")
    return float(value)


def _parse_report(path: Path) -> tuple[Optional[str], Dict[str, PipelineCaseResult]]:
    """Read the pipeline report artifact: the report's model plus its
    per-case numbers keyed by case id. Anything malformed refuses naming
    file and problem — a comparison over a half-readable report would be
    quote-worthy nonsense."""
    data = _read_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        raise _refuse(path, "the report must be a JSON object with a 'scenarios' list")
    model = data.get("llm_model")
    if model is not None and not isinstance(model, str):
        raise _refuse(path, "'llm_model' must be a string")
    results: Dict[str, PipelineCaseResult] = {}
    for entry in data["scenarios"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise _refuse(path, "every scenario entry needs a string 'id'")
        if entry["id"] in results:
            raise _refuse(path, f"two scenario entries carry id {entry['id']!r}")
        results[entry["id"]] = PipelineCaseResult(
            id=entry["id"],
            precision=_numeric(path, f"case {entry['id']!r} precision", entry.get("precision")),
            recall=_numeric(path, f"case {entry['id']!r} recall", entry.get("recall")),
            f1=_numeric(path, f"case {entry['id']!r} f1", entry.get("f1")),
            summary_fidelity=(
                None
                if entry.get("summary_fidelity") is None
                else _numeric(path, f"case {entry['id']!r} summary_fidelity", entry["summary_fidelity"])
            ),
        )
    return model, results


def _default_judge(settings: Settings) -> SummaryJudge:
    """The judge constructed exactly like the pipeline runner's: the
    configured provider through the shared chat-client recipe, with its
    failures named as the judge's."""
    if not settings.openrouter_api_key:
        raise BaselineEvalRefused(
            "OPENROUTER_API_KEY is not set. Export it (or put it in "
            "backend/.env.local) and re-run."
        )
    return _judge_naming_its_failures(SummaryFidelityJudge(chat_client(settings)))


def _adapt_variant(
    files: Dict[str, Path],
    marker: str,
    known_sources: set[str],
) -> tuple[Dict[str, AnalyzeResponse], DropRecords]:
    """One variant's files adapted into the response shape, drops recorded on
    the variant's own account. Validation happens here — before any judge
    call is spent — so a malformed file anywhere refuses the run for free."""
    drops = DropRecords()
    responses: Dict[str, AnalyzeResponse] = {
        scenario_id: load_baseline_case(path, drops, marker, known_sources)
        for scenario_id, path in files.items()
    }
    return responses, drops


def _evaluate_variant(
    responses: Dict[str, AnalyzeResponse],
    judge: SummaryJudge,
) -> EvalReport:
    """Score one adapted variant over the ten live cases with the shared
    loop, pinned to live mode — the same scoring a pipeline run pays."""

    def respond(request: AnalyzeRequest) -> AnalyzeResponse:
        # ``responses`` covers exactly the live Scenario ids the harness requests.
        return responses[request.scenario.id or ""]

    return evaluate_scenarios(
        LIVE_EVAL_SCENARIOS,
        respond,
        mode=Mode.live,
        judge=judge,
    )


def run_baseline_eval(
    settings: Settings,
    judge: Optional[SummaryJudge] = None,
    parametric_dir: Optional[Path] = None,
    full_corpus_dir: Optional[Path] = None,
    report_path: Optional[Path] = None,
) -> BaselineComparison:
    """Score both baseline variants over the ten live cases with the shared
    harness and join them against the pipeline report.

    Everything validates before anything is measured: the case files are
    discovered, validated, and adapted first, and the report is parsed and
    checked for the live case list — a refusal never spends a judge call.
    The fidelity judge defaults to the configured provider (ADR-0009) built
    exactly like the live-eval runner's; tests script the seam by passing
    one. Mode is pinned to live per case (artifact-shape compatibility), the
    pipeline numbers are read from the report, never re-run, and the join
    follows the live case list: a report lacking one of the ten cases
    refuses rather than comparing a subset. There is no model-consistency
    gate — the operator controls model matching manually — so both the
    configured model and the pipeline report's model travel with the
    result's metadata.

    ``parametric_dir``, ``full_corpus_dir``, and ``report_path`` override
    the canonical file locations (tests; the CLI passes its flags); the
    judge seam is injectable for the same reason.
    """
    parametric = parametric_dir if parametric_dir is not None else BASELINE_DIR
    full_corpus = full_corpus_dir if full_corpus_dir is not None else FULL_CORPUS_DIR
    parametric_files = _discover_case_files(parametric)
    full_corpus_files = _discover_case_files(full_corpus)
    resolved_report = report_path if report_path is not None else _newest_report(LOGS_DIR)
    pipeline_model, pipeline_by_id = _parse_report(resolved_report)
    missing = [case.id for case in LIVE_EVAL_SCENARIOS if case.id not in pipeline_by_id]
    if missing:
        raise BaselineEvalRefused(
            f"Pipeline report {resolved_report} lacks case(s) {', '.join(missing)} — "
            "the comparison joins on the live case list."
        )

    fidelity_judge = judge if judge is not None else _default_judge(settings)
    known_sources = set(load_documents())
    parametric_responses, parametric_drops = _adapt_variant(
        parametric_files, PARAMETRIC_MARKER, known_sources
    )
    full_corpus_responses, full_corpus_drops = _adapt_variant(
        full_corpus_files, FULL_CORPUS_MARKER, known_sources
    )
    parametric_report = _evaluate_variant(parametric_responses, fidelity_judge)
    full_corpus_report = _evaluate_variant(full_corpus_responses, fidelity_judge)

    cases = [
        ComparisonCase(
            case_id=case.id,
            pipeline=pipeline_by_id[case.id],
            parametric=parametric_result,
            full_corpus=full_corpus_result,
        )
        for case, parametric_result, full_corpus_result in zip(
            LIVE_EVAL_SCENARIOS, parametric_report.scenarios, full_corpus_report.scenarios
        )
    ]
    return BaselineComparison(
        report_path=resolved_report,
        pipeline_model=pipeline_model,
        cases=cases,
        parametric_report=parametric_report,
        full_corpus_report=full_corpus_report,
        parametric_drops=parametric_drops,
        full_corpus_drops=full_corpus_drops,
    )


def _mean_of_measured(values: Iterable[Optional[float]]) -> Optional[float]:
    """The mean over the measured values only — an unmeasured component is
    never dressed up as a zero."""
    measured = [value for value in values if value is not None]
    return sum(measured) / len(measured) if measured else None


def _delta_value(pipeline: Optional[float], baseline: Optional[float]) -> Optional[float]:
    """The numeric Δ(pipeline − baseline), unmeasured where either side is."""
    if pipeline is None or baseline is None:
        return None
    return pipeline - baseline


def _delta(pipeline: Optional[float], baseline: Optional[float]) -> str:
    value = _delta_value(pipeline, baseline)
    return "n/a" if value is None else f"{value:+.3f}"


def _pipeline_means(cases: List[ComparisonCase]) -> tuple[float, Optional[float]]:
    """The pipeline side's aggregate pair — the numbers both the printed
    footer and the artifact's table data quote: coverage F1 over all the
    joined cases, fidelity over the measured ones only."""
    return (
        sum(case.pipeline.f1 for case in cases) / len(cases),
        _mean_of_measured(case.pipeline.summary_fidelity for case in cases),
    )


def print_comparison(comparison: BaselineComparison) -> None:
    """The operator-facing output: one row per case — precision, recall,
    coverage F1, and summary fidelity for pipeline, parametric, and
    full-corpus — then the per-variant means and the Δ(pipeline − baseline)
    footer. The pipeline report that produced the left column is named
    alongside its model."""
    cases = comparison.cases
    print(f"Baseline comparison: pipeline vs parametric vs full-corpus ({len(cases)} case(s))")
    model = comparison.pipeline_model or "unknown model"
    print(f"Pipeline report: {comparison.report_path} (model: {model})")
    print()
    pipeline_caption = "pipeline (precision / recall / F1 / fidelity)"
    parametric_caption = "parametric (precision / recall / F1 / fidelity)"
    print(
        f"  {'case':<40}{pipeline_caption:<48}"
        f"{parametric_caption:<44}full-corpus (precision / recall / F1 / fidelity)"
    )
    for case in cases:
        pipeline_cell = (
            f"{case.pipeline.precision:.3f} / {case.pipeline.recall:.3f} / "
            f"{case.pipeline.f1:.3f} / {format_score(case.pipeline.summary_fidelity)}"
        )
        parametric_cell = (
            f"{case.parametric.precision:.3f} / {case.parametric.recall:.3f} / "
            f"{case.parametric.f1:.3f} / {format_score(case.parametric.summary_fidelity)}"
        )
        full_corpus_cell = (
            f"{case.full_corpus.precision:.3f} / {case.full_corpus.recall:.3f} / "
            f"{case.full_corpus.f1:.3f} / {format_score(case.full_corpus.summary_fidelity)}"
        )
        print(
            f"  {case.case_id:<40}{pipeline_cell:<48}{parametric_cell:<44}{full_corpus_cell}"
        )
    print()
    pipeline_f1, pipeline_fidelity = _pipeline_means(cases)
    parametric_f1 = comparison.parametric_report.mean_f1
    full_corpus_f1 = comparison.full_corpus_report.mean_f1
    parametric_fidelity = comparison.parametric_report.mean_summary_fidelity
    full_corpus_fidelity = comparison.full_corpus_report.mean_summary_fidelity
    print(
        f"  Mean coverage F1: pipeline {pipeline_f1:.3f} | parametric {parametric_f1:.3f} | "
        f"full-corpus {full_corpus_f1:.3f} | "
        f"Δ(pipeline − parametric) {_delta(pipeline_f1, parametric_f1)} | "
        f"Δ(pipeline − full-corpus) {_delta(pipeline_f1, full_corpus_f1)}"
    )
    print(
        f"  Mean summary fidelity: pipeline {format_score(pipeline_fidelity)} | "
        f"parametric {format_score(parametric_fidelity)} | "
        f"full-corpus {format_score(full_corpus_fidelity)} | "
        f"Δ(pipeline − parametric) {_delta(pipeline_fidelity, parametric_fidelity)} | "
        f"Δ(pipeline − full-corpus) {_delta(pipeline_fidelity, full_corpus_fidelity)}"
    )


def _default_output_path() -> Path:
    """The UTC-dated artifact path the spec's convention names, under the
    logs tree beside the pipeline reports it compares against."""
    date = datetime.now(timezone.utc).date().isoformat()
    return LOGS_DIR / f"{_ARTIFACT_STEM}-{date}.json"


def _cell(result: Union[PipelineCaseResult, EvalScenarioResult]) -> dict:
    """One answering path's four component numbers, as the table shows them.
    Both result types carry the same four fields — the pipeline's read from
    the report artifact, the baselines' measured by the shared loop."""
    return {
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "summary_fidelity": result.summary_fidelity,
    }


def _drops_counts(drops: DropRecords) -> dict:
    return {
        "duplicate_summaries": len(drops.duplicate_summaries),
        "unparseable_labels": len(drops.unparseable_labels),
    }


def _drops_records(drops: DropRecords) -> dict:
    """The two drop accounts as plain data — file, label, and (for labels)
    the parse failure — so the deterministic resolutions stay visible."""
    return {
        "duplicate_summaries": [
            {"file": str(duplicate.file), "label": duplicate.label}
            for duplicate in drops.duplicate_summaries
        ],
        "unparseable_labels": [
            {"file": str(dropped.file), "label": dropped.label, "reason": dropped.reason}
            for dropped in drops.unparseable_labels
        ],
    }


def _variant_artifact(report: EvalReport) -> dict:
    """One variant's report in the pipeline report's shape: per-case
    expected/produced dumps and per-pair fidelity verdicts (issue #83's
    persistence rule, inherited unchanged), with the mode the runs pinned."""
    return {"mode": Mode.live.value, **asdict(report)}


def _comparison_artifact(comparison: BaselineComparison) -> dict:
    """The table data: per-case rows for all three answering paths, the
    per-variant means, and the Δ(pipeline − baseline) footers as data —
    the single place to quote."""
    cases = comparison.cases
    pipeline_f1, pipeline_fidelity = _pipeline_means(cases)
    means = {
        "pipeline": {"f1": pipeline_f1, "summary_fidelity": pipeline_fidelity},
        PARAMETRIC_VARIANT: {
            "f1": comparison.parametric_report.mean_f1,
            "summary_fidelity": comparison.parametric_report.mean_summary_fidelity,
        },
        FULL_CORPUS_VARIANT: {
            "f1": comparison.full_corpus_report.mean_f1,
            "summary_fidelity": comparison.full_corpus_report.mean_summary_fidelity,
        },
    }
    return {
        "cases": [
            {
                "case_id": case.case_id,
                "pipeline": _cell(case.pipeline),
                PARAMETRIC_VARIANT: _cell(case.parametric),
                FULL_CORPUS_VARIANT: _cell(case.full_corpus),
            }
            for case in cases
        ],
        "means": means,
        "delta_vs_pipeline": {
            PARAMETRIC_VARIANT: {
                "f1": _delta_value(pipeline_f1, comparison.parametric_report.mean_f1),
                "summary_fidelity": _delta_value(
                    pipeline_fidelity, comparison.parametric_report.mean_summary_fidelity
                ),
            },
            FULL_CORPUS_VARIANT: {
                "f1": _delta_value(pipeline_f1, comparison.full_corpus_report.mean_f1),
                "summary_fidelity": _delta_value(
                    pipeline_fidelity, comparison.full_corpus_report.mean_summary_fidelity
                ),
            },
        },
    }


def combined_artifact(comparison: BaselineComparison, configured_model: str) -> dict:
    """The one combined JSON artifact: the metadata block (configured model,
    the pipeline report's model, generation time, the ADR-0017 threat note
    verbatim, the case inventory, the per-variant drop counts and records),
    both variants' per-case results in the pipeline report's shape, and the
    table data — the single place to quote.

    No model-consistency gate: the operator controls model matching manually,
    so both models travel side by side and the reader decides."""
    drops = {
        PARAMETRIC_VARIANT: comparison.parametric_drops,
        FULL_CORPUS_VARIANT: comparison.full_corpus_drops,
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "configured_model": configured_model,
        "pipeline_model": comparison.pipeline_model,
        "pipeline_report": str(comparison.report_path),
        "threat_note": THREAT_NOTE,
        "case_inventory": [case.case_id for case in comparison.cases],
        "drops": {name: _drops_counts(records) for name, records in drops.items()},
        "drop_records": {name: _drops_records(records) for name, records in drops.items()},
        "variants": {
            PARAMETRIC_VARIANT: _variant_artifact(comparison.parametric_report),
            FULL_CORPUS_VARIANT: _variant_artifact(comparison.full_corpus_report),
        },
        "comparison": _comparison_artifact(comparison),
    }


def main(argv: list | None = None) -> int:
    """The CLI entry point: refuse with exit 1 when the files or the
    configuration are wrong (the message names the file and the problem),
    abort with exit 1 when the fidelity judge's provider call fails —
    infrastructure failure is never measured quality — pause with exit 130
    on Ctrl-C, otherwise print the comparison table, write the combined
    artifact (default: the UTC-dated path under the logs tree; ``--output``
    moves it), and exit 0.
    ``argv`` defaults to the process arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Score both baseline variants with the shared eval harness and "
            "print the comparison against the pipeline report."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="pipeline report to compare against "
        f"(default: newest {_REPORT_GLOB} in {LOGS_DIR})",
    )
    parser.add_argument(
        "--parametric-dir",
        type=Path,
        default=None,
        help=f"directory holding the parametric case files (default: {BASELINE_DIR})",
    )
    parser.add_argument(
        "--full-corpus-dir",
        type=Path,
        default=None,
        help=f"directory holding the full-corpus case files (default: {FULL_CORPUS_DIR})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="where the combined artifact is written "
        f"(default: {_ARTIFACT_STEM}-<UTC-date>.json under {LOGS_DIR})",
    )
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
        comparison = run_baseline_eval(
            settings,
            parametric_dir=args.parametric_dir,
            full_corpus_dir=args.full_corpus_dir,
            report_path=args.report,
        )
    except (BaselineEvalRefused, ConfigurationError) as error:
        print(f"Baseline eval refused: {error}", file=sys.stderr)
        return 1
    except LlmError as error:
        print(f"Baseline eval aborted: an LLM call failed — {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Baseline eval paused: interrupted.", file=sys.stderr)
        return 130
    print_comparison(comparison)
    output = args.output if args.output is not None else _default_output_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(combined_artifact(comparison, settings.llm_model), indent=2) + "\n"
    )
    print(f"Combined artifact written to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
