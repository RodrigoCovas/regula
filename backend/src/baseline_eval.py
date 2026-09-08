"""Baseline-eval operator command (spec #90, ticket #91): score the
Parametric baseline with the shared eval harness and print the comparison
against the newest pipeline report.

The Baseline comparison measures what the workflow's machinery adds by
scoring answers produced without it. The Parametric baseline's answers are
stored — one structured completion per live case, generated in the OpenCode
agent harness — under ``logs/baseline-runs/parametric/`` as
``<scenario-id>.json`` in the stored template shape ``{findings, summaries,
actions}`` (ADR-0017). This command loads each file for the ten live cases
(``eval_harness.LIVE_EVAL_SCENARIOS``), validates it strictly — wrong keys,
wrong types, invalid strength, unknown source id, missing or unreadable
file, or a filename outside the ten live Scenario ids refuse the run naming
file and problem — and maps every validated file into the response shape the
shared evaluation harness consumes: Findings with Citations derived from the
parsed provision labels (sub-references fold onto the provision they
refine), Provision relevance and Citation strength attached through the
existing answer-citations builder, and the standing seek-counsel Action
appended last. The variant's Execution-trace marker names its provenance
(``baseline-parametric: agent-produced (OpenCode harness)``).

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
flatters precision. The drop counts ride the result; the artifact that
surfaces them in full lands with the combined-artifact ticket.

The two exceptions aside, nothing is normalized inside the pipeline:
malformed files are fixed by hand before the comparison runs.

Run from the repository root:

    python -m backend.src.baseline_eval [--report PATH] [--parametric-dir PATH]

and inside the stack:

    docker compose exec backend python -m backend.src.baseline_eval

Refusals exit 1; a judge or provider failure mid-run aborts with exit 1;
Ctrl-C pauses with exit 130; success prints the comparison table — per-case
precision, recall, coverage F1, and summary fidelity for pipeline and
parametric — with the per-variant means and the Δ(pipeline − parametric)
footer.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

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
    ProvisionKind,
    ProvisionTarget,
    Strength,
    Trace,
    answer_citations,
)


class BaselineEvalRefused(RuntimeError):
    """The baseline comparison cannot start; the message names the file and the problem."""


# Canonical file locations (ADR-0017): module constants so tests can point
# them at fixture directories.
LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"
BASELINE_DIR = LOGS_DIR / "baseline-runs" / "parametric"
_REPORT_GLOB = "live-eval-report-*.json"

# The variant's Execution-trace marker (spec #90): provenance travels with
# every answer so artifacts never mix variants.
PARAMETRIC_MARKER = "baseline-parametric: agent-produced (OpenCode harness)"

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

    path: Path
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
    """One live case joined across the two answering paths."""

    case_id: str
    pipeline: PipelineCaseResult
    parametric: EvalScenarioResult


@dataclass
class BaselineComparison:
    """The run's outcome: the joined per-case comparison, the parametric
    report in the shared shape, and the drop records (full surfacing lands
    with the combined-artifact ticket)."""

    report_path: Path
    pipeline_model: Optional[str]
    cases: List[ComparisonCase]
    parametric_report: EvalReport
    drops: DropRecords


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
        path=path,
        findings=findings,
        summaries=summaries,
        actions=list(raw_actions),
    )


def _citation_for(target: ProvisionTarget, label: str) -> Citation:
    """One derived Citation targeting the parsed provision: the exactly-one
    number field comes from the kind (``PROVISION_NUMBER_FIELDS``)."""
    if target.kind is ProvisionKind.article:
        return Citation(source_id=target.source_id, provision=label, article_number=target.number)
    if target.kind is ProvisionKind.recital:
        return Citation(source_id=target.source_id, provision=label, recital_number=target.number)
    return Citation(source_id=target.source_id, provision=label, annex_number=target.number)


def load_parametric_case(path: Path, drops: DropRecords) -> AnalyzeResponse:
    """Load, validate, and adapt one stored parametric case file into the
    response shape the shared harness consumes (ADR-0017).

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
    known_sources = set(load_documents())
    case = _validate_case_file(path, _read_json(path), known_sources)

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
        trace=Trace(workflow=PARAMETRIC_MARKER, summary=_TRACE_SUMMARY),
    )


def _discover_case_files(directory: Path) -> Dict[str, Path]:
    """The ten case files keyed by Scenario id: every ``*.json`` file must be
    named for a live Scenario id (a typo'd or stray file refuses the run) and
    all ten must be present."""
    if not directory.is_dir():
        raise BaselineEvalRefused(f"Baseline directory {directory} does not exist")
    scenario_ids = {case.scenario_id for case in LIVE_EVAL_SCENARIOS}
    files: Dict[str, Path] = {}
    for path in sorted(directory.glob("*.json")):
        if path.stem not in scenario_ids:
            raise BaselineEvalRefused(
                f"{path} is not one of the ten live Scenario ids — rename or remove it "
                f"(expected one of: {', '.join(sorted(scenario_ids))})"
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


def run_baseline_eval(
    settings: Settings,
    judge: Optional[SummaryJudge] = None,
    parametric_dir: Optional[Path] = None,
    report_path: Optional[Path] = None,
) -> BaselineComparison:
    """Score the Parametric baseline over the ten live cases with the shared
    harness and join it against the pipeline report.

    Everything validates before anything is measured: the case files are
    discovered, validated, and adapted first, and the report is parsed and
    checked for the live case list — a refusal never spends a judge call.
    The fidelity judge defaults to the configured provider (ADR-0009) built
    exactly like the live-eval runner's; tests script the seam by passing
    one. Mode is pinned to live per case (artifact-shape compatibility), the
    pipeline numbers are read from the report, never re-run, and the join
    follows the live case list: a report lacking one of the ten cases
    refuses rather than comparing a subset.

    ``parametric_dir`` and ``report_path`` override the canonical file
    locations (tests; the CLI passes its flags); the judge seam is
    injectable for the same reason.
    """
    directory = parametric_dir if parametric_dir is not None else BASELINE_DIR
    files = _discover_case_files(directory)
    resolved_report = report_path if report_path is not None else _newest_report(LOGS_DIR)
    pipeline_model, pipeline_by_id = _parse_report(resolved_report)
    missing = [case.id for case in LIVE_EVAL_SCENARIOS if case.id not in pipeline_by_id]
    if missing:
        raise BaselineEvalRefused(
            f"Pipeline report {resolved_report} lacks case(s) {', '.join(missing)} — "
            "the comparison joins on the live case list."
        )

    drops = DropRecords()
    responses: Dict[str, AnalyzeResponse] = {
        scenario_id: load_parametric_case(path, drops) for scenario_id, path in files.items()
    }

    def respond(request: AnalyzeRequest) -> AnalyzeResponse:
        return responses[request.scenario.id or ""]

    fidelity_judge = judge if judge is not None else _default_judge(settings)
    report = evaluate_scenarios(
        LIVE_EVAL_SCENARIOS,
        respond,
        mode=Mode.live,
        judge=fidelity_judge,
    )

    cases = [
        ComparisonCase(case_id=case.id, pipeline=pipeline_by_id[case.id], parametric=result)
        for case, result in zip(LIVE_EVAL_SCENARIOS, report.scenarios)
    ]
    return BaselineComparison(
        report_path=resolved_report,
        pipeline_model=pipeline_model,
        cases=cases,
        parametric_report=report,
        drops=drops,
    )


def _mean_of_measured(values) -> Optional[float]:
    """The mean over the measured values only — an unmeasured component is
    never dressed up as a zero."""
    measured = [value for value in values if value is not None]
    return sum(measured) / len(measured) if measured else None


def _delta(pipeline: Optional[float], parametric: Optional[float]) -> str:
    if pipeline is None or parametric is None:
        return "n/a"
    return f"{pipeline - parametric:+.3f}"


def print_comparison(comparison: BaselineComparison) -> None:
    """The operator-facing output: one row per case — precision, recall,
    coverage F1, and summary fidelity for pipeline and parametric — then the
    per-variant means and the Δ(pipeline − parametric) footer. The pipeline
    report that produced the left column is named alongside its model."""
    cases = comparison.cases
    print(f"Baseline comparison: pipeline vs parametric ({len(cases)} case(s))")
    model = comparison.pipeline_model or "unknown model"
    print(f"Pipeline report: {comparison.report_path} (model: {model})")
    print()
    pipeline_caption = "pipeline (precision / recall / F1 / fidelity)"
    print(
        f"  {'case':<40}{pipeline_caption:<48}"
        "parametric (precision / recall / F1 / fidelity)"
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
        print(f"  {case.case_id:<40}{pipeline_cell:<48}{parametric_cell}")
    print()
    pipeline_f1 = sum(case.pipeline.f1 for case in cases) / len(cases)
    parametric_f1 = comparison.parametric_report.mean_f1
    pipeline_fidelity = _mean_of_measured(case.pipeline.summary_fidelity for case in cases)
    parametric_fidelity = comparison.parametric_report.mean_summary_fidelity
    print(
        f"  Mean coverage F1: pipeline {pipeline_f1:.3f} | parametric {parametric_f1:.3f} | "
        f"Δ(pipeline − parametric) {_delta(pipeline_f1, parametric_f1)}"
    )
    print(
        f"  Mean summary fidelity: pipeline {format_score(pipeline_fidelity)} | "
        f"parametric {format_score(parametric_fidelity)} | "
        f"Δ(pipeline − parametric) {_delta(pipeline_fidelity, parametric_fidelity)}"
    )


def main(argv: list | None = None) -> int:
    """The CLI entry point: refuse with exit 1 when the files or the
    configuration are wrong (the message names the file and the problem),
    abort with exit 1 when the fidelity judge's provider call fails —
    infrastructure failure is never measured quality — pause with exit 130
    on Ctrl-C, otherwise print the comparison table and exit 0.
    ``argv`` defaults to the process arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Score the parametric baseline with the shared eval harness and "
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
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
        comparison = run_baseline_eval(
            settings,
            parametric_dir=args.parametric_dir,
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
