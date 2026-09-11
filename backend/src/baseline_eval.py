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
Δ(pipeline − baseline) footers.
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Protocol

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
from .eval_judge import SummaryFidelityJudge, SummaryJudge, judge_naming_its_failures
from .llm import LlmError
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
REPO_ROOT = LOGS_DIR.parent
PARAMETRIC_DIR = LOGS_DIR / "baseline-runs" / "parametric"
FULL_CORPUS_DIR = LOGS_DIR / "baseline-runs" / "full-corpus"
_REPORT_GLOB = "live-eval-report-*.json"

# The artifact's name follows the spec's convention: UTC-dated, under the
# logs tree beside the pipeline reports it compares against.
_ARTIFACT_STEM = "baseline-eval-report"

# The methodology note that must ride the artifact so the numbers are never
# quoted without their caveat: how the scored runs were produced, and where
# the tool-confounded predecessor runs are recorded (ADR-0017).
THREAT_NOTE = (
    "The scored runs are toolless: each answer was produced by an agent with "
    "every tool denied except reading files in its own run directory (the "
    "OpenCode harness, one structured completion per case from the locked "
    "prompt templates), so the Parametric baseline is strictly parametric. "
    "The earlier recorded runs (logs/baseline-runs/, artifact "
    "baseline-eval-report-2026-09-08.json) were produced with tools such as "
    "WebFetch available; that threat to validity is recorded in ADR-0017 and "
    "scored in that artifact."
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


@dataclass(frozen=True)
class VariantSpec:
    """One baseline variant's identity: its canonical name (the directory
    spelling, ADR-0017), the Execution-trace marker naming its provenance,
    and its canonical case-file directory."""

    name: str
    marker: str
    default_dir: Path


def _variant_specs() -> Dict[str, VariantSpec]:
    """The variant registry, in comparison-column order (parametric first).
    Read fresh on every run: the canonical directory constants stay the
    module-level test seam they are."""
    return {
        PARAMETRIC_VARIANT: VariantSpec(PARAMETRIC_VARIANT, PARAMETRIC_MARKER, PARAMETRIC_DIR),
        FULL_CORPUS_VARIANT: VariantSpec(FULL_CORPUS_VARIANT, FULL_CORPUS_MARKER, FULL_CORPUS_DIR),
    }


@dataclass
class VariantOutcome:
    """One variant's scored run: its Execution-trace marker (the provenance
    the artifact carries), the report in the shared shape, and the drop
    records its own files produced."""

    name: str
    marker: str
    report: EvalReport
    drops: DropRecords


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
    """One live case joined across the three answering paths: the pipeline's
    numbers read from the report, plus each variant's result keyed by its
    canonical name."""

    case_id: str
    pipeline: PipelineCaseResult
    baselines: Dict[str, EvalScenarioResult]


@dataclass
class BaselineComparison:
    """The run's outcome: the joined per-case comparison and each variant's
    scored outcome keyed by its canonical name (the combined artifact
    surfaces the drop records in full)."""

    report_path: Path
    pipeline_model: Optional[str]
    cases: List[ComparisonCase]
    outcomes: Dict[str, VariantOutcome]


class _FileChecks:
    """The checks one stored or report file must pass: every refusal names
    the file and the problem, so the file path travels with the checks
    instead of threading through every validator."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def refuse(self, problem: str) -> BaselineEvalRefused:
        """The one refusal shape."""
        return BaselineEvalRefused(f"{self.path}: {problem}")

    def read_json(self) -> object:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise self.refuse(f"cannot be read as JSON ({error})") from error

    def non_empty_str(self, where: str, value: object) -> str:
        if not isinstance(value, str) or not value:
            raise self.refuse(f"{where} must be a non-empty string")
        return value

    def strength(self, where: str, value: object) -> Strength:
        if not isinstance(value, str):
            raise self.refuse(f"{where} must be a bare 'strong'/'moderate'/'weak' string")
        try:
            return Strength(value)
        except ValueError:
            raise self.refuse(
                f"{where} carries invalid strength {value!r} — "
                "expected 'strong', 'moderate', or 'weak'",
            ) from None

    def source_id(self, where: str, value: object, known: set[str]) -> str:
        source = self.non_empty_str(f"{where} source_id", value)
        if source not in known:
            raise self.refuse(
                f"{where} names unknown source id {source!r} "
                f"(known sources: {', '.join(sorted(known))})"
            )
        return source

    def exact_keys(self, where: str, value: object, keys: set[str]) -> dict:
        if not isinstance(value, dict) or set(value) != keys:
            raise self.refuse(f"{where} must carry exactly the keys {sorted(keys)}")
        return value

    def numeric(self, where: str, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise self.refuse(f"{where} must be a number")
        return float(value)


def _validate_case_file(checks: _FileChecks, data: object, known_sources: set[str]) -> CaseFile:
    """Validate one stored case file against the template shape, key-exact
    and type-exact: a mismatch refuses the run naming file and problem —
    no normalization (ADR-0017)."""
    path = checks.path
    if not isinstance(data, dict):
        raise checks.refuse("the file is not a JSON object")
    if set(data) != _STORED_KEYS:
        raise checks.refuse(f"expected exactly the keys {sorted(_STORED_KEYS)}, got {sorted(data)}")

    raw_findings = data["findings"]
    if not isinstance(raw_findings, list):
        raise checks.refuse("'findings' must be a list")
    findings = []
    for index, raw_finding in enumerate(raw_findings):
        where = f"finding {index}"
        finding = checks.exact_keys(where, raw_finding, _FINDING_KEYS)
        statement = checks.non_empty_str(f"{where} statement", finding["statement"])
        strength = checks.strength(where, finding["strength"])
        raw_citations = finding["citations"]
        if not isinstance(raw_citations, list) or not raw_citations:
            raise checks.refuse(f"{where} must carry a non-empty citations list")
        citations = []
        for c_index, raw_citation in enumerate(raw_citations):
            c_where = f"{where} citation {c_index}"
            citation = checks.exact_keys(c_where, raw_citation, _CITATION_KEYS)
            citations.append(StoredCitation(
                source_id=checks.source_id(c_where, citation["source_id"], known_sources),
                provision=checks.non_empty_str(f"{c_where} provision", citation["provision"]),
            ))
        findings.append(StoredFinding(statement=statement, strength=strength, citations=citations))

    raw_summaries = data["summaries"]
    if not isinstance(raw_summaries, list):
        raise checks.refuse("'summaries' must be a list")
    summaries = []
    for index, raw_summary in enumerate(raw_summaries):
        where = f"summary {index}"
        summary = checks.exact_keys(where, raw_summary, _SUMMARY_KEYS)
        summaries.append(StoredSummary(
            source_id=checks.source_id(where, summary["source_id"], known_sources),
            provision=checks.non_empty_str(f"{where} provision", summary["provision"]),
            relevance=checks.non_empty_str(f"{where} relevance", summary["relevance"]),
            strength=checks.strength(where, summary["strength"]),
        ))

    raw_actions = data["actions"]
    if not isinstance(raw_actions, list) or not all(isinstance(action, str) for action in raw_actions):
        raise checks.refuse("'actions' must be a list of strings")

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


def _parsed_target(
    checks: _FileChecks, source_id: str, label: str, drops: DropRecords
) -> Optional[ProvisionTarget]:
    """The label's structural target, or None once the drop is recorded —
    the one shape both the citation and the summary loops share (ADR-0017):
    a label that does not parse is dropped, recorded, and counted."""
    try:
        return parse_provision(source_id, label)
    except ValueError as error:
        drops.unparseable_labels.append(
            DroppedLabel(file=checks.path, label=label, reason=str(error))
        )
        return None


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
    checks = _FileChecks(path)
    case = _validate_case_file(checks, checks.read_json(), sources)

    findings: List[Finding] = []
    for stored in case.findings:
        citations: List[Citation] = []
        for stored_citation in stored.citations:
            target = _parsed_target(checks, stored_citation.source_id, stored_citation.provision, drops)
            if target is None:
                continue
            citations.append(_citation_for(target, stored_citation.provision))
        findings.append(Finding(statement=stored.statement, strength=stored.strength, citations=citations))

    summaries: Dict[ProvisionTarget, GroundedSummary] = {}
    for stored_summary in case.summaries:
        target = _parsed_target(checks, stored_summary.source_id, stored_summary.provision, drops)
        if target is None:
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
    """The newest pipeline report by file time — not by name, which a
    lexicographic max would misread the day a ``glm53_v10`` report lands
    below ``glm53_v8``."""
    reports = list(logs_dir.glob(_REPORT_GLOB))
    if not reports:
        raise BaselineEvalRefused(
            f"No pipeline report found in {logs_dir} (no {_REPORT_GLOB} files). "
            "Run the live eval first or pass --report PATH."
        )
    return max(reports, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _parse_report(path: Path) -> tuple[Optional[str], Dict[str, PipelineCaseResult]]:
    """Read the pipeline report artifact: the report's model plus its
    per-case numbers keyed by case id. Anything malformed refuses naming
    file and problem — a comparison over a half-readable report would be
    quote-worthy nonsense."""
    checks = _FileChecks(path)
    data = checks.read_json()
    if not isinstance(data, dict) or not isinstance(data.get("scenarios"), list):
        raise checks.refuse("the report must be a JSON object with a 'scenarios' list")
    model = data.get("llm_model")
    if model is not None and not isinstance(model, str):
        raise checks.refuse("'llm_model' must be a string")
    results: Dict[str, PipelineCaseResult] = {}
    for entry in data["scenarios"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise checks.refuse("every scenario entry needs a string 'id'")
        if entry["id"] in results:
            raise checks.refuse(f"two scenario entries carry id {entry['id']!r}")
        results[entry["id"]] = PipelineCaseResult(
            id=entry["id"],
            precision=checks.numeric(f"case {entry['id']!r} precision", entry.get("precision")),
            recall=checks.numeric(f"case {entry['id']!r} recall", entry.get("recall")),
            f1=checks.numeric(f"case {entry['id']!r} f1", entry.get("f1")),
            summary_fidelity=(
                None
                if entry.get("summary_fidelity") is None
                else checks.numeric(f"case {entry['id']!r} summary_fidelity", entry["summary_fidelity"])
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
            ".env) and re-run."
        )
    return judge_naming_its_failures(SummaryFidelityJudge(chat_client(settings)))


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
    variant_dirs: Optional[Mapping[str, Path]] = None,
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

    ``variant_dirs`` overrides the canonical per-variant directories, keyed
    by the variant names (tests; the CLI passes its flags); ``report_path``
    overrides the newest-report default, and the judge seam is injectable
    for the same reason.
    """
    specs = _variant_specs()
    overrides = variant_dirs or {}
    files = {
        name: _discover_case_files(overrides.get(name) or spec.default_dir)
        for name, spec in specs.items()
    }
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
    # Both variants adapt before either is scored: a malformed file anywhere
    # refuses without a single judge call having been spent.
    adapted = {
        name: _adapt_variant(files[name], spec.marker, known_sources)
        for name, spec in specs.items()
    }
    outcomes = {
        name: VariantOutcome(
            name=name,
            marker=specs[name].marker,
            report=_evaluate_variant(responses, fidelity_judge),
            drops=drops,
        )
        for name, (responses, drops) in adapted.items()
    }

    cases = [
        ComparisonCase(
            case_id=case.id,
            pipeline=pipeline_by_id[case.id],
            baselines={
                name: outcome.report.scenarios[index]
                for name, outcome in outcomes.items()
            },
        )
        for index, case in enumerate(LIVE_EVAL_SCENARIOS)
    ]
    return BaselineComparison(
        report_path=resolved_report,
        pipeline_model=pipeline_model,
        cases=cases,
        outcomes=outcomes,
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


class _CaseMetrics(Protocol):
    """The four component numbers both answering-path results carry — the
    pipeline's read from the report artifact, the baselines' measured by
    the shared loop (ADR-0010)."""

    precision: float
    recall: float
    f1: float
    summary_fidelity: Optional[float]


def _cell_text(result: _CaseMetrics) -> str:
    """One answering path's four component numbers as one printed cell."""
    return (
        f"{result.precision:.3f} / {result.recall:.3f} / "
        f"{result.f1:.3f} / {format_score(result.summary_fidelity)}"
    )


def _caption(name: str) -> str:
    """A column's caption: the variant's name over the four components."""
    return f"{name} (precision / recall / F1 / fidelity)"


def print_comparison(comparison: BaselineComparison) -> None:
    """The operator-facing output: one row per case — precision, recall,
    coverage F1, and summary fidelity for the pipeline and each baseline
    variant in registry order — then the per-variant means and the
    Δ(pipeline − baseline) footers. The pipeline report that produced the
    left column is named alongside its model."""
    cases = comparison.cases
    names = list(comparison.outcomes)
    print(f"Baseline comparison: pipeline vs {' vs '.join(names)} ({len(cases)} case(s))")
    model = comparison.pipeline_model or "unknown model"
    print(f"Pipeline report: {comparison.report_path} (model: {model})")
    print()
    last = len(names) - 1
    header = f"  {'case':<40}{'pipeline (precision / recall / F1 / fidelity)':<48}"
    for index, name in enumerate(names):
        header += f"{_caption(name):<44}" if index < last else _caption(name)
    print(header)
    for case in cases:
        row = f"  {case.case_id:<40}{_cell_text(case.pipeline):<48}"
        for index, name in enumerate(names):
            cell = _cell_text(case.baselines[name])
            row += f"{cell:<44}" if index < last else cell
        print(row)
    print()
    pipeline_f1, pipeline_fidelity = _pipeline_means(cases)
    print(
        f"  Mean coverage F1: pipeline {pipeline_f1:.3f} | "
        + " | ".join(f"{name} {comparison.outcomes[name].report.mean_f1:.3f}" for name in names)
        + " | "
        + " | ".join(
            f"Δ(pipeline − {name}) {_delta(pipeline_f1, comparison.outcomes[name].report.mean_f1)}"
            for name in names
        )
    )
    print(
        f"  Mean summary fidelity: pipeline {format_score(pipeline_fidelity)} | "
        + " | ".join(
            f"{name} {format_score(comparison.outcomes[name].report.mean_summary_fidelity)}"
            for name in names
        )
        + " | "
        + " | ".join(
            f"Δ(pipeline − {name}) "
            f"{_delta(pipeline_fidelity, comparison.outcomes[name].report.mean_summary_fidelity)}"
            for name in names
        )
    )


def _default_output_path() -> Path:
    """The UTC-dated artifact path the spec's convention names, under the
    logs tree beside the pipeline reports it compares against."""
    date = datetime.now(timezone.utc).date().isoformat()
    return LOGS_DIR / f"{_ARTIFACT_STEM}-{date}.json"


def _cell(result: _CaseMetrics) -> dict:
    """One answering path's four component numbers, as the table shows them."""
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


def _artifact_path(path: Path) -> str:
    """The path as the artifact records it: relative to the repo root when
    the file lives inside the tree — the artifact travels — else absolute."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _drops_records(drops: DropRecords) -> dict:
    """The two drop accounts as plain data — file, label, and (for labels)
    the parse failure — so the deterministic resolutions stay visible."""
    return {
        "duplicate_summaries": [
            {"file": _artifact_path(duplicate.file), "label": duplicate.label}
            for duplicate in drops.duplicate_summaries
        ],
        "unparseable_labels": [
            {"file": _artifact_path(dropped.file), "label": dropped.label, "reason": dropped.reason}
            for dropped in drops.unparseable_labels
        ],
    }


def _variant_artifact(report: EvalReport, marker: str) -> dict:
    """One variant's report in the pipeline report's shape: per-case
    expected/produced dumps and per-pair fidelity verdicts (issue #83's
    persistence rule, inherited unchanged), with the mode the runs pinned
    and the variant's Execution-trace marker naming its provenance — so
    artifacts never mix variants (spec #90, story 19)."""
    return {
        "mode": Mode.live.value,
        "execution_trace_marker": marker,
        **asdict(report),
    }


def _comparison_artifact(comparison: BaselineComparison) -> dict:
    """The table data: per-case rows for all three answering paths, the
    per-variant means, and the Δ(pipeline − baseline) footers as data —
    the single place to quote."""
    pipeline_f1, pipeline_fidelity = _pipeline_means(comparison.cases)
    means = {
        "pipeline": {"f1": pipeline_f1, "summary_fidelity": pipeline_fidelity},
        **{
            name: {
                "f1": outcome.report.mean_f1,
                "summary_fidelity": outcome.report.mean_summary_fidelity,
            }
            for name, outcome in comparison.outcomes.items()
        },
    }
    return {
        "cases": [
            {
                "case_id": case.case_id,
                "pipeline": _cell(case.pipeline),
                **{name: _cell(result) for name, result in case.baselines.items()},
            }
            for case in comparison.cases
        ],
        "means": means,
        "delta_vs_pipeline": {
            name: {
                "f1": _delta_value(pipeline_f1, comparison.outcomes[name].report.mean_f1),
                "summary_fidelity": _delta_value(
                    pipeline_fidelity,
                    comparison.outcomes[name].report.mean_summary_fidelity,
                ),
            }
            for name in comparison.outcomes
        },
    }


def combined_artifact(comparison: BaselineComparison, configured_model: str) -> dict:
    """The one combined JSON artifact: the metadata block (configured model,
    the pipeline report's model, generation time, the ADR-0017 threat note
    verbatim, the case inventory, the per-variant drop counts and records),
    both variants' per-case results in the pipeline report's shape, and the
    table data — the single place to quote.

    No model-consistency gate: the operator controls model matching manually,
    so both models travel side by side and the reader decides. The two
    fidelity judges are named alongside — the pipeline report's fidelity was
    judged by its own run's model, the baselines' by the configured one —
    so a fidelity delta between differing judges reads as what it is."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "configured_model": configured_model,
        "pipeline_model": comparison.pipeline_model,
        "fidelity_judge_models": {
            "pipeline": comparison.pipeline_model,
            "baselines": configured_model,
        },
        "pipeline_report": _artifact_path(comparison.report_path),
        "threat_note": THREAT_NOTE,
        "case_inventory": [case.case_id for case in comparison.cases],
        "drops": {
            name: _drops_counts(outcome.drops) for name, outcome in comparison.outcomes.items()
        },
        "drop_records": {
            name: _drops_records(outcome.drops) for name, outcome in comparison.outcomes.items()
        },
        "variants": {
            name: _variant_artifact(outcome.report, outcome.marker)
            for name, outcome in comparison.outcomes.items()
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
        help=f"directory holding the parametric case files (default: {PARAMETRIC_DIR})",
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
            variant_dirs={
                name: dir_
                for name, dir_ in (
                    (PARAMETRIC_VARIANT, args.parametric_dir),
                    (FULL_CORPUS_VARIANT, args.full_corpus_dir),
                )
                if dir_ is not None
            },
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
