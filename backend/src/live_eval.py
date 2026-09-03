"""Live eval runner — the operator quality command (spec #9, ticket #20).

One command measures Live-mode answer quality: it runs the curated Live
cases (``eval_harness.LIVE_EVAL_SCENARIOS``, hand-authored ground truth)
through the analysis endpoint in Live mode and scores each produced Answer
with the three never-blended components of ADR-0010: provision coverage
(deterministic set F1 over Citation targets, strength-weighted on the
expected side), summary fidelity (the strict rubric judge of
``eval_judge``, one batched call per case, schema-validated verdicts), and
strength agreement (the operator's expected ratings against the
Summarizer's produced ratings, ADR-0011, over rated shared targets only,
read at small weight). It prints all three per case plus the aggregate
means.

Summary fidelity measures where the ground truth summarizes a cited
provision (the labels #51 authored, one per cited provision); elsewhere it
reports unmeasured, and
the judge is never woken for nothing. The judge is the configured provider
itself (ADR-0009): if any LLM call fails mid-run — the workflow's or the
judge's, an unreachable provider or a reply that fails its schema — the run
aborts with the detail, never scoring silence.

With ``--output PATH`` the command also writes a JSON artifact: the run's
metadata, the per-case component scores, and the produced-versus-expected
dump (statements, Strengths, Citation targets, relevance summaries, the
rated Citation strengths) for the human audit ADR-0010 prescribes before
numbers are quoted.

Every command checkpoints (ADR-0013): a run id — given via ``--run-id`` or
minted on the spot (UTC timestamp plus four random hex characters) — names
a JSON file under ``data/eval-runs/`` that records each completed case the
moment it is scored, so a run that dies mid-flight (a provider outage, an
operator Ctrl-C) resumes from the printed command instead of starting over
and re-paying for the cases already measured. A resume re-runs only the
pending cases: the checkpoint is a per-scenario cache under today's case
list — results substitute only while their case's definition hash and the
configured model still match, and a stale or unreadable checkpoint refuses
the run rather than quietly mixing measurements. On success with
``--output`` the artifact supersedes the checkpoint and the file is
removed; without ``--output`` the checkpoint stays as the run's only
record, and re-invoking the id re-serves the full report with no LLM calls
at all.

The command is operator-run and never part of CI: it requires a real API key
(``OPENROUTER_API_KEY``, exported or in ``backend/.env.local``) and an
ingested Corpus, refusing clearly without either. Demo mode's automated
suite is untouched: ``eval_harness.run_eval`` keeps scoring the Demo
tripwires, so the tripwire property survives alongside this measurement.

Run from the repository root:

    python -m backend.src.live_eval [--run-id ID] [--output PATH]

or inside the stack:

    docker compose exec backend python -m backend.src.live_eval [--run-id ID] [--output PATH]

Resume an interrupted or failed run by re-running it with the same
``--run-id`` — the abort message prints the exact command.
"""

import argparse
import json
import secrets
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi.testclient import TestClient

from .availability import (
    INGEST_COMMAND,
    INGEST_COMMAND_IN_STACK,
    NOT_AVAILABLE_WORKFLOW,
    START_POSTGRES_COMMAND,
    UNREACHABLE_STORE_ERRORS,
    vector_store_is_empty,
)
from .config import ConfigurationError, Settings, chat_client, load_settings
from .eval_harness import (
    LIVE_EVAL_SCENARIOS,
    EvalReport,
    EvalScenario,
    EvalScenarioResult,
    evaluate_scenarios,
    scenario_definition_hash,
)
from .eval_judge import SummaryFidelityJudge, SummaryJudge
from .llm import LlmError
from .models import AnalyzeRequest, AnalyzeResponse, Mode


class LiveEvalRefused(RuntimeError):
    """The quality run cannot start; the message names what to do about it."""


# Where run checkpoints live (ADR-0013): one JSON file per run id, gitignored,
# anchored to the repository root so the command finds it from anywhere.
CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "data" / "eval-runs"


def checkpoint_path(run_id: str, checkpoint_dir: Optional[Path] = None) -> Path:
    """Where a run id's checkpoint file lives (ADR-0013)."""
    return (checkpoint_dir if checkpoint_dir is not None else CHECKPOINT_DIR) / f"{run_id}.json"


def _load_document(path: Path) -> dict:
    """Parse a checkpoint document, converting every malformed payload —
    truncated JSON, a non-object, missing metadata — into the same clear
    refusal (ADR-0013): an unreadable checkpoint is the operator's decision,
    never a traceback and never a silent fresh start."""
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise TypeError("checkpoint is not a JSON object")
        if (
            not isinstance(data.get("llm_model"), str)
            or not isinstance(data.get("scenario_hashes"), dict)
            or not isinstance(data.get("scenarios"), list)
        ):
            raise TypeError("checkpoint is missing its metadata fields")
        return data
    except (json.JSONDecodeError, OSError, TypeError) as error:
        raise LiveEvalRefused(
            f"Checkpoint file {path} is unreadable ({error}). "
            "Delete it or resume under a different --run-id."
        ) from error


def _report_over(cases: List[EvalScenarioResult]) -> EvalReport:
    """The aggregate report over a non-empty set of case results — the one
    means computation both the stitched report and the partial re-print use.
    An unmeasured component averages over the cases where it measured, never
    over all of them."""
    measured_fidelity = [case.summary_fidelity for case in cases if case.summary_fidelity is not None]
    measured_agreement = [case.strength_agreement for case in cases if case.strength_agreement is not None]
    return EvalReport(
        scenarios=list(cases),
        mean_f1=sum(case.f1 for case in cases) / len(cases),
        mean_summary_fidelity=sum(measured_fidelity) / len(measured_fidelity) if measured_fidelity else None,
        mean_strength_agreement=sum(measured_agreement) / len(measured_agreement) if measured_agreement else None,
    )


class CheckpointedRun:
    """One run's resumable progress on disk (ADR-0013).

    The file at ``path`` holds the run's metadata plus every completed case's
    ``EvalScenarioResult`` — the exact shape the final artifact gives a case —
    rewritten atomically (temp file + rename) each time a case lands, so a
    run that dies mid-flight loses no measured work. A resume loads it and
    substitutes results per case: today's case list wins (a case added to the
    list runs fresh, one removed leaves an unused record behind), and a
    completed result is kept only while its case's definition hash and the
    configured model still match — a stale checkpoint aborts instead of
    quietly mixing measurements (ADR-0009).
    """

    def __init__(self, path: Path, llm_model: str, scenarios: List[EvalScenario]) -> None:
        self.path = path
        self.llm_model = llm_model
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._order = [scenario.id for scenario in scenarios]
        self._cases = {scenario.id: scenario for scenario in scenarios}
        self.results: Dict[str, EvalScenarioResult] = {}
        self._hashes: Dict[str, str] = {}

    @classmethod
    def start(cls, path: Path, llm_model: str, scenarios: List[EvalScenario]) -> "CheckpointedRun":
        """Open the run, adopting any checkpoint already at ``path``."""
        run = cls(path, llm_model, scenarios)
        if path.exists():
            run._load()
        return run

    def _load(self) -> None:
        data = _load_document(self.path)
        if data["llm_model"] != self.llm_model:
            raise LiveEvalRefused(
                f"Checkpoint {self.path} was produced by model {data['llm_model']!r} but the "
                f"configured model is {self.llm_model!r}: results are model-sensitive "
                "(ADR-0009), so the numbers must not be mixed. Start a new run."
            )
        stored_start = data.get("started_at")
        if isinstance(stored_start, str):
            # The metadata describes the run that measured the cases: a
            # resume adopts the original start instead of re-stamping.
            self.started_at = stored_start
        for case in data["scenarios"]:
            try:
                case_id: str = case["id"]
            except (KeyError, TypeError) as error:
                raise LiveEvalRefused(
                    f"Checkpoint file {self.path} is unreadable ({error}). "
                    "Delete it or resume under a different --run-id."
                ) from error
            scenario = self._cases.get(case_id)
            if scenario is None:
                # The case left the live list since this record landed: the
                # current list wins, so the record simply goes unused.
                continue
            try:
                result = EvalScenarioResult(**case)
            except TypeError as error:
                raise LiveEvalRefused(
                    f"Checkpoint file {self.path} is unreadable ({error}). "
                    "Delete it or resume under a different --run-id."
                ) from error
            if data["scenario_hashes"].get(case_id) != scenario_definition_hash(scenario):
                raise LiveEvalRefused(
                    f"Live case {case_id!r} changed since run {self.path.stem!r} started "
                    "(question, description, or ground truth edited), so its checkpointed "
                    "result is stale. Start a new run."
                )
            self.results[case_id] = result
            self._hashes[case_id] = data["scenario_hashes"][case_id]

    def on_result(self, result: EvalScenarioResult) -> None:
        """The harness's per-case callback: record, then persist immediately."""
        self.results[result.id] = result
        self._hashes[result.id] = scenario_definition_hash(self._cases[result.id])
        self._save()

    def pending(self) -> List[EvalScenario]:
        """The cases the run still owes work for, in case-list order."""
        return [self._cases[case_id] for case_id in self._order if case_id not in self.results]

    def report(self, fresh: Optional[EvalReport] = None) -> EvalReport:
        """The run's full report: checkpointed results substituted per case
        into today's case-list order, fresh results filling the rest — the
        aggregates always describe exactly the list that ran."""
        merged = dict(self.results)
        if fresh is not None:
            merged.update({result.id: result for result in fresh.scenarios})
        return _report_over([merged[case_id] for case_id in self._order if case_id in merged])

    def artifact(self) -> dict:
        """The checkpoint document: run metadata over the completed cases."""
        return {
            "run_id": self.path.stem,
            "mode": Mode.live.value,
            "started_at": self.started_at,
            "llm_model": self.llm_model,
            "scenario_hashes": dict(self._hashes),
            "scenarios": [
                asdict(self.results[case_id]) for case_id in self._order if case_id in self.results
            ],
        }

    def _save(self) -> None:
        """Rewrite the checkpoint atomically: a half-written file must never
        masquerade as progress, so the payload lands as a temp file renamed
        into place."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(self.path.name + ".tmp")
        temp.write_text(json.dumps(self.artifact(), indent=2) + "\n")
        temp.replace(self.path)


def _judge_naming_its_failures(judge: SummaryJudge) -> SummaryJudge:
    """The judge with its failures annotated: when an LLM call aborts the run,
    the message must say which call it was — the judge's LlmErrors are
    re-raised prefixed with "judge", so they read differently from the
    workflow stages' own failures under the CLI's single abort message."""
    class _BlamedJudge:
        def compare(self, pairs):
            try:
                return judge.compare(pairs)
            except LlmError as error:
                raise LlmError(f"summary-fidelity judge: {error}") from error

    return _BlamedJudge()


def ensure_runnable(settings: Settings) -> None:
    """Refuse clearly when the run's prerequisites are missing.

    Both checks happen before any request is sent — a measurement over an
    un-ingested store would be garbage scored as if it were signal. Only
    genuine store outages get recovery advice here; anything else the probe
    raises surfaces loudly, mirroring the request-path classification.
    """
    if not settings.openrouter_api_key:
        raise LiveEvalRefused(
            "OPENROUTER_API_KEY is not set. Export it (or put it in "
            "backend/.env.local) and re-run."
        )
    try:
        empty = vector_store_is_empty(settings.database_url)
    except UNREACHABLE_STORE_ERRORS as error:
        raise LiveEvalRefused(
            f"Cannot reach the vector store at {settings.database_url} ({error}). "
            f"Start it with: {START_POSTGRES_COMMAND}"
        ) from error
    if empty:
        raise LiveEvalRefused(
            "The vector store holds no Chunks — the Corpus is not ingested yet. "
            f"Ingest once with: {INGEST_COMMAND} "
            f"(inside the stack: {INGEST_COMMAND_IN_STACK}), then re-run."
        )


def run_live_eval(
    settings: Settings,
    judge: Optional[SummaryJudge] = None,
    scenarios: Optional[List[EvalScenario]] = None,
    run_id: Optional[str] = None,
    checkpoint_dir: Optional[Path] = None,
) -> EvalReport:
    """Run every curated Live case through /api/analyze in Live mode and score it.

    Pins Live mode per request (ADR-0008) regardless of how the app booted —
    the mirror of ``run_eval``'s Demo pinning. The app's request path is also
    pointed at the operator's settings for the duration (restored afterwards),
    so the run measures the configured deployment — key, store, model — not
    the import-time boot state. The requests cross the real endpoint, so
    provider fakes installed at the composition root (the test seam) are
    honoured and every request appends its observability record like any
    other.

    The fidelity judge defaults to the configured provider (ADR-0009) through
    the one ``chat_client`` recipe the workflow itself runs on; tests script
    the seam by passing one. The default judge carries its name on its
    failures — when an LLM call aborts the run, the operator must be able to
    tell which call it was: the judge's failures say "judge", the workflow
    stages' speak for themselves. Any LLM failure — unreachable provider,
    schema-invalid reply, workflow or judge — aborts the run:
    infrastructure failure is never measured quality.

    A Not-available response mid-run means the ground shifted under the run
    (the store emptied, an outage began): it is infrastructure failure, never
    measured quality, so the run aborts instead of scoring silent zeros.

    With ``run_id`` the run checkpoints (ADR-0013): each completed case is
    persisted via ``CheckpointedRun`` before the next starts, cases already
    recorded under the id are skipped, and the returned report stitches
    checkpointed and fresh results in case-list order. ``checkpoint_dir``
    overrides where the file lives (tests; the CLI uses ``CHECKPOINT_DIR``).
    Without a ``run_id`` the behaviour is exactly the uncheckpointed run.
    """
    selected = scenarios if scenarios is not None else LIVE_EVAL_SCENARIOS
    checkpoint = (
        CheckpointedRun.start(checkpoint_path(run_id, checkpoint_dir), settings.llm_model, selected)
        if run_id is not None
        else None
    )
    pending = checkpoint.pending() if checkpoint is not None else list(selected)
    if not pending:
        # Fully resumed: every case the list asks for is already measured and
        # recorded, so the report is served from the checkpoint alone — no
        # precondition probe, no provider call, nothing left to run.
        assert checkpoint is not None
        return checkpoint.report()

    ensure_runnable(settings)

    from . import main

    client = TestClient(main.app)
    app_settings = main.settings
    # Point the app's request path at the operator's settings for the
    # duration: the app's import-time snapshot can be stale (tests re-patch
    # the environment), and the run must measure the configured deployment —
    # key, store, model — not whatever the app happened to boot with.
    main.settings = settings

    def respond(request: AnalyzeRequest) -> AnalyzeResponse:
        response = client.post("/api/analyze", json=request.model_dump())
        response.raise_for_status()
        analyzed = AnalyzeResponse.model_validate(response.json())
        if analyzed.trace.workflow == NOT_AVAILABLE_WORKFLOW:
            raise LiveEvalRefused(
                f"case {request.scenario.id!r} got a Not-available response "
                f"({analyzed.trace.summary}) — the run's prerequisites broke "
                "mid-flight; fix them and re-run."
            )
        return analyzed

    try:
        report = evaluate_scenarios(
            pending,
            respond,
            mode=Mode.live,
            judge=(
                judge
                if judge is not None
                else _judge_naming_its_failures(SummaryFidelityJudge(chat_client(settings)))
            ),
            on_result=checkpoint.on_result if checkpoint is not None else None,
        )
    finally:
        main.settings = app_settings
    return checkpoint.report(report) if checkpoint is not None else report


def _fmt(score: Optional[float]) -> str:
    """Three decimals for a measured score, ``n/a`` for one that is not —
    an unmeasured component must never dress up as a zero."""
    return f"{score:.3f}" if score is not None else "n/a"


def print_report(report: EvalReport, llm_model: str) -> None:
    """Per-case component scores, then the aggregates — the operator-facing
    output. Three components, three means, never blended (ADR-0010). The
    model that produced the numbers is named: results are model-sensitive
    by design (ADR-0009)."""
    print(f"Live eval: {len(report.scenarios)} case(s) through /api/analyze in Live mode")
    print(f"Model: {llm_model}")
    for result in report.scenarios:
        print(
            f"  {result.id}: coverage precision={result.precision:.3f} "
            f"recall={result.recall:.3f} F1={result.f1:.3f} | "
            f"summary fidelity={_fmt(result.summary_fidelity)} | "
            f"strength agreement={_fmt(result.strength_agreement)}"
        )
    print(f"Aggregate mean coverage F1: {report.mean_f1:.3f}")
    print(f"Aggregate mean summary fidelity: {_fmt(report.mean_summary_fidelity)}")
    print(f"Aggregate mean strength agreement: {_fmt(report.mean_strength_agreement)}")


def report_artifact(report: EvalReport, llm_model: str) -> dict:
    """The JSON artifact the CLI writes: run metadata over the full report —
    the model that produced the numbers (ADR-0009), per-case component
    scores, and the produced-versus-expected dump."""
    return {
        "mode": Mode.live.value,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "llm_model": llm_model,
        **asdict(report),
    }


def write_report_artifact(report: EvalReport, path: Path, llm_model: str) -> None:
    """Write the artifact, creating missing parent directories on the way."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report_artifact(report, llm_model), indent=2) + "\n")


def _mint_run_id() -> str:
    """A fresh run id (ADR-0013): a UTC timestamp plus four random hex
    characters — sortable, and unique even when two runs start in the same
    second."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


def _resume_hint(run_id: str) -> str:
    """The one command that picks an interrupted run back up."""
    return f"Resume with: python -m backend.src.live_eval --run-id {run_id}"


def _print_partial_progress(path: Path) -> None:
    """Re-print the completed cases from the checkpoint, so an interrupted or
    aborted run leaves its measured numbers visible, not only on disk. The
    file was written by this very run, so an unreadable one is silently
    skipped — the abort message and the resume hint still stand."""
    try:
        data = _load_document(path)
        cases = [EvalScenarioResult(**case) for case in data["scenarios"]]
    except (LiveEvalRefused, TypeError):
        return
    if cases:
        print_report(_report_over(cases), llm_model=data["llm_model"])


def main(argv: list | None = None) -> int:
    """The CLI entry point: refuse with exit 1 when prerequisites are missing,
    abort with exit 1 when an LLM call fails mid-run (the workflow's or the
    judge's — a malformed reply or an unreachable provider is infrastructure
    failure, never measured quality), pause with exit 130 on Ctrl-C, otherwise
    print the report, write the artifact when --output is given, and exit 0.

    Every command checkpoints (ADR-0013): a run id — given via --run-id or
    minted on the spot — names a checkpoint file under data/eval-runs/ that
    records each completed case, so a failed or interrupted run resumes with
    the printed command instead of starting over. On success with --output
    the artifact supersedes the checkpoint and the file is removed; without
    --output the checkpoint stays as the run's only record.
    ``argv`` defaults to the process arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Live eval and print per-case component scores.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="write the per-case report artifact (JSON) to this path",
    )
    parser.add_argument(
        "--run-id",
        "-r",
        default=None,
        help="checkpoint (and resume) under this run id; minted when omitted (ADR-0013)",
    )
    args = parser.parse_args(argv)

    run_id = args.run_id or _mint_run_id()
    checkpoint_file = checkpoint_path(run_id)
    print(f"Run id: {run_id} (checkpoint: {checkpoint_file})")

    try:
        settings = load_settings()
        report = run_live_eval(settings, run_id=run_id)
    except (LiveEvalRefused, ConfigurationError) as error:
        print(f"Live eval refused: {error}", file=sys.stderr)
        return 1
    except LlmError as error:
        print(f"Live eval aborted: an LLM call failed — {error}", file=sys.stderr)
        _print_partial_progress(checkpoint_file)
        print(_resume_hint(run_id), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Live eval paused: interrupted — completed cases are checkpointed.", file=sys.stderr)
        _print_partial_progress(checkpoint_file)
        print(_resume_hint(run_id), file=sys.stderr)
        return 130
    print_report(report, llm_model=settings.llm_model)
    if args.output is not None:
        write_report_artifact(report, args.output, llm_model=settings.llm_model)
        checkpoint_file.unlink(missing_ok=True)
        print(f"Report artifact written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
