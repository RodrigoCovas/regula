"""Live eval runner — the operator quality command (spec #9, ticket #20).

One command measures Live-mode answer quality: it runs the curated Live
cases (``eval_harness.LIVE_EVAL_SCENARIOS``, hand-authored ground truth)
through the analysis endpoint in Live mode and scores each produced Answer
with the three never-blended components of ADR-0010: provision coverage
(deterministic set F1 over Citation targets, strength-weighted on the
expected side), summary fidelity (the strict rubric judge of
``eval_judge``, one batched call per case, schema-validated verdicts), and
strength agreement (max-rule Citation strength on both sides, read at small
weight). It prints all three per case plus the aggregate means.

Summary fidelity measures only where the ground truth summarizes a cited
provision (#51 authors those labels); elsewhere it reports unmeasured, and
the judge is never woken for nothing. The judge is the configured provider
itself (ADR-0009): if it cannot be reached — or a reply fails the verdict
schema — the run aborts with the detail, never scoring silence.

With ``--output PATH`` the command also writes a JSON artifact: the run's
metadata, the per-case component scores, and the produced-versus-expected
dump (statements, Strengths, Citation targets, relevance summaries) for the
human audit ADR-0010 prescribes before numbers are quoted.

The command is operator-run and never part of CI: it requires a real API key
(``OPENROUTER_API_KEY``, exported or in ``backend/.env.local``) and an
ingested Corpus, refusing clearly without either. Demo mode's automated
suite is untouched: ``eval_harness.run_eval`` keeps scoring the Demo
tripwires, so the tripwire property survives alongside this measurement.

Run from the repository root:

    python -m backend.src.live_eval [--output PATH]

or inside the stack:

    docker compose exec backend python -m backend.src.live_eval [--output PATH]
"""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

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
from .eval_harness import LIVE_EVAL_SCENARIOS, EvalReport, EvalScenario, evaluate_scenarios
from .eval_judge import SummaryFidelityJudge, SummaryJudge
from .llm import LlmError
from .models import AnalyzeRequest, AnalyzeResponse, Mode


class LiveEvalRefused(RuntimeError):
    """The quality run cannot start; the message names what to do about it."""


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
    the seam by passing one. A judge failure — an unreachable provider, a
    verdict that fails its schema — aborts the run: infrastructure failure is
    never measured quality.

    A Not-available response mid-run means the ground shifted under the run
    (the store emptied, an outage began): it is infrastructure failure, never
    measured quality, so the run aborts instead of scoring silent zeros.
    """
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
        return evaluate_scenarios(
            scenarios if scenarios is not None else LIVE_EVAL_SCENARIOS,
            respond,
            mode=Mode.live,
            judge=judge if judge is not None else SummaryFidelityJudge(chat_client(settings)),
        )
    finally:
        main.settings = app_settings


def _fmt(score: Optional[float]) -> str:
    """Three decimals for a measured score, ``n/a`` for one that is not —
    an unmeasured component must never dress up as a zero."""
    return f"{score:.3f}" if score is not None else "n/a"


def print_report(report: EvalReport) -> None:
    """Per-case component scores, then the aggregates — the operator-facing
    output. Three components, three means, never blended (ADR-0010)."""
    print(f"Live eval: {len(report.scenarios)} case(s) through /api/analyze in Live mode")
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


def report_artifact(report: EvalReport) -> dict:
    """The JSON artifact the CLI writes: run metadata over the full report —
    per-case component scores plus the produced-versus-expected dump."""
    return {
        "mode": Mode.live.value,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **asdict(report),
    }


def write_report_artifact(report: EvalReport, path: Path) -> None:
    """Write the artifact, creating missing parent directories on the way."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report_artifact(report), indent=2) + "\n")


def main(argv: list | None = None) -> int:
    """The CLI entry point: refuse with exit 1 when prerequisites are missing
    or the judge fails (an unreachable provider, an invalid verdict),
    otherwise print the report, write the artifact when --output is given,
    and exit 0. ``argv`` defaults to the process arguments."""
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
    args = parser.parse_args(argv)

    try:
        report = run_live_eval(load_settings())
    except (LiveEvalRefused, ConfigurationError) as error:
        print(f"Live eval refused: {error}", file=sys.stderr)
        return 1
    except LlmError as error:
        print(f"Live eval aborted: the summary-fidelity judge failed — {error}", file=sys.stderr)
        return 1
    print_report(report)
    if args.output is not None:
        write_report_artifact(report, args.output)
        print(f"Report artifact written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
