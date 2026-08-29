"""Live eval runner — the operator quality command (spec #9, ticket #20).

One command measures Live-mode answer quality: it runs the curated Live
cases (``eval_harness.LIVE_EVAL_SCENARIOS``, hand-authored ground truth)
through the analysis endpoint in Live mode and scores each produced Answer
with the semantic matcher plus citation fidelity — ADR-0001's steps 2 and 3
finally judging real production. It prints per-case precision, recall, and
weighted F1 plus the aggregate mean F1.

The command is operator-run and never part of CI: it requires a real API key
(``OPENROUTER_API_KEY``, exported or in ``backend/.env.local``) and an
ingested Corpus, refusing clearly without either. Demo mode's automated
suite is untouched: ``eval_harness.run_eval`` keeps scoring the verbatim
tripwires, so the tripwire property survives alongside this measurement.

Run from the repository root:

    python -m backend.src.live_eval

or inside the stack:

    docker compose exec backend python -m backend.src.live_eval
"""

import sys

from fastapi.testclient import TestClient

from .availability import (
    INGEST_COMMAND,
    INGEST_COMMAND_IN_STACK,
    NOT_AVAILABLE_WORKFLOW,
    START_POSTGRES_COMMAND,
    UNREACHABLE_STORE_ERRORS,
    vector_store_is_empty,
)
from .config import ConfigurationError, Settings, load_settings
from .eval_harness import LIVE_EVAL_SCENARIOS, EvalReport, evaluate_scenarios
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


def run_live_eval(settings: Settings) -> EvalReport:
    """Run every curated Live case through /api/analyze in Live mode and score it.

    Pins Live mode per request (ADR-0008) regardless of how the app booted —
    the mirror of ``run_eval``'s Demo pinning. The app's request path is also
    pointed at the operator's settings for the duration (restored afterwards),
    so the run measures the configured deployment — key, store, model — not
    the import-time boot state. The requests cross the real endpoint, so
    provider fakes installed at the composition root (the test seam) are
    honoured and every request appends its observability record like any
    other.

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
        return evaluate_scenarios(LIVE_EVAL_SCENARIOS, respond, mode=Mode.live)
    finally:
        main.settings = app_settings


def print_report(report: EvalReport) -> None:
    """Per-case scores, then the aggregate — the operator-facing output."""
    print(f"Live eval: {len(report.scenarios)} case(s) through /api/analyze in Live mode")
    for result in report.scenarios:
        print(
            f"  {result.id}: precision={result.precision:.3f} "
            f"recall={result.recall:.3f} weighted F1={result.f1:.3f}"
        )
    print(f"Aggregate mean weighted F1: {report.mean_f1:.3f}")


def main() -> int:
    """The CLI entry point: refuse with exit 1 when prerequisites are missing,
    otherwise print the report and exit 0."""
    try:
        report = run_live_eval(load_settings())
    except (LiveEvalRefused, ConfigurationError) as error:
        print(f"Live eval refused: {error}", file=sys.stderr)
        return 1
    print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
