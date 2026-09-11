"""Availability: whether Live mode can serve a request right now.

The conditions hang off the Live dispatch point, each answering with a
Not-available response instead of a server error or demo content: no provider
key is configured (ADR-0008 — a Readiness gap, since the backend boots without
a Live-capable configuration), the Corpus is un-ingested (store-emptiness only
this iteration; staleness detection after re-chunking or embedding-model
changes is deferred), the vector store cannot be reached at all, and the LLM
provider cannot be reached at all. Demo mode never reaches for the store or
the LLM — availability is a Live-mode concern alone.

This module also hosts the sibling shape every Not-available response shares,
including the standing English-only Known limitation.
"""

import logging
from contextlib import closing

import psycopg2

from .config import Settings
from .db import PgVectorStore, connect
from .models import AnalyzeResponse, Answer, Trace

logger = logging.getLogger(__name__)

# The one documented ingestion command (backend/src/ingest.py docstring, README).
INGEST_COMMAND = "python -m backend.src.ingest"
INGEST_COMMAND_IN_STACK = "docker compose exec backend python -m backend.src.ingest"

# The one documented way to bring the stack's database up (README, docker-compose.yml).
START_POSTGRES_COMMAND = "docker compose up -d postgres"

# The standing escape hatch shared by every Not-available response (ADR-0008:
# switching to Demo is a per-run choice, no restart).
SWITCH_TO_DEMO_ACTION = (
    "Re-run with mode 'demo' to analyze the canonical bank cloud outage scenario "
    "via scenario.id 'bank-cloud-outage'."
)

ENGLISH_ONLY_LIMITATION = (
    "Known limitation: the corpus is English-only; questions in other languages are answered in English."
)

# A Known limitation, never an Action (ADR-0004): the boundary moved out of
# the demo's action sheet so a limitation can never read like a directive.
PROTOTYPE_LIMITATION = (
    "Known limitation: Regula is a research prototype, not legal advice; answers "
    "require verification by a qualified professional."
)


# Failure modes that mean "cannot reach the store": connection-level outages
# only. Anything else — a missing table, a bad query, a canceled statement —
# must surface as a server error, never as unreachable-store recovery advice.
UNREACHABLE_STORE_ERRORS = (ConnectionError, psycopg2.OperationalError)

# The workflow marker every Not-available response carries in its Execution trace.
NOT_AVAILABLE_WORKFLOW = "not-available"


def store_unreachable(error: Exception) -> bool:
    """Whether ``error`` means the store could not be reached.

    A builtin ConnectionError is always an outage. A psycopg2 OperationalError
    is an outage only when the server never answered: no SQLSTATE at all (pure
    connection failure) or an explicit Class-08 connection-exception code.
    Sibling OperationalError subclasses that carry their own code — a statement
    timeout (QueryCanceled 57014), an admin shutdown (57P01) — mean Postgres
    was up and reachable; mislabeling those as outages would send operators to
    restart a healthy database.
    """
    if isinstance(error, ConnectionError):
        return True
    if not isinstance(error, psycopg2.OperationalError):
        return False
    return error.pgcode is None or error.pgcode.startswith("08")


def stored_chunk_count(database_url: str) -> int:
    """How many Chunks the pgvector store currently holds.

    Opens a short-lived connection per probe so a stale pooled connection can
    never wedge the guard and an ingestion that happens after boot is visible
    on the next request. Raises if the store cannot be reached; callers turn
    that into its own Not-available response rather than a server error.
    """
    with closing(connect(database_url)) as connection:
        return PgVectorStore(connection).count_chunks()


def vector_store_is_empty(database_url: str) -> bool:
    """Whether the store holds no Chunks — this iteration's un-ingested check."""
    return stored_chunk_count(database_url) == 0


def log_startup_store_warning(settings: Settings) -> None:
    """Warn once at boot when the vector store cannot serve Live mode.

    An empty or unreachable store is never a startup failure — the process
    boots in any mode so Demo mode keeps serving keyless stakeholders
    untouched. Exactly one warning, naming the ingest command when the store
    is empty.
    """
    try:
        empty = vector_store_is_empty(settings.database_url)
    except Exception as error:  # any store failure must not abort boot
        logger.warning(
            "Could not check whether the Corpus is ingested (%s); "
            "Live mode requests will fail until the vector store is reachable.",
            error,
        )
        return
    if empty:
        logger.warning(
            "Vector store holds no Chunks yet: Live mode answers stay Not-available "
            "until the Corpus is ingested (%s).",
            INGEST_COMMAND,
        )


def not_available_response(actions: list[str], summary: str) -> AnalyzeResponse:
    """The sibling shape every Not-available response shares.

    Never serves demo content: empty Findings and Citations, the
    "not-available" workflow marker, and the English-only Known limitation.
    """
    return AnalyzeResponse(
        answer=Answer(findings=[], citations=[], actions=actions),
        trace=Trace(workflow=NOT_AVAILABLE_WORKFLOW, summary=summary),
        detailed_trace=[],
        known_limitations=[ENGLISH_ONLY_LIMITATION],
    )


def missing_api_key_response() -> AnalyzeResponse:
    """Not-available response for Live mode while no provider key is configured.

    ADR-0008: the backend boots without a Live-capable configuration, so a
    missing key is a per-request Readiness gap, not a boot refusal. The reply
    names the exact variable and the documented key home — settings load at
    boot, so picking the key up does need a restart, unlike ingestion.
    """
    return not_available_response(
        actions=[
            "Live mode needs an LLM provider key, but OPENROUTER_API_KEY is not configured.",
            "Set it in .env (or export it) and restart the backend; no re-ingestion is needed.",
            SWITCH_TO_DEMO_ACTION,
        ],
        summary="Live mode selected but no provider key is configured; no analysis performed.",
    )


def un_ingested_corpus_response() -> AnalyzeResponse:
    """Not-available response for Live mode while the vector store is empty.

    Names the exact ingest command and how to retry so recovery needs no
    documentation.
    """
    return not_available_response(
        actions=[
            "The Corpus is not ingested yet, so Live mode has nothing to retrieve from.",
            f"Ingest it once with: {INGEST_COMMAND}",
            f"Inside the Docker stack, run: {INGEST_COMMAND_IN_STACK}",
            "Ingestion takes effect immediately — re-run your request afterwards; no restart is needed.",
            SWITCH_TO_DEMO_ACTION,
        ],
        summary="Live mode selected but the vector store is empty; no retrieval performed.",
    )


def unreachable_store_response(error: Exception) -> AnalyzeResponse:
    """Not-available response for Live mode while the store cannot be reached.

    An outage is not the un-ingested case: re-running ingestion cannot fix a
    store the backend cannot reach, so this names the outage and never the
    ingest command. The underlying cause is included — the stack is
    permanently local (ADR-0002), so the operator reading the answer can act
    on it.
    """
    return not_available_response(
        actions=[
            f"The vector store is not reachable ({error}), so Live mode has nothing to retrieve from.",
            f"Start the stack's database with: {START_POSTGRES_COMMAND}",
            "Once it is up, re-run your request; no restart is needed.",
            SWITCH_TO_DEMO_ACTION,
        ],
        summary="Live mode selected but the vector store is unreachable; no retrieval performed.",
    )


def unreachable_llm_response(error: Exception) -> AnalyzeResponse:
    """Not-available response for Live mode while OpenRouter cannot be reached.

    Mirrors the unreachable-store case: a network outage is not a server
    error, and the operator reading the answer can act on it — check the
    connection and key — so the reply names the cause verbatim. A reachable
    provider that rejected or malformed the answer never lands here; that
    stays a server error.
    """
    return not_available_response(
        actions=[
            f"The LLM provider is not reachable ({error}), so Live mode cannot run the workflow.",
            "Check the network connection and OPENROUTER_API_KEY, then re-run your request; no restart is needed.",
            SWITCH_TO_DEMO_ACTION,
        ],
        summary="Live mode selected but the LLM provider is unreachable; no analysis performed.",
    )
