"""Progress registry: in-memory, thread-safe phase reporting for Live runs.

Long Live-mode answers need visible progress while the workflow runs — the
frontend polls a progress endpoint instead of staring at a silent spinner.
This module is the backend half of that contract (spec #25, ticket #28).

The Live workflow reports each phase transition — Planner → Researcher →
Verifier → Proposer, named after the workflow agents — through an optional
progress sink (a thin callback into the shared registry) with an
informative message per phase. The progress endpoint reads the same
registry and returns the current phase plus the ordered transition
history; an unknown request id gets the Not-available-shaped reply naming
what happened.

Entries expire: every operation first sweeps entries older than the TTL,
so finished runs never accumulate and an expired request id reads back as
unknown — never as a stale phase.

In-memory state is safe under the single-worker deployment (the
permanently-local stack, ADR-0002): both ``python -m backend.src.main``
and the Docker image run one uvicorn worker, so every request served by
this process shares the same registry. Multiple uvicorn workers would
shard the registry and make progress unavailable across requests — that
deployment, like any shared or remote progress store, is out of scope.
"""

import logging
import threading
import time
from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field

from .availability import not_available_response
from .models import AnalyzeResponse

logger = logging.getLogger(__name__)

# How long a progress record stays readable after registration. Covers a
# generous Live run; expired records read back as unknown.
PROGRESS_TTL_SECONDS = 15 * 60

# The workflow's four phases, named after the workflow agents in the order
# they run — the one vocabulary the sink, the workflow, and the frontend's
# stage display share.
WorkflowPhase = Literal["planner", "researcher", "verifier", "proposer"]


class PhaseReport(BaseModel):
    """One reported phase entry: which workflow agent was entered and what
    it is doing — the pair the sink, the registry, and the endpoint all
    carry together."""

    phase: WorkflowPhase
    message: str


# The one seam the workflow reports through: a callback receiving each
# entered phase and its informative message.
ProgressSink = Callable[[PhaseReport], None]


class ProgressTransition(PhaseReport):
    """One recorded phase entry plus how long after registration it was
    reached."""

    elapsed_ms: int


class ProgressSnapshot(BaseModel):
    """The registry's read shape for one request id: the current phase and
    the ordered transition history.

    ``phase`` and ``message`` are the latest transition's — never a copy
    stored alongside the history.

    ``error`` is set when the run terminated with a failure — the frontend
    reads it as a terminal outcome and stops polling.
    """

    request_id: str
    phase: Optional[WorkflowPhase] = None
    message: Optional[str] = None
    transitions: list[ProgressTransition] = Field(default_factory=list)
    error: Optional[str] = None


class _Entry:
    """One registered run's mutable progress record."""

    def __init__(self, request_id: str, started_at: float) -> None:
        self.request_id = request_id
        self.started_at = started_at
        self.transitions: list[ProgressTransition] = []
        self.completed_response: Optional[AnalyzeResponse] = None
        self.error: Optional[str] = None


class ProgressRegistry:
    """Thread-safe in-memory store of Live-run phase progress.

    ``register`` opens an entry for a request id (the frontend's
    per-submission id, delivered on POST /api/analyze as the X-Request-Id
    header); ``transition`` appends one phase entry in order; ``get``
    returns the current snapshot, or None when the id is unknown — never
    submitted, or expired. Every operation sweeps expired entries first.
    """

    def __init__(
        self,
        ttl_seconds: float = PROGRESS_TTL_SECONDS,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._now = now
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.Lock()

    def register(self, request_id: str) -> None:
        """Open (or reset) the progress record for one request id."""
        with self._lock:
            self._purge_expired_locked()
            self._entries[request_id] = _Entry(request_id, self._now())

    def transition(self, request_id: str, report: PhaseReport) -> bool:
        """Record one phase transition; False when the id is unknown or
        expired — the record must never silently resurrect."""
        with self._lock:
            self._purge_expired_locked()
            entry = self._entries.get(request_id)
            if entry is None:
                return False
            entry.transitions.append(
                ProgressTransition(
                    phase=report.phase,
                    message=report.message,
                    elapsed_ms=int((self._now() - entry.started_at) * 1000),
                )
            )
            return True

    def get(self, request_id: str) -> Optional[ProgressSnapshot]:
        """The current snapshot, or None when the id is unknown or expired."""
        with self._lock:
            self._purge_expired_locked()
            entry = self._entries.get(request_id)
            if entry is None:
                return None
            current = entry.transitions[-1] if entry.transitions else None
            return ProgressSnapshot(
                request_id=entry.request_id,
                phase=current.phase if current else None,
                message=current.message if current else None,
                transitions=list(entry.transitions),
                error=entry.error,
            )

    def complete(self, request_id: str, response: AnalyzeResponse) -> bool:
        """Mark the run as complete with the final response; False when the id
        is unknown, expired, or already failed — a terminal failure never
        yields to a later completion."""
        with self._lock:
            self._purge_expired_locked()
            entry = self._entries.get(request_id)
            if entry is None:
                return False
            if entry.error is not None:
                return False
            entry.completed_response = response
            return True

    def fail(self, request_id: str, error_message: str) -> bool:
        """Mark the run as terminally failed; False when the id is unknown,
        expired, or already completed — a terminal outcome never yields to a
        later failure. The error is visible through ``get`` — the frontend
        reads it as a terminal outcome and stops polling."""
        with self._lock:
            self._purge_expired_locked()
            entry = self._entries.get(request_id)
            if entry is None:
                return False
            if entry.completed_response is not None:
                return False
            entry.error = error_message
            return True

    def get_completed_response(self, request_id: str) -> Optional[AnalyzeResponse]:
        """The completed response if the run finished, None otherwise."""
        with self._lock:
            self._purge_expired_locked()
            entry = self._entries.get(request_id)
            if entry is None:
                return None
            return entry.completed_response

    def _purge_expired_locked(self) -> None:
        """Drop entries whose TTL has passed (the caller holds the lock)."""
        now = self._now()
        expired = [
            request_id
            for request_id, entry in self._entries.items()
            if now - entry.started_at >= self._ttl_seconds
        ]
        for request_id in expired:
            del self._entries[request_id]


# The shared instance both request paths default to: the analyze endpoint
# registers runs and hands the workflow its sink against this registry; the
# progress endpoint reads it back.
progress_registry = ProgressRegistry()


def progress_sink(request_id: str, *, registry: Optional[ProgressRegistry] = None) -> ProgressSink:
    """A phase reporter bound to one request id, against the shared registry.

    A transition for a request id that expired mid-run is dropped with a
    warning, never raised into the workflow.
    """
    target = registry if registry is not None else progress_registry

    def report(phase_report: PhaseReport) -> None:
        if not target.transition(request_id, phase_report):
            logger.warning(
                "progress transition for unknown or expired request id %r dropped (phase %r)",
                request_id,
                phase_report.phase,
            )

    return report


def unknown_request_response(request_id: str) -> AnalyzeResponse:
    """Not-available reply for a request id the registry knows nothing about.

    Names what happened — never submitted with this id, or the record
    expired — and how to proceed, in the sibling shape every Not-available
    response shares.
    """
    return not_available_response(
        actions=[
            f"No progress was recorded for request id '{request_id}': the Scenario was never "
            f"submitted with this id, or its progress record expired (records live "
            f"{PROGRESS_TTL_SECONDS // 60} minutes).",
            "Re-submit the Scenario: every submission tracks progress under its own request id.",
        ],
        summary=f"No progress record for request id '{request_id}'.",
    )
