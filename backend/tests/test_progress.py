"""The progress registry: register, transition, expiry, unknown ids (spec #25, ticket #28).

The registry is the pure module tested directly — no HTTP. The endpoint
reads the same registry (test_progress_endpoint.py); the Live workflow
reports through the sink (the run_live_analysis seam below).
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Ensure backend.src is importable when running tests from repo root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import logging

from src.progress import (
    PROGRESS_TTL_SECONDS,
    ProgressRegistry,
    progress_sink,
    unknown_request_response,
)

from fakes import FakeClock
from test_analyze import assert_not_available_shape


def test_register_then_get_reports_pending_with_empty_history():
    registry = ProgressRegistry()
    registry.register("run-1")
    snapshot = registry.get("run-1")
    assert snapshot is not None
    assert snapshot.request_id == "run-1"
    assert snapshot.phase is None
    assert snapshot.message is None
    assert snapshot.transitions == []


def test_transitions_accumulate_in_order_and_get_reports_the_current_phase():
    registry = ProgressRegistry()
    registry.register("run-1")
    registry.transition("run-1", "planner", "decomposing the question")
    registry.transition("run-1", "researcher", "retrieving Evidence")
    snapshot = registry.get("run-1")
    assert snapshot is not None
    assert snapshot.phase == "researcher"
    assert snapshot.message == "retrieving Evidence"
    assert [(t.phase, t.message) for t in snapshot.transitions] == [
        ("planner", "decomposing the question"),
        ("researcher", "retrieving Evidence"),
    ]


def test_transition_elapsed_is_since_registration():
    clock = FakeClock()
    registry = ProgressRegistry(now=clock)
    registry.register("run-1")
    clock.advance(1.25)
    registry.transition("run-1", "planner", "decomposing")
    snapshot = registry.get("run-1")
    assert snapshot is not None
    assert snapshot.transitions[0].elapsed_ms == 1250


def test_unknown_id_reads_back_as_none_and_transitions_are_rejected():
    registry = ProgressRegistry()
    assert registry.get("never-registered") is None
    assert registry.transition("never-registered", "planner", "x") is False


def test_expired_entries_read_back_as_unknown_and_are_purged():
    clock = FakeClock()
    registry = ProgressRegistry(ttl_seconds=10, now=clock)
    registry.register("run-1")
    registry.transition("run-1", "planner", "decomposing")
    clock.advance(10)
    # Expired: reads back as unknown, never as a stale phase.
    assert registry.get("run-1") is None
    # The expired record is purged: re-registering opens a fresh, empty entry.
    registry.register("run-1")
    snapshot = registry.get("run-1")
    assert snapshot is not None
    assert snapshot.transitions == []


def test_a_transition_for_an_expired_entry_is_dropped():
    clock = FakeClock()
    registry = ProgressRegistry(ttl_seconds=10, now=clock)
    registry.register("run-1")
    clock.advance(10)
    assert registry.transition("run-1", "researcher", "late") is False


def test_sink_reports_into_the_bound_registry():
    registry = ProgressRegistry()
    registry.register("run-1")
    sink = progress_sink("run-1", registry=registry)
    sink("planner", "decomposing")
    snapshot = registry.get("run-1")
    assert snapshot is not None
    assert snapshot.phase == "planner"
    assert snapshot.message == "decomposing"


def test_sink_drop_for_an_unknown_id_warns_instead_of_raising(caplog):
    registry = ProgressRegistry()
    sink = progress_sink("unknown", registry=registry)
    with caplog.at_level(logging.WARNING, logger="src.progress"):
        sink("planner", "decomposing")
    assert "unknown" in caplog.text
    assert registry.get("unknown") is None


def test_concurrent_transitions_all_land():
    registry = ProgressRegistry()
    registry.register("run-1")

    def worker(thread_no: int) -> None:
        for index in range(100):
            assert registry.transition("run-1", "planner", f"t{thread_no}-{index}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, range(8)))
    snapshot = registry.get("run-1")
    assert snapshot is not None
    assert len(snapshot.transitions) == 8 * 100
    messages = {t.message for t in snapshot.transitions}
    assert messages == {f"t{thread_no}-{index}" for thread_no in range(8) for index in range(100)}


def test_unknown_request_response_is_not_available_shaped():
    reply = unknown_request_response("nope")
    data = reply.model_dump()
    assert_not_available_shape(data)
    actions = " ".join(data["answer"]["actions"]).lower()
    assert "nope" in actions, "the reply names the unknown id"
    assert "expired" in actions, "the reply names the TTL expiry as a cause"
    assert "re-submit" in actions, "the reply says how to proceed"
    assert any("english-only" in line.lower() for line in data["known_limitations"])


def test_default_ttl_is_long_enough_for_a_slow_live_run():
    assert PROGRESS_TTL_SECONDS >= 10 * 60


def test_live_workflow_reports_each_phase_in_order_through_the_sink():
    """The workflow→registry seam: run_live_analysis reports every phase,
    in agent order, with informative messages — never through HTTP."""
    from src.live_workflow import run_live_analysis
    from src.models import AnalyzeRequest, Scenario

    from fakes import FakeRetriever, make_offline_llm

    reported: list[tuple[str, str]] = []

    def sink(phase: str, message: str) -> None:
        reported.append((phase, message))

    run_live_analysis(
        AnalyzeRequest(
            scenario=Scenario(id="my-app", description="Spanish fintech lending"),
            question="What regulations apply?",
        ),
        llm=make_offline_llm(),
        retriever=FakeRetriever(),
        progress=sink,
    )
    assert [phase for phase, _ in reported] == ["planner", "researcher", "verifier", "proposer"]
    for _, message in reported:
        assert message, "every phase carries an informative message"
    messages = " ".join(message for _, message in reported).lower()
    assert "target" in messages, "the Planner's message names research targets"
    assert "evidence" in messages, "the Researcher's message names Evidence"
    assert "claim" in messages, "the Verifier's message names Claims"
    assert "action" in messages, "the Proposer's message names Actions"
