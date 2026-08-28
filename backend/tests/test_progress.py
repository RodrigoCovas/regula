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
    PhaseReport,
    ProgressRegistry,
    progress_sink,
    unknown_request_response,
)

from fakes import FakeClock
from progress_assertions import assert_phases_in_agent_order, assert_unknown_request_reply


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
    registry.transition("run-1", PhaseReport(phase="planner", message="decomposing the question"))
    registry.transition("run-1", PhaseReport(phase="researcher", message="retrieving Evidence"))
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
    registry.transition("run-1", PhaseReport(phase="planner", message="decomposing"))
    snapshot = registry.get("run-1")
    assert snapshot is not None
    assert snapshot.transitions[0].elapsed_ms == 1250


def test_unknown_id_reads_back_as_none_and_transitions_are_rejected():
    registry = ProgressRegistry()
    assert registry.get("never-registered") is None
    assert (
        registry.transition("never-registered", PhaseReport(phase="planner", message="x"))
        is False
    )


def test_expired_entries_read_back_as_unknown_and_are_purged():
    clock = FakeClock()
    registry = ProgressRegistry(ttl_seconds=10, now=clock)
    registry.register("run-1")
    registry.transition("run-1", PhaseReport(phase="planner", message="decomposing"))
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
    assert registry.transition("run-1", PhaseReport(phase="researcher", message="late")) is False


def test_sink_reports_into_the_bound_registry():
    registry = ProgressRegistry()
    registry.register("run-1")
    sink = progress_sink("run-1", registry=registry)
    sink(PhaseReport(phase="planner", message="decomposing"))
    snapshot = registry.get("run-1")
    assert snapshot is not None
    assert snapshot.phase == "planner"
    assert snapshot.message == "decomposing"


def test_sink_drop_for_an_unknown_id_warns_instead_of_raising(caplog):
    registry = ProgressRegistry()
    sink = progress_sink("unknown", registry=registry)
    with caplog.at_level(logging.WARNING, logger="src.progress"):
        sink(PhaseReport(phase="planner", message="decomposing"))
    assert "unknown" in caplog.text
    assert registry.get("unknown") is None


def test_concurrent_transitions_all_land():
    registry = ProgressRegistry()
    registry.register("run-1")

    def worker(thread_no: int) -> None:
        for index in range(100):
            assert registry.transition(
                "run-1", PhaseReport(phase="planner", message=f"t{thread_no}-{index}")
            )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, range(8)))
    snapshot = registry.get("run-1")
    assert snapshot is not None
    assert len(snapshot.transitions) == 8 * 100
    messages = {t.message for t in snapshot.transitions}
    assert messages == {f"t{thread_no}-{index}" for thread_no in range(8) for index in range(100)}


def test_unknown_request_response_is_not_available_shaped():
    reply = unknown_request_response("nope")
    assert_unknown_request_reply(reply.model_dump(), "nope")


def test_default_ttl_keeps_a_record_readable_through_a_slow_live_run():
    """The default TTL is behavioural: a record still reads back — never as
    expired — after the slowest plausible Live run."""
    clock = FakeClock()
    registry = ProgressRegistry(now=clock)
    registry.register("run-1")
    clock.advance(10 * 60)
    snapshot = registry.get("run-1")
    assert snapshot is not None
    assert snapshot.request_id == "run-1"


def test_fail_records_terminal_error_in_snapshot():
    registry = ProgressRegistry()
    registry.register("run-1")
    registry.transition("run-1", PhaseReport(phase="planner", message="decomposing"))
    assert registry.fail("run-1", "the LLM provider rejected the request") is True
    snapshot = registry.get("run-1")
    assert snapshot is not None
    assert snapshot.error == "the LLM provider rejected the request"
    assert snapshot.phase == "planner"
    assert [(t.phase, t.message) for t in snapshot.transitions] == [
        ("planner", "decomposing"),
    ]


def test_fail_for_unknown_id_returns_false():
    registry = ProgressRegistry()
    assert registry.fail("never-registered", "boom") is False


def test_failed_entry_still_expires_by_ttl():
    clock = FakeClock()
    registry = ProgressRegistry(ttl_seconds=10, now=clock)
    registry.register("run-1")
    registry.fail("run-1", "unexpected error")
    clock.advance(10)
    assert registry.get("run-1") is None


def test_fail_prevents_later_complete():
    registry = ProgressRegistry()
    registry.register("run-1")
    registry.fail("run-1", "boom")
    from src.models import AnalyzeResponse, Answer, Trace
    response = AnalyzeResponse(
        answer=Answer(findings=[], citations=[], actions=[]),
        trace=Trace(workflow="test", summary=""),
    )
    assert registry.complete("run-1", response) is False
    snapshot = registry.get("run-1")
    assert snapshot is not None
    assert snapshot.error == "boom"
    assert registry.get_completed_response("run-1") is None


def test_complete_prevents_later_fail():
    registry = ProgressRegistry()
    registry.register("run-1")
    from src.models import AnalyzeResponse, Answer, Trace
    response = AnalyzeResponse(
        answer=Answer(findings=[], citations=[], actions=[]),
        trace=Trace(workflow="test", summary=""),
    )
    assert registry.complete("run-1", response) is True
    assert registry.fail("run-1", "late error") is False
    snapshot = registry.get("run-1")
    assert snapshot is not None
    assert snapshot.error is None
    assert registry.get_completed_response("run-1") is response


def test_live_workflow_reports_each_phase_in_order_through_the_sink():
    """The workflow→registry seam: run_live_analysis reports every phase,
    in agent order, with informative messages — never through HTTP."""
    from src.live_workflow import run_live_analysis
    from src.models import AnalyzeRequest, Scenario

    from fakes import FakeRetriever, make_offline_llm

    reported: list[PhaseReport] = []

    def sink(report: PhaseReport) -> None:
        reported.append(report)

    run_live_analysis(
        AnalyzeRequest(
            scenario=Scenario(id="my-app", description="A Spanish fintech startup that uses machine learning to assess creditworthiness for consumer loans. The platform automatically approves or denies applications based on applicant data including income, employment history, and spending patterns. The company operates only in Spain and plans to expand to other EU markets."),
            question="What regulations apply?",
        ),
        llm=make_offline_llm(),
        retriever=FakeRetriever(),
        progress=sink,
    )
    assert_phases_in_agent_order([(report.phase, report.message) for report in reported])
