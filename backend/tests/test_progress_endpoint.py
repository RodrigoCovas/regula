"""The progress endpoint over the HTTP seam (spec #25, ticket #28).

Boots Live mode with deterministic fakes at the composition root — the same
seams the pipeline tests use — and reads GET /api/progress/{request_id}
against the same registry the analyze endpoint writes through the
X-Request-Id header.
"""

import sys
import threading
import time
from pathlib import Path

# Ensure backend.src is importable when running tests from repo root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import pytest

from src.main import REQUEST_ID_HEADER
from src.progress import ProgressRegistry

from conftest import boot_live_with_fakes
from fakes import FakeClock, FakeRetriever, make_offline_llm
from progress_assertions import assert_phases_in_agent_order, assert_unknown_request_reply


@pytest.fixture(autouse=True)
def _fresh_progress_registry(monkeypatch):
    """Every endpoint test reads a fresh registry: request ids can never leak
    between tests through the module-global shared instance."""
    import src.main as main

    monkeypatch.setattr(main, "progress_registry", ProgressRegistry())
    yield


def post_canonical(client, request_id=None):
    """POST the canonical Scenario, optionally tagged with a request id."""
    headers = {REQUEST_ID_HEADER: request_id} if request_id else {}
    return client.post(
        "/api/analyze",
        headers=headers,
        json={
            "scenario": {"id": "spanish-fintech-startup-uses-9e165169", "description": "A Spanish fintech startup that uses machine learning to assess creditworthiness for consumer loans. The platform automatically approves or denies applications based on applicant data including income, employment history, and spending patterns. The company operates only in Spain and plans to expand to other EU markets."},
            "question": "What regulations apply?",
        },
    )


def get_progress(client, request_id):
    return client.get(f"/api/progress/{request_id}")


def test_live_run_reports_each_phase_in_order_through_the_endpoint(monkeypatch):
    """A completed Live run returns the AnalyzeResponse through the progress
    endpoint, allowing the frontend to stop polling and display results."""
    with boot_live_with_fakes(monkeypatch, make_offline_llm(), FakeRetriever()) as live_client:
        resp = post_canonical(live_client, request_id="run-1")
        assert resp.status_code == 200
        progress = get_progress(live_client, "run-1")
        assert progress.status_code == 200
        data = progress.json()
    # After completion, the endpoint returns the AnalyzeResponse, not the snapshot
    assert data["trace"]["workflow"] == "planner -> researcher -> verifier -> proposer"
    assert "answer" in data
    assert "findings" in data["answer"]
    assert "actions" in data["answer"]
    assert "citations" in data["answer"]


class GatedLlm:
    """Wraps a ScriptedLlm: the first completion call blocks until released,
    keeping the workflow paused inside the Planner's LLM call."""

    def __init__(self, base):
        self._base = base
        self._gate = threading.Event()

    def complete(self, system, user, schema):
        if not self._gate.is_set():
            self._gate.wait(timeout=10)
        return self._base.complete(system, user, schema)

    def release(self):
        self._gate.set()


def _wait_for_phase(client, request_id, phase, deadline=5.0):
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        data = get_progress(client, request_id).json()
        if data.get("phase") == phase:
            return data
        time.sleep(0.01)
    raise AssertionError(f"phase {phase!r} never appeared for request id {request_id!r}")


def test_phases_are_visible_through_the_endpoint_while_the_run_reports_them(monkeypatch):
    """Progress is visible mid-run, not after the fact: while the workflow
    is paused inside the Planner's LLM call, the endpoint already serves the
    reported Planner phase; once the run completes, the AnalyzeResponse is
    returned instead of the progress snapshot."""
    from src.config import load_settings

    import src.main as main

    gated = GatedLlm(make_offline_llm())
    live_client = boot_live_with_fakes(monkeypatch, gated, FakeRetriever())
    # The client stays un-entered so each request runs in its own portal:
    # an entered client funnels every request through one portal, and the
    # blocked POST would wedge the polling GET. The un-entered client never
    # runs the lifespan, so Live mode comes from replacing the module-global
    # settings directly — monkeypatched, so nothing leaks between tests.
    monkeypatch.setattr(main, "settings", load_settings())
    result = {}

    def submit():
        result["resp"] = post_canonical(live_client, request_id="inflight")

    thread = threading.Thread(target=submit, daemon=True)
    thread.start()
    try:
        mid_run = _wait_for_phase(live_client, "inflight", "planner")
        assert [t["phase"] for t in mid_run["transitions"]] == ["planner"]
        assert mid_run["message"], "an informative message accompanies the phase"
    finally:
        gated.release()
        thread.join(timeout=10)
    assert result["resp"].status_code == 200
    # After completion, the endpoint returns the AnalyzeResponse
    final = get_progress(live_client, "inflight").json()
    assert final["trace"]["workflow"] == "planner -> researcher -> verifier -> proposer"


def test_unknown_request_id_gets_the_not_available_shaped_reply(monkeypatch):
    """A request id the registry knows nothing about gets the Not-available
    sibling shape naming what happened — never a bare 404, never demo
    content."""
    with boot_live_with_fakes(monkeypatch, make_offline_llm(), FakeRetriever()) as live_client:
        resp = get_progress(live_client, "never-submitted")
    assert resp.status_code == 200
    assert_unknown_request_reply(resp.json(), "never-submitted")


def test_expired_progress_reads_back_as_unknown_through_the_endpoint(monkeypatch):
    """The TTL cleanup is observable through the endpoint: a run whose
    record expired reads back as the Not-available reply, never a stale
    phase."""
    import src.main as main

    clock = FakeClock()
    monkeypatch.setattr(main, "progress_registry", ProgressRegistry(ttl_seconds=5, now=clock))
    with boot_live_with_fakes(monkeypatch, make_offline_llm(), FakeRetriever()) as live_client:
        resp = post_canonical(live_client, request_id="short-lived")
        assert resp.status_code == 200
        clock.advance(6)
        data = get_progress(live_client, "short-lived").json()
    assert data["trace"]["workflow"] == "not-available"
    assert "short-lived" in " ".join(data["answer"]["actions"]).lower()


def test_requests_without_a_request_id_header_register_no_progress(monkeypatch):
    """The header is the only registration channel: without it, the analyze
    endpoint leaves no trace and the id reads back as unknown."""
    with boot_live_with_fakes(monkeypatch, make_offline_llm(), FakeRetriever()) as live_client:
        resp = post_canonical(live_client)
        assert resp.status_code == 200
        data = get_progress(live_client, "absent").json()
    assert data["trace"]["workflow"] == "not-available"
