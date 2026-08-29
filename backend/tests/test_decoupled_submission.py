"""Decoupled Live submission: POST returns promptly, analysis continues
in a background thread (spec #25, issue #41).

The UI-facing submission path must not depend on the original POST
connection remaining open. A dropped client or proxy connection does not
lose the run; progress polling still reaches the eventual Answer or a
terminal failure.

Boots Live mode with deterministic fakes at the composition root and
verifies:
- POST /api/analyze with X-Request-Id returns a ProgressSnapshot promptly
- The analysis continues in the background and completes
- GET /api/progress/{request_id} returns the AnalyzeResponse after completion
- Workflow failures are recorded and visible through polling
- Non-UI callers (no request id) retain the synchronous contract
"""

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import pytest

from src.live_workflow import LIVE_WORKFLOW_MARKER
from src.main import REQUEST_ID_HEADER
from src.progress import ProgressRegistry

from conftest import boot_live_with_fakes, poll_progress
from fakes import FakeRetriever, make_offline_llm


@pytest.fixture(autouse=True)
def _fresh_progress_registry(monkeypatch):
    import src.main as main

    monkeypatch.setattr(main, "progress_registry", ProgressRegistry())
    yield


@pytest.fixture(autouse=True)
def _clean_background_threads():
    import src.main as main

    yield
    with main._background_threads_lock:
        threads = list(main._background_threads)
    for thread in threads:
        thread.join(timeout=5)
    with main._background_threads_lock:
        main._background_threads.clear()


def post_canonical(client, request_id=None):
    headers = {REQUEST_ID_HEADER: request_id} if request_id else {}
    return client.post(
        "/api/analyze",
        headers=headers,
        json={
            "scenario": {"id": "spanish-fintech-startup-uses-9e165169", "description": "A Spanish fintech startup that uses machine learning to assess creditworthiness for consumer loans. The platform automatically approves or denies applications based on applicant data including income, employment history, and spending patterns. The company operates only in Spain and plans to expand to other EU markets."},
            "question": "What regulations apply?",
        },
    )


class GatedLlm:
    """Wraps a ScriptedLlm: the first completion call blocks until released."""

    def __init__(self, base):
        self._base = base
        self._gate = threading.Event()
        self.call_count = 0

    def complete(self, system, user, schema):
        self.call_count += 1
        if not self._gate.is_set():
            self._gate.wait(timeout=10)
        return self._base.complete(system, user, schema)

    def release(self):
        self._gate.set()


def test_post_with_request_id_returns_promptly_with_progress_snapshot(monkeypatch):
    """POST /api/analyze with X-Request-Id returns a ProgressSnapshot
    promptly, before the analysis completes. The response has the
    ProgressSnapshot shape (request_id, transitions), not AnalyzeResponse."""
    gated = GatedLlm(make_offline_llm())
    live_client = boot_live_with_fakes(monkeypatch, gated, FakeRetriever())

    from src.config import load_settings
    import src.main as main
    monkeypatch.setattr(main, "settings", load_settings())

    result = {}

    def submit():
        result["resp"] = post_canonical(live_client, request_id="decoupled-run")

    thread = threading.Thread(target=submit, daemon=True)
    thread.start()

    try:
        end = time.monotonic() + 5.0
        while time.monotonic() < end:
            if "resp" in result:
                break
            time.sleep(0.01)
        assert "resp" in result, "POST did not return promptly"

        resp = result["resp"]
        assert resp.status_code == 200
        data = resp.json()

        assert data["request_id"] == "decoupled-run"
        assert isinstance(data["transitions"], list)
        assert "answer" not in data, "POST returned AnalyzeResponse instead of ProgressSnapshot"
    finally:
        gated.release()
        thread.join(timeout=10)


def test_background_analysis_completes_and_is_visible_through_polling(monkeypatch):
    """The analysis continues in the background after POST returns.
    Once complete, GET /api/progress/{request_id} returns the AnalyzeResponse."""
    gated = GatedLlm(make_offline_llm())
    live_client = boot_live_with_fakes(monkeypatch, gated, FakeRetriever())

    from src.config import load_settings
    import src.main as main
    monkeypatch.setattr(main, "settings", load_settings())

    resp = post_canonical(live_client, request_id="bg-complete")
    assert resp.status_code == 200
    post_data = resp.json()
    assert post_data["request_id"] == "bg-complete"

    gated.release()

    data = poll_progress(live_client, "bg-complete", lambda d: "answer" in d, timeout=10.0)

    assert "answer" in data, "analysis did not complete in background"
    assert data["trace"]["workflow"] == LIVE_WORKFLOW_MARKER
    assert "findings" in data["answer"]


def test_background_failure_is_visible_through_polling(monkeypatch):
    """A workflow failure in the background thread is recorded and visible
    through the progress endpoint as a terminal error."""

    class RaisingLlm:
        def complete(self, system, user, schema):
            raise RuntimeError("background provider failure")

    with boot_live_with_fakes(monkeypatch, RaisingLlm(), FakeRetriever(), raise_server_exceptions=False) as live_client:
        resp = post_canonical(live_client, request_id="bg-failed")

        assert resp.status_code == 200
        post_data = resp.json()
        assert post_data["request_id"] == "bg-failed"
        assert isinstance(post_data["transitions"], list)

        data = poll_progress(live_client, "bg-failed", lambda d: d.get("error"))

    assert data["request_id"] == "bg-failed"
    assert isinstance(data.get("error"), str)
    assert "background provider failure" in data["error"]


def test_post_without_request_id_retains_synchronous_contract(monkeypatch):
    """Non-UI callers (no X-Request-Id) still get the synchronous
    AnalyzeResponse from POST /api/analyze."""
    with boot_live_with_fakes(monkeypatch, make_offline_llm(), FakeRetriever()) as live_client:
        resp = post_canonical(live_client)
        assert resp.status_code == 200
        data = resp.json()

    assert "answer" in data
    assert "findings" in data["answer"]
    assert data["trace"]["workflow"] == LIVE_WORKFLOW_MARKER
    assert "request_id" not in data or data.get("transitions") is None


def test_availability_gate_failure_registers_progress_for_polling(monkeypatch):
    """When an availability gate fails (e.g., un-ingested corpus), the
    result is registered in the progress registry so polling can retrieve it."""
    with boot_live_with_fakes(monkeypatch, make_offline_llm(), FakeRetriever(), chunk_count=0) as live_client:
        resp = post_canonical(live_client, request_id="gate-failed")
        assert resp.status_code == 200
        post_data = resp.json()
        assert post_data["request_id"] == "gate-failed"

        data = poll_progress(live_client, "gate-failed", lambda d: "answer" in d)
    assert "answer" in data
    assert data["trace"]["workflow"] == "not-available"


def test_submission_survives_client_disconnect_simulation(monkeypatch):
    """End-to-end: the analysis completes even after the submitting client
    disconnects. The POST returns promptly with a ProgressSnapshot; a
    separate polling client (simulating a different connection) retrieves
    the eventual Answer after the submitting client is gone."""
    gated = GatedLlm(make_offline_llm())
    live_client = boot_live_with_fakes(monkeypatch, gated, FakeRetriever())

    from src.config import load_settings
    import src.main as main
    monkeypatch.setattr(main, "settings", load_settings())

    with live_client:
        resp = post_canonical(live_client, request_id="disconnect-run")
        assert resp.status_code == 200
        post_data = resp.json()
        assert post_data["request_id"] == "disconnect-run"
        assert "answer" not in post_data, "POST should return ProgressSnapshot, not AnalyzeResponse"

        from fastapi.testclient import TestClient
        poll_client = TestClient(main.app)
        with poll_client:
            gated.release()

            data = poll_progress(poll_client, "disconnect-run", lambda d: "answer" in d, timeout=10.0)

        assert "answer" in data, "analysis did not complete after submitting client disconnected"
        assert data["trace"]["workflow"] == LIVE_WORKFLOW_MARKER
