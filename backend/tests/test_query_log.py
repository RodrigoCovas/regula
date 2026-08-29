"""Per-request observability over the HTTP seam (spec #9, ticket #19).

Every /api/analyze request appends exactly one JSONL record to the
configured query log: what ran (mode, workflow), how it behaved (explicit
success/failure status, latency, retrieved-Chunk counts), and what it cost
(token usage when the LLM ran). Failures are recorded rather than silently
absent. No network, no Ollama, no pgvector: fakes ride in at the
composition root.
"""

import sys
from pathlib import Path

# Ensure backend.src is importable when running tests from repo root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import json
from datetime import datetime
from types import SimpleNamespace

from fastapi.testclient import TestClient

from src import availability
from src.llm import LlmError
from src.live_workflow import LIVE_WORKFLOW_MARKER
from src.main import app
from src.query_log import aggregate_token_usage, append_query_record

from conftest import boot_live_with_fakes
from fakes import FakeRetriever, make_offline_llm


def read_records(path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def post_canonical(client, scenario_id="spanish-fintech-startup-uses-9e165169"):
    return client.post(
        "/api/analyze",
        json={
            "scenario": {"id": scenario_id, "description": "A Spanish fintech startup that uses machine learning to assess creditworthiness for consumer loans. The platform automatically approves or denies applications based on applicant data including income, employment history, and spending patterns. The company operates only in Spain and plans to expand to other EU markets."},
            "question": "What regulations apply?",
        },
    )


def test_each_demo_request_appends_one_success_record(query_log_path):
    """The canonical demo request lands exactly one record carrying the
    behaviour basics: status, mode, workflow, latency, and the retrieved
    provision count — and no token usage, because no LLM ran."""
    with TestClient(app) as client:
        resp = post_canonical(client)
    assert resp.status_code == 200
    data = resp.json()

    records = read_records(query_log_path)
    assert len(records) == 1, "exactly one record per request"
    record = records[0]
    assert record["status"] == "success"
    assert record["mode"] == "demo"
    assert record["scenario_id"] == "spanish-fintech-startup-uses-9e165169"
    assert record["workflow"] == data["trace"]["workflow"]
    # Demo "retrieval" resolves provisions via corpus lookups; the count
    # mirrors the Citations those lookups produced.
    assert record["retrieved_chunks"] == len(data["answer"]["citations"])
    assert record["retrieved_chunks"] > 0
    assert record["latency_ms"] >= 0
    datetime.fromisoformat(record["timestamp"])
    assert record["tokens"] is None
    assert record["error"] is None


def test_noop_scenario_logs_zero_retrieved_chunks(query_log_path):
    """A scenario the demo cannot serve is still one honest record: zero
    Chunks, the noop workflow marker."""
    with TestClient(app) as client:
        resp = post_canonical(client, scenario_id="unknown-company")
    assert resp.status_code == 200
    assert resp.json()["trace"]["workflow"] == "noop"

    record = read_records(query_log_path)[0]
    assert record["status"] == "success"
    assert record["workflow"] == "noop"
    assert record["retrieved_chunks"] == 0
    assert record["tokens"] is None


def test_live_success_reports_mode_chunk_pool_and_aggregated_tokens(query_log_path, monkeypatch):
    """A served Live answer logs the Evidence-pool size and the token usage
    summed across every completed LLM call (planner + researcher + verifier +
    proposer + summarizer)."""
    per_call = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    llm = make_offline_llm(usage=per_call)
    with boot_live_with_fakes(monkeypatch, llm, FakeRetriever()) as live_client:
        resp = post_canonical(live_client, scenario_id="my-fintech-app")
    assert resp.status_code == 200
    data = resp.json()

    records = read_records(query_log_path)
    assert len(records) == 1
    record = records[0]
    assert record["status"] == "success"
    assert record["mode"] == "live"
    assert record["scenario_id"] == "my-fintech-app"
    assert record["workflow"] == LIVE_WORKFLOW_MARKER
    researcher = next(s for s in data["detailed_trace"] if s["step"] == "researcher")
    assert record["retrieved_chunks"] == len(researcher["retrieved"])
    # Five completed calls × the canned per-call usage.
    assert record["tokens"] == {"prompt": 500, "completion": 100, "total": 600}


class FailsAfterRetrievalLlm:
    """Plans fine, then dies at drafting — after the Researcher already
    gathered Evidence, mirroring a real mid-workflow outage."""

    def __init__(self):
        self.usage: list[dict] = [
            {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
        ]

    def complete(self, system, user, schema):
        from src.live_workflow import DraftClaims, Plan, ResearchTarget

        if schema is DraftClaims:
            raise LlmError("OpenRouter rejected the request (HTTP 500): upstream exploded")
        return Plan(targets=[ResearchTarget(query="creditworthiness")])


def test_failure_is_recorded_with_an_explicit_failure_status(query_log_path, monkeypatch):
    """A request that dies mid-workflow still appends its record: status
    'failure', the cause verbatim — plus what it had already retrieved and
    spent by then, because cost stays observable precisely when things go
    wrong."""
    with boot_live_with_fakes(
        monkeypatch, FailsAfterRetrievalLlm(), FakeRetriever(), raise_server_exceptions=False
    ) as live_client:
        resp = post_canonical(live_client)
    assert resp.status_code == 500

    records = read_records(query_log_path)
    assert len(records) == 1, "the failed request is not silently absent"
    record = records[0]
    assert record["status"] == "failure"
    assert "OpenRouter rejected" in record["error"]
    assert record["mode"] == "live"
    assert record["workflow"] is None
    assert record["latency_ms"] >= 0
    # The Chunks were in hand before the drafting call exploded.
    assert record["retrieved_chunks"] == 3
    assert record["tokens"] == {"prompt": 50, "completion": 10, "total": 60}


def test_not_available_response_logs_as_a_success_with_no_chunks(query_log_path, monkeypatch):
    """A Not-available reply is the system behaving correctly: a success
    record whose workflow marker shows it never reached the workflow."""
    with boot_live_with_fakes(monkeypatch, make_offline_llm(), FakeRetriever()) as live_client:
        monkeypatch.setattr(availability, "stored_chunk_count", lambda _database_url: 0)
        resp = post_canonical(live_client)
    assert resp.status_code == 200
    assert resp.json()["trace"]["workflow"] == availability.NOT_AVAILABLE_WORKFLOW

    record = read_records(query_log_path)[0]
    assert record["status"] == "success"
    assert record["mode"] == "live"
    assert record["workflow"] == availability.NOT_AVAILABLE_WORKFLOW
    assert record["retrieved_chunks"] == 0
    assert record["tokens"] is None


def test_requests_append_one_line_each_in_order(query_log_path):
    with TestClient(app) as client:
        post_canonical(client)
        post_canonical(client, scenario_id="other-company")

    records = read_records(query_log_path)
    assert [r["scenario_id"] for r in records] == ["spanish-fintech-startup-uses-9e165169", "other-company"]
    assert records[0]["timestamp"] <= records[1]["timestamp"]


# --- Pure module behaviour, tested directly -----------------------------------


def test_aggregate_token_usage_sums_provider_dicts_and_tolerates_missing_keys():
    llm = SimpleNamespace(
        usage=[
            {"prompt_tokens": 10},
            {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        ]
    )
    # Each call's total derives from its own parts when the provider omitted
    # it: 10 for the first, the reported 7 for the second.
    assert aggregate_token_usage(llm) == {"prompt": 15, "completion": 2, "total": 17}


def test_aggregate_token_usage_derives_total_when_the_provider_omits_it():
    llm = SimpleNamespace(usage=[{"prompt_tokens": 10, "completion_tokens": 4}])
    tokens = aggregate_token_usage(llm)
    assert tokens is not None
    assert tokens["total"] == 14


def test_aggregate_token_usage_returns_none_when_the_llm_reported_nothing():
    assert aggregate_token_usage(SimpleNamespace()) is None
    assert aggregate_token_usage(SimpleNamespace(usage=[])) is None
    assert aggregate_token_usage(object()) is None


def test_a_broken_log_destination_never_takes_serving_down(tmp_path, caplog):
    """Observability is best-effort by contract: an unwritable destination
    degrades to a logged warning instead of an OSError in the request path."""
    import logging

    target = tmp_path / "queries.jsonl"
    target.mkdir()  # a directory where the log file belongs: writes must fail

    with caplog.at_level(logging.WARNING, logger="src.query_log"):
        landed = append_query_record(target, {"status": "success"})

    assert landed is False
    assert any("query log" in r.getMessage().lower() for r in caplog.records)


def test_appended_lines_are_complete_json_objects_terminated_by_newlines(tmp_path):
    target = tmp_path / "nested" / "queries.jsonl"
    append_query_record(target, {"n": 1})
    append_query_record(target, {"n": 2})

    text = target.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert [json.loads(line) for line in text.splitlines()] == [{"n": 1}, {"n": 2}]
