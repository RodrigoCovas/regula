import sys
from pathlib import Path

# Ensure backend.src is importable when running tests from repo root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import psycopg2
import pytest
from fastapi.testclient import TestClient

from src import availability
from src.config import ConfigurationError
from src.llm import LlmError, LlmUnreachableError
from src.live_workflow import LIVE_WORKFLOW_MARKER
from src.main import REQUEST_ID_HEADER, app

from conftest import CANONICAL_SCENARIO_DESCRIPTION, CANONICAL_SCENARIO_ID, boot_live_with_fakes, boot_with_env, install_fake_pipeline, poll_progress
from fakes import FakeRetriever, make_offline_llm

client = TestClient(app)


def analyze(scenario_id, question, **scenario_extra):
    """POST /api/analyze with the given scenario/question and return the parsed response."""
    payload = {
        "scenario": {"id": scenario_id, **scenario_extra},
        "question": question,
    }
    resp = client.post("/api/analyze", json=payload)
    assert resp.status_code == 200
    return resp.json()


def test_analyze_bank_cloud_outage_demo():
    data = analyze(
        "bank-cloud-outage",
        "What regulatory obligations should the bank consider in relation to this incident, its reliance on the cloud provider, and its data protection obligations towards customer data on the disrupted systems?",
        description=CANONICAL_SCENARIO_DESCRIPTION,
    )
    assert "answer" in data
    assert "trace" in data
    assert "detailed_trace" in data
    answer = data["answer"]
    assert "findings" in answer
    # The demo serves the authored bank-case gold end-to-end: 13 Findings and
    # 29 Citations (issue #54).
    assert len(answer["findings"]) == 13
    assert len(answer["citations"]) == 29
    srcs = {c.get("source_id") for c in answer.get("citations", [])}
    assert "gdpr" in srcs
    assert "dora" in srcs
    for citation in answer.get("citations", []):
        targets = [citation.get(f) for f in ("article_number", "recital_number", "annex_number")]
        assert sum(t is not None for t in targets) == 1, "citation must target exactly one provision"
        assert citation.get("section")
        assert citation.get("provision")
        # Every demo target resolves in the corpus, so the quote is present
        # wherever the corpus supplies text — for this case, all 29.
        assert citation.get("quote")
    # The bank case cites articles only: no recital or annex targets.
    assert all(c.get("article_number") is not None for c in answer["citations"])
    # Each finding is tagged with an evidence strength
    for finding in answer.get("findings", []):
        assert finding.get("strength") in {"strong", "moderate", "weak"}
    trace_steps = data.get("detailed_trace", [])
    researcher_steps = [step for step in trace_steps if step.get("step") == "researcher"]
    assert researcher_steps, "Expected a researcher step in detailed_trace"
    assert "tool_calls" in researcher_steps[0]
    assert len(researcher_steps[0]["tool_calls"]) >= 1
    # Known limitation: corpus is English-only — its own sibling field, never an Action
    actions_text = " ".join(answer.get("actions", [])).lower()
    assert "english-only" in " ".join(data["known_limitations"]).lower()
    assert "english-only" not in actions_text


def test_demo_actions_are_referral_voiced_and_the_prototype_boundary_is_a_limitation():
    """ADR-0004: demo Actions name only what a qualified professional can
    settle, and the research-prototype boundary lives in known_limitations —
    a Known limitation is never an Action."""
    data = analyze("bank-cloud-outage", "What regulations apply?")
    actions = data["answer"]["actions"]
    assert actions, "the demo sheet is not empty"
    assert all(a.startswith("Have a qualified") for a in actions), actions
    actions_text = " ".join(actions).lower()
    assert "research prototype" not in actions_text
    assert "not legal advice" not in actions_text
    limitations = " ".join(data["known_limitations"]).lower()
    assert "research prototype" in limitations
    assert "not legal advice" in limitations


def test_response_shape_has_siblings_not_nested():
    """API returns answer, trace, detailed_trace, known_limitations as siblings; trace is NOT nested inside answer."""
    data = analyze("bank-cloud-outage", "What regulations apply?")

    # Top-level siblings
    assert set(data.keys()) == {"answer", "trace", "detailed_trace", "known_limitations"}
    # Answer does NOT contain trace or detailed_trace
    assert "trace" not in data["answer"]
    assert "detailed_trace" not in data["answer"]


def test_findings_present_strength_first_with_badges():
    """The demo sheet presents the strong Findings first (the Answer's
    strength-first order) and badges every Finding with its Strength."""
    data = analyze("bank-cloud-outage", "What regulations apply?")
    findings = data["answer"]["findings"]

    order = {"strong": 0, "moderate": 1, "weak": 2}
    ranks = [order[f["strength"]] for f in findings]
    assert ranks == sorted(ranks), "Findings present strength-first"
    strengths = {f["strength"] for f in findings}
    assert "strong" in strengths, "expected strong Findings in the bank-case gold"
    assert "moderate" in strengths, "expected derived/contingent Findings rated moderate"
    # The incident-management Finding leads the sheet and cites the DORA
    # incident-management Articles it names.
    assert findings[0]["strength"] == "strong"
    assert any(c.get("article_number") == 17 for c in findings[0].get("citations", [])), (
        "the leading Finding should cite DORA Article 17"
    )


def test_unsupported_claims_absent_from_answer_recorded_in_trace():
    """Unsupported claims are discarded from Answer but recorded in the trace."""
    data = analyze("bank-cloud-outage", "What regulations apply?")

    # Answer has no unsupported claims (no finding about automatic breach
    # status, customer compensation, provider responsibility, etc.)
    finding_statements = " ".join(f["statement"].lower() for f in data["answer"]["findings"])
    assert "automatically a personal data breach" not in finding_statements
    assert "compensate" not in finding_statements
    assert "the cloud provider, not the bank, is responsible" not in finding_statements

    # Trace records the discarded unsupported claims
    trace = data["trace"]
    assert "unsupported_claims_discarded" in trace
    discarded = trace["unsupported_claims_discarded"]
    assert isinstance(discarded, list)
    assert len(discarded) > 0
    # Check some known unsupported claims are listed
    assert any("automatically a personal data breach" in c.lower() for c in discarded)
    assert any("compensate" in c.lower() for c in discarded)
    assert any("the cloud provider, not the bank" in c.lower() for c in discarded)

    # The verifier step records a per-claim decision, and every discarded
    # claim appears there as rejected
    verifier_steps = [s for s in data["detailed_trace"] if s.get("step") == "verifier"]
    assert verifier_steps, "Expected a verifier step in detailed_trace"
    decisions = {d["claim"]: d["status"] for d in verifier_steps[0]["claim_decisions"]}
    assert len(decisions) >= len(discarded)
    for claim in discarded:
        assert decisions[claim] == "rejected"


def test_non_canonical_scenario_gets_helpful_response_not_keyword_routed():
    """Non-canonical scenarios get a helpful response, NOT silently routed to demo via keywords."""
    # This scenario has bank + cloud + outage keywords but wrong id
    data = analyze(
        "other-scenario",
        "What regulatory obligations apply after this cloud outage?",
        description="A bank's online banking is down after a major cloud outage at its provider",
    )

    # Should NOT get the demo findings (which would have 13 findings with citations)
    assert len(data["answer"]["findings"]) == 0
    assert len(data["answer"]["citations"]) == 0

    # Should get helpful actions explaining how to invoke the demo
    actions = data["answer"]["actions"]
    assert len(actions) >= 1
    assert any("bank-cloud-outage" in a for a in actions)
    assert any("demo" in a.lower() for a in actions)

    # Trace should indicate noop
    trace = data["trace"]
    assert trace.get("workflow") == "noop"
    assert "bank-cloud-outage" in trace.get("summary", "").lower()


def test_demo_not_available_response_hints_at_the_demo_button():
    """Issue #48: submitting a custom Scenario in Demo mode yields the
    Not-available response with a hint pointing at the demo button — the
    populate-the-form-only path a web user can act on immediately."""
    data = analyze("custom-scenario", "Does this loan scoring violate GDPR?")
    actions = data["answer"]["actions"]
    assert any("Try the demo scenario" in action for action in actions)


def post_with_request_id(request_id, scenario_id, question):
    """POST /api/analyze as the UI does: with the X-Request-Id header whose
    id the client then polls on the progress endpoint."""
    return client.post(
        "/api/analyze",
        headers={REQUEST_ID_HEADER: request_id},
        json={"scenario": {"id": scenario_id}, "question": question},
    )


def test_ui_demo_submission_stays_reachable_through_progress_polling():
    """The frontend's decoupled client (#41) treats the progress endpoint as
    the source of truth for the answer, so a UI-submitted Demo run — like a
    Live one — must register its served response under the submitted request
    id. Polling reaches the demo answer, never the unknown-request reply
    (issue #48: the Demo/Live toggle keeps the demo flow working)."""
    resp = post_with_request_id(
        "demo-poll-run",
        "bank-cloud-outage",
        "What regulations apply?",
    )
    assert resp.status_code == 200

    data = poll_progress(client, "demo-poll-run", lambda d: "answer" in d)
    assert data["answer"]["findings"], "polling must reach the demo answer"


def test_ui_demo_custom_scenario_polls_into_the_hint_response():
    """A UI-submitted custom Scenario in Demo mode polls into the helpful
    noop response — with the demo-button hint — not the unknown-request
    reply."""
    resp = post_with_request_id(
        "demo-poll-custom",
        "custom-scenario",
        "Does this loan scoring violate GDPR?",
    )
    assert resp.status_code == 200

    data = poll_progress(client, "demo-poll-custom", lambda d: "answer" in d)
    assert data["trace"]["workflow"] == "noop"
    assert any("Try the demo scenario" in a for a in data["answer"]["actions"])


def test_ui_demo_submission_failure_is_visible_through_polling(monkeypatch):
    """A Demo dispatch that dies records its failure under the submitted
    request id: polling reaches the terminal error, never a misleading
    unknown-request reply."""
    import src.main as main

    def raising_dispatch(_run):
        raise RuntimeError("demo dispatch failed")

    monkeypatch.setattr(main, "_dispatch_analyze", raising_dispatch)
    failing_client = TestClient(main.app, raise_server_exceptions=False)
    with failing_client as c:
        resp = c.post(
            "/api/analyze",
            headers={REQUEST_ID_HEADER: "demo-poll-failed"},
            json={"scenario": {"id": "bank-cloud-outage"}, "question": "What regulations apply?"},
        )
        assert resp.status_code == 500

    data = poll_progress(client, "demo-poll-failed", lambda d: d.get("error"))
    assert "demo dispatch failed" in data["error"]


def test_exact_scenario_id_required():
    """Only exact scenario.id == 'bank-cloud-outage' triggers the demo."""
    # Test with similar but different id (a look-alike of the canonical id)
    data = analyze("bank-cloud-outage-demo", "What regulations apply?")
    assert len(data["answer"]["findings"]) == 0

    # Test with exact match
    data = analyze("bank-cloud-outage", "What regulations apply?")
    assert len(data["answer"]["findings"]) == 13  # all 13 bank-case findings


def test_known_limitations_on_every_response():
    """known_limitations is a sibling on both the canonical and the not-available response."""
    canonical = analyze("bank-cloud-outage", "What regulations apply?")
    non_canonical = analyze("other-scenario", "¿Qué regulaciones aplican?")

    for data in (canonical, non_canonical):
        assert isinstance(data["known_limitations"], list)
        assert any("english-only" in line.lower() for line in data["known_limitations"])
        # Limitations never leak into Actions
        assert not any("english-only" in a.lower() for a in data["answer"]["actions"])


def test_corpus_english_only_limitation_surfaced():
    """The helpful response surfaces the English-only corpus limitation."""
    data = analyze("other", "¿Qué regulaciones aplican?")
    assert any("english" in line.lower() for line in data["known_limitations"])


def test_exactly_one_target_citation_invariant():
    """Every citation targets exactly one of article/recital/annex."""
    data = analyze("bank-cloud-outage", "What regulations apply?")
    for citation in data["answer"]["citations"]:
        targets = [
            citation.get("article_number") is not None,
            citation.get("recital_number") is not None,
            citation.get("annex_number") is not None,
        ]
        assert sum(targets) == 1, f"Citation must target exactly one provision: {citation}"


def test_demo_citations_carry_fixed_relevance_and_rated_strength():
    """Demo Answers ship fixed Provision relevance and the locked rated
    Citation strength consistent with the locked demo content (ADR-0011) —
    one entry per cited provision, keylessly (#47)."""
    data = analyze("bank-cloud-outage", "What regulations apply?")
    citations = data["answer"]["citations"]

    assert citations, "the demo answer cites provisions"
    assert all(c["relevance"] for c in citations), "every cited provision carries relevance"
    assert all(c["strength"] in {"strong", "moderate", "weak"} for c in citations)

    # One entry per cited provision: the demo's targets are distinct, so the
    # flat list matches the per-Finding citations one-for-one.
    assert len(citations) == sum(len(f["citations"]) for f in data["answer"]["findings"])

    # Spot-check the locked ratings where the operator's ground truth pins
    # them: the incident-management process is strong, the simplified
    # framework provision is the weak framing one, and GDPR's security
    # Article is a supporting duty.
    by_number = {(c["source_id"], c.get("article_number")): c for c in citations}
    assert by_number[("dora", 17)]["strength"] == "strong"
    assert by_number[("dora", 16)]["strength"] == "weak"
    assert by_number[("gdpr", 32)]["strength"] == "moderate"


def test_demo_detailed_trace_honestly_serves_the_locked_relevance():
    """The demo's detailed trace names where its Provision relevance came
    from: the locked demo content, served by the summarizer step (#47)."""
    data = analyze("bank-cloud-outage", "What regulations apply?")
    steps = [s["step"] for s in data["detailed_trace"]]
    assert steps == ["planner", "researcher", "verifier", "summarizer"]
    summarizer = data["detailed_trace"][-1]
    assert "locked Provision relevance" in summarizer["action"]


def boot_live(monkeypatch, **overrides):
    """Boot in Live mode with a throwaway key — the preamble every Live
    dispatch test shares."""
    env = {"REGULA_MODE": "live", "OPENROUTER_API_KEY": "sk-or-test"}
    env.update(overrides)
    return boot_with_env(monkeypatch, **env)


def post_canonical_scenario(client):
    """POST the canonical bank-cloud-outage Scenario — the request every
    availability test puts to the endpoint, in either mode."""
    return client.post(
        "/api/analyze",
        json={
            "scenario": {"id": CANONICAL_SCENARIO_ID, "description": CANONICAL_SCENARIO_DESCRIPTION},
            "question": "What regulations apply?",
        },
    )


def assert_not_available_shape(data):
    """The sibling contract every Not-available response shares: empty Answer,
    the not-available workflow marker, and no demo content anywhere."""
    assert set(data.keys()) == {"answer", "trace", "detailed_trace", "known_limitations"}
    assert data["answer"]["findings"] == []
    assert data["answer"]["citations"] == []
    assert data["trace"]["workflow"] == availability.NOT_AVAILABLE_WORKFLOW


def test_live_request_over_an_ingested_store_runs_the_real_pipeline(monkeypatch):
    """With the Corpus ingested, a Live request reaches the real workflow —
    it is answered by the pipeline, not a Not-available reply, and never by
    demo content."""
    with boot_live_with_fakes(monkeypatch, make_offline_llm(), FakeRetriever()) as live_client:
        resp = post_canonical_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {"answer", "trace", "detailed_trace", "known_limitations"}
    assert data["trace"]["workflow"] == LIVE_WORKFLOW_MARKER
    assert "demo" not in " ".join(data["answer"]["actions"]).lower()
    assert any("english-only" in line.lower() for line in data["known_limitations"])


def test_configuration_error_is_startup_failure_not_mid_request_server_error(monkeypatch):
    """Invalid mode value aborts startup; requests are never served a 500 from bad config."""
    client = boot_with_env(monkeypatch, REGULA_MODE="banana")
    with pytest.raises(ConfigurationError):
        with client:
            pass


def test_default_boot_serves_demo_mode(monkeypatch):
    """Default boot (no env vars) serves Demo mode through the real lifecycle."""
    with boot_with_env(monkeypatch) as demo_client:
        resp = demo_client.post(
            "/api/analyze",
            json={"scenario": {"id": "bank-cloud-outage"}, "question": "What regulations apply?"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["answer"]["findings"]) == 13
    assert data["trace"]["workflow"] != "not-available"


# --- Un-ingested guard: Live mode over an empty vector store (issue #16) ---


def patch_stored_chunk_count(monkeypatch, fake):
    """Install ``fake`` as the vector-store probe for the next probes."""
    monkeypatch.setattr(availability, "stored_chunk_count", fake)


def test_live_request_before_ingestion_names_the_exact_ingest_command(monkeypatch):
    """A Live request over an empty store gets a Not-available response naming
    the exact ingest command and how to retry — never demo content, never a
    server error."""
    with boot_live(monkeypatch) as live_client:
        patch_stored_chunk_count(monkeypatch, lambda _database_url: 0)
        resp = post_canonical_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()
    assert_not_available_shape(data)
    actions = data["answer"]["actions"]
    assert any(availability.INGEST_COMMAND in a for a in actions), actions
    assert any("re-run" in a.lower() for a in actions), actions
    # Recovery must not require reading documentation or restarting the process.
    assert any("restart" in a.lower() for a in actions), actions
    assert any("english-only" in line.lower() for line in data["known_limitations"])


def test_after_ingestion_the_same_request_passes_the_guard_without_restart(monkeypatch):
    """Ingesting after boot takes effect on the next request: the guard no
    longer intercepts it and the request is answered by the Live pipeline."""
    with boot_live_with_fakes(monkeypatch, make_offline_llm(), FakeRetriever()) as live_client:
        resp = post_canonical_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()
    # The un-ingested Not-available response would name the ingest command;
    # a guarded release runs the workflow instead.
    assert not any(availability.INGEST_COMMAND in a for a in data["answer"]["actions"])
    assert "empty" not in data["trace"]["summary"].lower()
    assert data["trace"]["workflow"] == LIVE_WORKFLOW_MARKER


def test_ingestion_landing_after_boot_takes_effect_on_the_next_request(monkeypatch):
    """The full AC-3 arc in one process: the store is empty at boot (the guard
    intercepts), ingestion lands afterwards, and the very next request passes
    the guard — no restart in between."""
    chunk_count = [0]  # mutable counter: the store's contents, as the probe sees it

    def count_as_ingestion_lands(_database_url):
        return chunk_count[0]

    patch_stored_chunk_count(monkeypatch, count_as_ingestion_lands)
    with boot_live(monkeypatch) as live_client:
        before = post_canonical_scenario(live_client).json()
        assert before["trace"]["workflow"] == availability.NOT_AVAILABLE_WORKFLOW
        assert any(availability.INGEST_COMMAND in a for a in before["answer"]["actions"])

        chunk_count[0] = 42  # ingest lands, same process
        install_fake_pipeline(make_offline_llm(), FakeRetriever())

        after = post_canonical_scenario(live_client).json()
    # The un-ingested Not-available response would name the ingest command; a
    # guarded release is answered by the Live workflow instead.
    assert not any(availability.INGEST_COMMAND in a for a in after["answer"]["actions"])
    assert "empty" not in after["trace"]["summary"].lower()
    assert after["trace"]["workflow"] == LIVE_WORKFLOW_MARKER


def test_live_request_with_unreachable_store_names_the_outage_never_the_ingest_command(monkeypatch):
    """An unreachable store at request time is deliberately NOT treated as the
    un-ingested case: it gets its own Not-available response naming the outage
    — re-running ingestion cannot fix a store the backend cannot reach, so the
    two conditions must stay distinguishable for operators."""
    def unreachable(_database_url):
        raise ConnectionError("connection refused")

    patch_stored_chunk_count(monkeypatch, unreachable)
    with boot_live(monkeypatch) as live_client:
        resp = post_canonical_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()
    assert_not_available_shape(data)
    actions = data["answer"]["actions"]
    assert any("not reachable" in a.lower() for a in actions), actions
    # Recovery is about reachability, not ingestion — and the summary must not
    # be mistaken for the un-ingested (empty store) response's.
    assert all(availability.INGEST_COMMAND not in a for a in actions), actions
    assert availability.INGEST_COMMAND not in resp.text
    assert "empty" not in data["trace"]["summary"].lower()
    assert "unreachable" in data["trace"]["summary"].lower()
    assert any("docker compose" in a for a in actions), actions
    assert any("re-run" in a.lower() for a in actions), actions
    assert any("english-only" in line.lower() for line in data["known_limitations"])


def test_live_request_with_a_pure_connection_outage_gets_the_unreachable_response(monkeypatch):
    """The issue's literal scenario: PostgreSQL down at request time raises
    psycopg2 OperationalError with no SQLSTATE (the server was never reached).
    It must get the unreachable-store reply, like any other outage."""
    def postgres_down(_database_url):
        raise psycopg2.OperationalError("connection refused")

    patch_stored_chunk_count(monkeypatch, postgres_down)
    with boot_live(monkeypatch) as live_client:
        resp = post_canonical_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()
    assert_not_available_shape(data)
    assert "unreachable" in data["trace"]["summary"].lower()
    assert availability.INGEST_COMMAND not in resp.text


def test_live_request_with_a_canceled_query_on_a_reachable_store_is_a_server_error(monkeypatch):
    """OperationalError subclasses with their own SQLSTATE mean the server
    answered — the store is reachable. A statement timeout (QueryCanceled,
    57014) is a real server error, not 'your database is down' advice."""
    def statement_timeout(_database_url):
        error = psycopg2.errors.QueryCanceled("statement timeout")
        error.pgcode = "57014"
        raise error

    patch_stored_chunk_count(monkeypatch, statement_timeout)
    with boot_live(monkeypatch, raise_server_exceptions=False) as live_client:
        resp = post_canonical_scenario(live_client)
    assert resp.status_code == 500
    # Neither outage nor ingestion advice: this needs a different fix.
    assert "not reachable" not in resp.text
    assert availability.INGEST_COMMAND not in resp.text


def test_live_request_with_a_broken_store_schema_is_still_a_server_error(monkeypatch):
    """Only connection-level failures count as unreachable: a reachable store
    with a broken schema is a real server error, not misleading recovery
    advice about the database being down."""
    def broken_schema(_database_url):
        raise psycopg2.ProgrammingError('relation "chunks" does not exist')

    patch_stored_chunk_count(monkeypatch, broken_schema)
    with boot_live(monkeypatch, raise_server_exceptions=False) as live_client:
        resp = post_canonical_scenario(live_client)
    assert resp.status_code == 500
    # Neither outage nor ingestion advice: this needs a different fix.
    assert "not reachable" not in resp.text
    assert availability.INGEST_COMMAND not in resp.text


# --- Unreachable LLM provider: the third Not-available condition (review of #17) ---


class UnreachableLlm:
    """The LLM seam when OpenRouter cannot be reached at all."""

    def complete(self, system, user, schema):
        raise LlmUnreachableError(
            "Could not reach OpenRouter at https://openrouter.ai/api/v1/chat/completions: "
            "connection refused. Check the network connection."
        )


def test_live_request_with_unreachable_llm_names_the_outage_not_a_server_error(monkeypatch):
    """OpenRouter down at request time is a genuine outage: the reply names it
    and how to recover — never a 500, never demo content, never ingest advice."""
    with boot_live(monkeypatch) as live_client:
        patch_stored_chunk_count(monkeypatch, lambda _database_url: 42)
        install_fake_pipeline(UnreachableLlm(), FakeRetriever())
        resp = post_canonical_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()
    assert_not_available_shape(data)
    actions = data["answer"]["actions"]
    assert any("not reachable" in a.lower() for a in actions), actions
    assert any("openrouter_api_key" in a.lower() for a in actions), actions
    assert any("re-run" in a.lower() for a in actions), actions
    assert "unreachable" in data["trace"]["summary"].lower()
    # Recovery is about the provider, not ingestion.
    assert availability.INGEST_COMMAND not in resp.text
    assert any("english-only" in line.lower() for line in data["known_limitations"])


def test_live_request_with_a_reachable_but_failing_llm_is_a_server_error(monkeypatch):
    """A reachable provider that rejects the request is not an outage: masking
    it as 'provider unreachable' would send operators chasing their network or
    key instead of the actual failure."""
    class RateLimitedLlm:
        def complete(self, system, user, schema):
            raise LlmError("OpenRouter rejected the request (HTTP 429): Rate limit exceeded")

    with boot_live(monkeypatch, raise_server_exceptions=False) as live_client:
        patch_stored_chunk_count(monkeypatch, lambda _database_url: 42)
        install_fake_pipeline(RateLimitedLlm(), FakeRetriever())
        resp = post_canonical_scenario(live_client)
    assert resp.status_code == 500
    assert "not reachable" not in resp.text


def test_boot_with_empty_store_succeeds_logging_exactly_one_warning(monkeypatch, caplog):
    """Booting before ingestion works: one warning naming the ingest command,
    nothing more — boot never depends on an ingested Corpus."""
    import logging

    with caplog.at_level(logging.WARNING, logger="src.availability"):
        with boot_with_env(monkeypatch):
            pass
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    assert availability.INGEST_COMMAND in warnings[0].getMessage()


def test_boot_succeeds_even_when_the_store_cannot_be_reached(monkeypatch, caplog):
    """An unreachable store degrades to one warning; the process still boots
    and Demo mode keeps serving."""
    import logging

    def unreachable(_database_url) -> int:
        raise ConnectionError("connection refused")

    patch_stored_chunk_count(monkeypatch, unreachable)
    with caplog.at_level(logging.WARNING, logger="src.availability"):
        with boot_with_env(monkeypatch) as demo_client:
            resp = post_canonical_scenario(demo_client)
    assert resp.status_code == 200
    assert len(resp.json()["answer"]["findings"]) == 13
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]


def test_demo_mode_serves_despite_an_empty_store(monkeypatch):
    """Store state never gates Demo mode: an empty vector store still serves
    the canonical scenario in full."""
    with boot_with_env(monkeypatch) as demo_client:
        patch_stored_chunk_count(monkeypatch, lambda _database_url: 0)
        resp = post_canonical_scenario(demo_client)
    assert resp.status_code == 200
    assert len(resp.json()["answer"]["findings"]) == 13


def test_demo_mode_never_touches_the_store_even_when_it_cannot_be_reached(monkeypatch):
    """Demo mode is unaffected by store state in every case: the request path
    never probes the vector store at all."""
    calls = []

    def explosive_probe(_database_url) -> int:
        calls.append(1)
        raise AssertionError("Demo mode must never probe the vector store")

    with boot_with_env(monkeypatch) as demo_client:
        # Armed only after boot: startup may probe (and survive failure); the
        # Demo-mode request path must not.
        patch_stored_chunk_count(monkeypatch, explosive_probe)
        resp = post_canonical_scenario(demo_client)
    assert resp.status_code == 200
    assert len(resp.json()["answer"]["findings"]) == 13
    assert calls == []


# --- Per-run mode: analysis requests carry mode explicitly (ADR-0008, issue #44) ---


def post_mode(client, mode=None, request_id=None, scenario_id="bank-cloud-outage"):
    """POST a Scenario with an explicit ``mode`` (or none, to exercise the
    server default) and an optional progress request id. ``scenario_id``
    defaults to the canonical demo id; pass a derived id to pin down ADR-0005."""
    from src.main import REQUEST_ID_HEADER

    payload = {
        "scenario": {"id": scenario_id, "description": "A bank reliant on an external cloud provider."},
        "question": "What regulations apply?",
    }
    if mode is not None:
        payload["mode"] = mode
    headers = {REQUEST_ID_HEADER: request_id} if request_id else {}
    return client.post("/api/analyze", json=payload, headers=headers)


def assert_live_pipeline_answered(data):
    """The canned offline pipeline's Answer: two kept Findings, never demo
    content — proof the request ran Live mode with the provider fakes."""
    findings = data["answer"]["findings"]
    assert len(findings) == 2, findings
    assert findings[0]["statement"] == "Creditworthiness evaluation is a high-risk use case."
    assert data["trace"]["workflow"] == LIVE_WORKFLOW_MARKER


def assert_demo_answered(data):
    """The deterministic demo sheet: every canonical Finding, never pipeline content."""
    assert len(data["answer"]["findings"]) == 13
    assert all(
        f["statement"] != "Creditworthiness evaluation is a high-risk use case."
        for f in data["answer"]["findings"]
    )


def assert_missing_key_guidance(data):
    """The missing-key Not-available response names the exact variable and the
    documented key home — the recovery guidance ADR-0008 asks for."""
    actions = data["answer"]["actions"]
    assert any("OPENROUTER_API_KEY" in a for a in actions), actions
    assert any(".env" in a for a in actions), actions
    assert "key" in data["trace"]["summary"].lower()


def test_request_with_explicit_live_mode_overrides_a_demo_server_default(monkeypatch):
    """One backend booted in the default Demo mode serves Live mode when the
    request carries mode explicitly — mode is a per-run choice (ADR-0008)."""
    with boot_with_env(monkeypatch, OPENROUTER_API_KEY="sk-or-test") as client:
        patch_stored_chunk_count(monkeypatch, lambda _database_url: 42)
        install_fake_pipeline(make_offline_llm(), FakeRetriever())
        resp = post_mode(client, mode="live")
    assert resp.status_code == 200
    assert_live_pipeline_answered(resp.json())


def test_request_with_explicit_demo_mode_overrides_a_live_server_default(monkeypatch):
    """A backend whose server default is Live still serves the deterministic
    demo when the request carries mode='demo' — and the pipeline never runs."""
    with boot_live_with_fakes(monkeypatch, make_offline_llm(), FakeRetriever()) as client:
        resp = post_mode(client, mode="demo")
    assert resp.status_code == 200
    assert_demo_answered(resp.json())


def test_omitted_mode_falls_back_to_the_live_server_default(monkeypatch):
    """REGULA_MODE survives as the server-side default: a request that omits
    mode on a Live-default backend runs Live (the omitted-mode Demo default is
    covered by test_default_boot_serves_demo_mode)."""
    with boot_live_with_fakes(monkeypatch, make_offline_llm(), FakeRetriever()) as client:
        resp = post_mode(client)
    assert resp.status_code == 200
    assert_live_pipeline_answered(resp.json())


def test_invalid_explicit_mode_is_a_client_error_never_a_crash(monkeypatch):
    """An unknown mode value is rejected by request validation (422), never
    dispatched and never a 500."""
    with boot_with_env(monkeypatch) as client:
        resp = post_mode(client, mode="banana")
    assert resp.status_code == 422


def test_explicit_demo_mode_still_serves_only_the_exact_canonical_scenario(monkeypatch):
    """ADR-0005 unchanged under per-run mode: a request carrying mode='demo'
    with a derived id gets the not-available reply naming the canonical id —
    never demo content, never keyword routing."""
    with boot_with_env(monkeypatch) as client:
        resp = post_mode(client, mode="demo", scenario_id="bank-cloud-outage-demo")
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"]["findings"] == []
    assert data["answer"]["citations"] == []
    assert data["trace"]["workflow"] == "noop"
    actions = data["answer"]["actions"]
    assert any("bank-cloud-outage" in a for a in actions), actions


def test_keyless_boot_serves_demo_and_a_live_request_answers_not_available(monkeypatch):
    """ADR-0008: with default configuration and no provider key the backend
    boots and serves Demo; a Live request surfaces the missing key as a
    Readiness gap — the Not-available response with recovery guidance, never
    a boot refusal and never a crash."""
    with boot_with_env(monkeypatch) as client:
        patch_stored_chunk_count(monkeypatch, lambda _database_url: 42)
        install_fake_pipeline(make_offline_llm(), FakeRetriever())

        demo_resp = post_canonical_scenario(client)
        live_resp = post_mode(client, mode="live")

    assert demo_resp.status_code == 200
    assert_demo_answered(demo_resp.json())

    assert live_resp.status_code == 200
    data = live_resp.json()
    assert_not_available_shape(data)
    assert_missing_key_guidance(data)
    # The pipeline must never have run: the reply is not the faked Live Answer.
    assert data["answer"]["findings"] == []


def test_live_request_with_empty_string_key_answers_not_available(monkeypatch):
    """The compose path forwards OPENROUTER_API_KEY even when the operator's
    env file omits it — as an empty string. A Live request then surfaces the
    same missing-key Readiness gap as a truly unset key: the Not-available
    response with recovery guidance, never a provider error."""
    with boot_with_env(monkeypatch, OPENROUTER_API_KEY="") as client:
        patch_stored_chunk_count(monkeypatch, lambda _database_url: 42)
        install_fake_pipeline(make_offline_llm(), FakeRetriever())

        live_resp = post_mode(client, mode="live")

    assert live_resp.status_code == 200
    data = live_resp.json()
    assert_not_available_shape(data)
    assert_missing_key_guidance(data)
    # The pipeline must never have run: the reply is not the faked Live Answer.
    assert data["answer"]["findings"] == []


def test_keyless_live_ui_submission_polls_into_the_not_available_response(monkeypatch):
    """The decoupled UI path surfaces the same Readiness gap: a keyless Live
    submission registers in the progress registry and completes with the
    missing-key Not-available response for polling clients."""
    with boot_with_env(monkeypatch) as client:
        patch_stored_chunk_count(monkeypatch, lambda _database_url: 42)
        install_fake_pipeline(make_offline_llm(), FakeRetriever())
        resp = post_mode(client, mode="live", request_id="keyless-ui")
        assert resp.status_code == 200
        snapshot = resp.json()
        assert snapshot["request_id"] == "keyless-ui"
        assert "answer" not in snapshot, "POST should return a ProgressSnapshot"

        data = poll_progress(client, "keyless-ui", lambda d: "answer" in d)

    assert_not_available_shape(data)
    assert_missing_key_guidance(data)
