"""Live-mode pipeline over the HTTP seam (spec #9, ticket #17).

Every test boots the app in Live mode with an ingested store (the emptiness
probe is patched) and injects deterministic fakes for the two provider
protocols at the composition root: no network, no Ollama, no pgvector.
"""

import sys
from pathlib import Path

# Ensure backend.src is importable when running tests from repo root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import json

import pytest
from fastapi.testclient import TestClient

from src import availability
from src.live_workflow import LIVE_WORKFLOW_MARKER
from src.main import app
from src.models import Strength
from src.retrieval import MAX_RETRIEVED_CHUNKS

from conftest import install_fake_pipeline
from fakes import (
    AUTOMATED_DECISION_CHUNK,
    DEFINITIONS_CHUNK,
    HIGH_RISK_CHUNK,
    FakeEmbedder,
    FakeRetriever,
    grounded_verdict,
    make_chunk,
    make_offline_llm,
)


@pytest.fixture(autouse=True)
def _offline_live_dependencies(monkeypatch):
    """Boot Live mode offline: ingested store probe, fakes at the composition root."""
    monkeypatch.setenv("REGULA_MODE", "live")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setattr(availability, "stored_chunk_count", lambda _database_url: 42)
    install_fake_pipeline(make_offline_llm(), FakeRetriever())
    yield


@pytest.fixture()
def live_client():
    with TestClient(app) as client:
        yield client


def post_arbitrary_scenario(client, scenario_id="my-fintech-app", question="Does automated loan scoring trigger high-risk obligations?"):
    return client.post(
        "/api/analyze",
        json={
            "scenario": {"id": scenario_id, "description": "A Spanish fintech evaluating loan applications automatically"},
            "question": question,
        },
    )


def test_arbitrary_scenario_returns_evidence_backed_findings_with_metadata_derived_citations(live_client):
    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    findings = data["answer"]["findings"]
    assert len(findings) >= 1
    for finding in findings:
        assert finding["strength"] in {"strong", "moderate", "weak"}
        assert finding["citations"], "every Finding is evidence-backed"
        for citation in finding["citations"]:
            targets = [citation[f] for f in ("article_number", "recital_number", "annex_number")]
            assert sum(t is not None for t in targets) == 1
            # Derived from Chunk metadata, never inferred from text:
            assert citation["source_id"] in {HIGH_RISK_CHUNK.source_id, AUTOMATED_DECISION_CHUNK.source_id, DEFINITIONS_CHUNK.source_id}

    # The strong claim's citation carries the canned Chunk's provision number.
    strong = next(f for f in findings if f["strength"] == "strong")
    assert [c["source_id"] for c in strong["citations"]] == ["ai-act"]
    assert strong["citations"][0]["article_number"] == HIGH_RISK_CHUNK.article_number

    # The answer's flat citation list mirrors the per-finding citations.
    assert len(data["answer"]["citations"]) == sum(len(f["citations"]) for f in findings)

    # The workflow ran — this is not a Not-available response.
    assert data["trace"]["workflow"] == LIVE_WORKFLOW_MARKER
    assert any("english-only" in line.lower() for line in data["known_limitations"])


def test_unsupported_claims_absent_from_answer_recorded_in_trace(live_client):
    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    statements = " ".join(f["statement"].lower() for f in data["answer"]["findings"])
    assert "special-category" not in statements

    discarded = data["trace"]["unsupported_claims_discarded"]
    assert any("special-category" in c.lower() for c in discarded)

    verifier_steps = [s for s in data["detailed_trace"] if s["step"] == "verifier"]
    decisions = {d["claim"]: d for d in verifier_steps[0]["claim_decisions"]}
    rejected_claim = next(d for claim, d in decisions.items() if "special-category" in claim.lower())
    assert rejected_claim["status"] == "rejected"
    assert "evidence" in rejected_claim["reason"].lower()


def test_weak_framing_finding_stays_in_the_answer(live_client):
    resp = post_arbitrary_scenario(live_client)
    data = resp.json()

    weak = [f for f in data["answer"]["findings"] if f["strength"] == "weak"]
    assert weak, "weak framing Findings stay in the Answer"
    framing = next(f for f in weak if "qualifies as an ai system" in f["statement"].lower())
    assert any(c["article_number"] == DEFINITIONS_CHUNK.article_number for c in framing["citations"])


def test_empty_retrieval_is_insufficient_evidence_never_invented_findings(monkeypatch):
    """Nothing relevant retrieved → honest Insufficient-evidence response:
    empty Findings, a Known limitation naming the gap, Actions to narrow."""
    install_fake_pipeline(make_offline_llm(), FakeRetriever(chunks=[]))
    with TestClient(app) as client:
        resp = post_arbitrary_scenario(client)
    assert resp.status_code == 200
    data = resp.json()

    assert data["answer"]["findings"] == []
    assert data["answer"]["citations"] == []
    assert any("nothing relevant" in line.lower() for line in data["known_limitations"])
    actions = data["answer"]["actions"]
    assert len(actions) >= 1
    assert any("rephrase" in a.lower() for a in actions)
    # The workflow still ran its gates — this is not a Not-available reply.
    assert data["trace"]["workflow"] == LIVE_WORKFLOW_MARKER
    # No claims were drafted or verified against no Evidence.
    assert data["trace"]["unsupported_claims_discarded"] == []
    assert "nothing" in data["trace"]["summary"].lower()


def test_researcher_touches_the_corpus_only_via_the_retrieval_tool(live_client):
    retriever = FakeRetriever()
    llm = make_offline_llm()
    install_fake_pipeline(llm, retriever)

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200

    # Every planned query went through the retrieval tool, and nothing else.
    planned = ["creditworthiness", "automated decisions"]
    assert retriever.queries == planned
    researcher_steps = [s for s in resp.json()["detailed_trace"] if s["step"] == "researcher"]
    tool_calls = researcher_steps[0]["tool_calls"]
    assert [call["tool"] for call in tool_calls] == ["retrieve_chunks"] * len(planned)
    assert [call["input"]["query"] for call in tool_calls] == planned
    assert all(call["chunks_returned"] == 3 for call in tool_calls)
    # Each LLM call is visible in the scripted client's log.
    assert len(llm.calls) == 3
    # The Verifier receives the drafted claims as JSON, not a Python repr.
    verifier_user = llm.calls[2][1]
    claims_segment = verifier_user.split("Drafted claims:\n")[1]
    assert json.loads(claims_segment)[0]["statement"].startswith("Creditworthiness")


def test_drafted_claim_without_a_verdict_is_still_recorded_as_rejected(monkeypatch):
    """A drafted claim the Verifier never returned a verdict for may not
    vanish silently: it is recorded in the Execution trace as rejected."""
    from src.live_workflow import DraftClaim, DraftClaims, Plan, ResearchTarget, Verdicts

    llm = make_offline_llm()
    llm.claims = DraftClaims(
        claims=[
            DraftClaim(statement="A claim that will get no verdict.", evidence_refs=["E1"]),
            DraftClaim(statement="The system qualifies as an AI system under the definitions.", evidence_refs=["E3"]),
        ]
    )
    llm.plan = Plan(targets=[ResearchTarget(query="creditworthiness")])
    install_fake_pipeline(llm, FakeRetriever())
    with TestClient(app) as client:
        resp = post_arbitrary_scenario(client)
    assert resp.status_code == 200
    data = resp.json()

    statements = [f["statement"] for f in data["answer"]["findings"]]
    assert "A claim that will get no verdict." not in statements

    discarded = data["trace"]["unsupported_claims_discarded"]
    assert "A claim that will get no verdict." in discarded

    verifier_steps = [s for s in data["detailed_trace"] if s["step"] == "verifier"]
    decision = next(d for d in verifier_steps[0]["claim_decisions"] if d["claim"] == "A claim that will get no verdict.")
    assert decision["status"] == "rejected"
    assert "no decision" in decision["reason"]

    # The verdicted weak framing finding is unaffected.
    assert any(f["strength"] == "weak" for f in data["answer"]["findings"])


# --- Fair-share fill: no research target may starve under the locked budget ---


def test_fair_share_fill_keeps_every_target_represented_under_the_budget(live_client):
    """A greedy first target cannot consume the whole single-pass pool: each
    target claims its fair share before leftovers backfill the budget."""
    broad = [make_chunk(source_id=f"broad-{i}", number=i + 1) for i in range(MAX_RETRIEVED_CHUNKS)]
    narrow = [
        make_chunk(source_id="narrow-a", number=21),
        make_chunk(source_id="narrow-b", number=22),
    ]
    retriever = FakeRetriever(per_query={
        "creditworthiness": broad,
        "automated decisions": narrow,
    })
    install_fake_pipeline(make_offline_llm(), retriever)

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    researcher_steps = [s for s in data["detailed_trace"] if s["step"] == "researcher"]
    retrieved = researcher_steps[0]["retrieved"]
    sources = {r["source_id"] for r in retrieved}
    assert len(retrieved) <= MAX_RETRIEVED_CHUNKS, "the locked single-pass budget holds"
    assert any(s.startswith("narrow-") for s in sources), "the second target is represented"
    assert any(s.startswith("broad-") for s in sources), "the first target is still represented"


def test_duplicate_drafted_statements_each_get_their_own_decision(monkeypatch):
    """Verdicts are matched to drafts one-for-one per statement: two drafts of
    the same statement sharing one verdict means one kept Finding and one
    recorded rejection — neither copy may vanish from the trace."""
    from src.live_workflow import DraftClaim, DraftClaims, Plan, ResearchTarget, Verdicts

    duplicated = "The system qualifies as an AI system under the definitions."
    llm = make_offline_llm()
    llm.plan = Plan(targets=[ResearchTarget(query="definitions")])
    llm.claims = DraftClaims(
        claims=[
            DraftClaim(statement=duplicated, evidence_refs=["E1"]),
            DraftClaim(statement=duplicated, evidence_refs=["E1"]),
        ]
    )
    llm.verdicts = Verdicts(verdicts=[grounded_verdict(duplicated, Strength.weak, ["E1"])])
    install_fake_pipeline(llm, FakeRetriever(per_query={"definitions": [DEFINITIONS_CHUNK]}))
    with TestClient(app) as client:
        resp = post_arbitrary_scenario(client)
    assert resp.status_code == 200
    data = resp.json()

    findings = [f for f in data["answer"]["findings"] if f["statement"] == duplicated]
    assert len(findings) == 1, "exactly one Finding survives — one verdict was returned"

    verifier_steps = [s for s in data["detailed_trace"] if s["step"] == "verifier"]
    decisions = [d for d in verifier_steps[0]["claim_decisions"] if d["claim"] == duplicated]
    statuses = sorted(d["status"] for d in decisions)
    assert statuses == ["kept", "rejected"], statuses
    rejected = next(d for d in decisions if d["status"] == "rejected")
    assert "no decision" in rejected["reason"]

    discarded = data["trace"]["unsupported_claims_discarded"]
    assert discarded.count(duplicated) == 1


def test_unknown_evidence_labels_are_dropped_never_become_citations(monkeypatch):
    """A supported claim citing a label that matches no Chunk keeps only its
    resolvable Citations — an unknown label can never be hallucinated into
    a Citation."""
    from src.live_workflow import DraftClaim, DraftClaims, Plan, ResearchTarget, Verdicts

    statement = "Partially grounded claim."
    llm = make_offline_llm()
    llm.plan = Plan(targets=[ResearchTarget(query="creditworthiness")])
    llm.claims = DraftClaims(claims=[DraftClaim(statement=statement, evidence_refs=["E1", "E99"])])
    llm.verdicts = Verdicts(verdicts=[grounded_verdict(statement, Strength.moderate, ["E1", "E99"])])
    install_fake_pipeline(llm, FakeRetriever())
    with TestClient(app) as client:
        resp = post_arbitrary_scenario(client)
    assert resp.status_code == 200
    data = resp.json()

    findings = [f for f in data["answer"]["findings"] if f["statement"] == statement]
    assert len(findings) == 1, "the resolvable reference still supports the Finding"
    citations = findings[0]["citations"]
    assert [c["source_id"] for c in citations] == [HIGH_RISK_CHUNK.source_id]
    assert all(c["article_number"] != 99 for c in citations)

    verifier_steps = [s for s in data["detailed_trace"] if s["step"] == "verifier"]
    decision = next(d for d in verifier_steps[0]["claim_decisions"] if d["claim"] == statement)
    assert decision["status"] == "kept"


# --- Insufficient evidence over a junk-only Corpus (ticket #18) ----------------


class JunkOnlyStore:
    """An adversarial SearchStore: whatever distance bound it is given, it
    returns only junk hits — a Corpus holding nothing relevant to the query."""

    def __init__(self, hits):
        self.hits = hits

    def search_chunks(self, query_embedding, limit, max_distance):
        return list(self.hits)


def test_junk_only_corpus_answers_insufficient_evidence_never_fabricated_findings():
    """Every stored Chunk sits beyond the relevance threshold: the served
    answer says so honestly — zero Findings, a Known limitation naming the
    gap, narrowing Actions — even though the scripted LLM stands ready to
    fabricate grounded-looking claims if it were ever asked to draft."""
    from src.models import ScoredChunk
    from src.retrieval import VectorRetriever

    junk = [
        ScoredChunk(chunk=make_chunk(source_id=f"junk-{i}", number=i + 1), distance=0.9)
        for i in range(3)
    ]
    retriever = VectorRetriever(store=JunkOnlyStore(junk), embedder=FakeEmbedder())
    llm = make_offline_llm()
    install_fake_pipeline(llm, retriever)

    with TestClient(app) as client:
        resp = post_arbitrary_scenario(client)
    assert resp.status_code == 200
    data = resp.json()

    assert data["answer"]["findings"] == [], "a fabricated Finding must be impossible"
    assert data["answer"]["citations"] == []
    assert any("nothing relevant" in line.lower() for line in data["known_limitations"])
    actions = data["answer"]["actions"]
    assert any("rephrase" in a.lower() for a in actions), actions
    assert any("corpus" in a.lower() for a in actions), actions
    # The Live workflow ran and answered honestly — not a Not-available reply.
    assert data["trace"]["workflow"] == LIVE_WORKFLOW_MARKER
    assert "relevant enough" in data["trace"]["summary"].lower()
    # Nothing was drafted over junk Evidence, so nothing was discarded either.
    assert data["trace"]["unsupported_claims_discarded"] == []
    from src.live_workflow import DraftClaims

    drafting_calls = [call for call in llm.calls if call[2] is DraftClaims]
    assert drafting_calls == [], "no Claim may be drafted over junk-only Evidence"
