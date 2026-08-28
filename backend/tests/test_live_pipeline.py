"""Live-mode pipeline over the HTTP seam (spec #9, tickets #17 and #18).

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
from src.live_workflow import (
    LIVE_WORKFLOW_MARKER,
    NARROW_THE_QUESTION_ACTIONS,
    SEATS_PER_TARGET,
    SEEK_COUNSEL_ACTION,
    ActionProposal,
    ActionProposals,
    DraftClaims,
)
from src.main import app
from src.models import Strength
from src.retrieval import PER_TARGET_DEPTH, VectorRetriever

from conftest import install_fake_pipeline
from fakes import (
    AUTOMATED_DECISION_CHUNK,
    DEFINITIONS_CHUNK,
    HIGH_RISK_CHUNK,
    FakeEmbedder,
    FakeRetriever,
    FakeSearchStore,
    chunk_hit,
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


def assert_insufficient_evidence_response(data):
    """The shared shape of an honest Insufficient-evidence answer: zero
    Findings and Citations, a Known limitation naming the gap, narrowing
    Actions — served by the live workflow itself, never demo content."""
    assert data["answer"]["findings"] == []
    assert data["answer"]["citations"] == []
    assert any("nothing relevant" in line.lower() for line in data["known_limitations"])
    actions = data["answer"]["actions"]
    assert any("rephrase" in a.lower() for a in actions), actions
    assert any("corpus" in a.lower() for a in actions), actions
    assert data["trace"]["workflow"] == LIVE_WORKFLOW_MARKER


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

    assert_insufficient_evidence_response(data)
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
    assert len(llm.calls) == 4
    # The Verifier receives the drafted claims as JSON, not a Python repr.
    verifier_user = llm.calls[2][1]
    claims_segment = verifier_user.split("Drafted claims:\n")[1]
    assert json.loads(claims_segment)[0]["statement"].startswith("Creditworthiness")
    # The Proposer receives the kept Findings with their labelled Citations.
    proposer_user = llm.calls[3][1]
    assert "Kept Findings with their supporting Citations" in proposer_user
    assert "[C1]" in proposer_user


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


# --- Fair-share fill: no research target may starve under the plan-derived pool ---


def served_evidence(data) -> list[dict]:
    """The Evidence pool as served to the agents, from the researcher's trace."""
    researcher_steps = [s for s in data["detailed_trace"] if s["step"] == "researcher"]
    return researcher_steps[0]["retrieved"]


def run_plan(live_client, targets, per_query):
    """Serve a scenario under a plan with the given Research targets."""
    from src.live_workflow import Plan, ResearchTarget

    llm = make_offline_llm()
    llm.plan = Plan(targets=[ResearchTarget(query=target) for target in targets])
    install_fake_pipeline(llm, FakeRetriever(per_query=per_query))
    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    return resp.json()


def depth_results(target):
    """One full-depth retrieval for a target: PER_TARGET_DEPTH unique Chunks."""
    return [make_chunk(source_id=f"{target}-{i}", number=i + 1) for i in range(PER_TARGET_DEPTH)]


def test_fair_share_fill_keeps_every_target_represented_under_the_derived_pool(live_client):
    """A greedy first target cannot consume the whole Evidence pool: each
    target claims its fair share before leftovers backfill the rest — and
    the pool itself derives from the plan (#23), so two targets seat more
    Chunks than any single retrieval could return."""
    targets = ["creditworthiness", "automated decisions"]
    broad = depth_results("broad")
    narrow = [
        make_chunk(source_id="narrow-a", number=21),
        make_chunk(source_id="narrow-b", number=22),
    ]
    data = run_plan(
        live_client,
        targets,
        {"creditworthiness": broad, "automated decisions": narrow},
    )

    retrieved = served_evidence(data)
    sources = {r["source_id"] for r in retrieved}
    # Every fresh Chunk fits under the plan-derived cap (len(targets) targets ×
    # SEATS_PER_TARGET seats); the thin second target simply leaves seats open.
    assert len(retrieved) == len(broad) + len(narrow)
    assert len(retrieved) > PER_TARGET_DEPTH, "the widened pool out-seats one retrieval's depth — #23's fix"
    assert len(retrieved) <= SEATS_PER_TARGET * len(targets), "the plan-derived cap holds"
    assert any(s.startswith("narrow-") for s in sources), "the second target is represented"
    assert any(s.startswith("broad-") for s in sources), "the first target is still represented"


def test_evidence_pool_scales_with_the_planned_target_count(live_client):
    """Seat demand tracks the Planner's decomposition (ADR-0003): a third
    research target widens the pool again, and every target keeps its share
    of seats."""
    targets = ["creditworthiness", "automated decisions", "deployer obligations"]
    data = run_plan(live_client, targets, {target: depth_results(target) for target in targets})

    retrieved = served_evidence(data)
    sources = {r["source_id"] for r in retrieved}
    assert len(retrieved) == SEATS_PER_TARGET * len(targets)
    for target in targets:
        assert any(s.startswith(f"{target}-") for s in sources), f"{target} keeps representation"


def test_pool_derives_from_the_plan_not_from_retrieval_volume(live_client):
    """A lone broad target retrieves ``PER_TARGET_DEPTH`` Chunks but seats
    only ``SEATS_PER_TARGET`` of them: the pool is the plan's seat count —
    extra retrieval volume alone never widens what the agents reason over."""
    data = run_plan(live_client, ["creditworthiness"], {"creditworthiness": depth_results("broad")})

    retrieved = served_evidence(data)
    assert len(retrieved) == SEATS_PER_TARGET


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


def test_junk_only_corpus_answers_insufficient_evidence_never_fabricated_findings():
    """Every stored Chunk sits beyond the relevance threshold: the served
    answer says so honestly — zero Findings, a Known limitation naming the
    gap, narrowing Actions — even though the scripted LLM stands ready to
    fabricate grounded-looking claims if it were ever asked to draft."""
    junk = [chunk_hit(source_id=f"junk-{i}", number=i + 1, distance=0.9) for i in range(3)]
    # The shared FakeSearchStore replays its hits whatever bound it is given,
    # so this models a Corpus holding nothing relevant to the query.
    retriever = VectorRetriever(store=FakeSearchStore(hits=junk), embedder=FakeEmbedder())
    llm = make_offline_llm()
    install_fake_pipeline(llm, retriever)

    with TestClient(app) as client:
        resp = post_arbitrary_scenario(client)
    assert resp.status_code == 200
    data = resp.json()

    assert_insufficient_evidence_response(data)
    assert "relevant enough" in data["trace"]["summary"].lower()
    # Nothing was drafted over junk Evidence, so nothing was discarded either.
    assert data["trace"]["unsupported_claims_discarded"] == []
    drafting_calls = [call for call in llm.calls if call[2] is DraftClaims]
    assert drafting_calls == [], "no Claim may be drafted over junk-only Evidence"


# --- Proposer: referral Actions grounded in kept Findings (ticket #24) --------


def proposer_step(data) -> dict:
    """The Proposer's detailed-trace step."""
    return [s for s in data["detailed_trace"] if s["step"] == "proposer"][0]


def test_happy_path_serves_grounded_referral_actions_with_the_hand_off_last(live_client):
    """The Live happy path under-delivers no more: a validated grounded
    Action reaches the Answer, referral-voiced, with the standing seek-counsel
    hand-off appended last — the sheet can no longer be empty (#24)."""
    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    actions = data["answer"]["actions"]
    assert actions, "the happy path never returns an empty action sheet"
    assert actions[-1] == SEEK_COUNSEL_ACTION
    assert actions[0] == "Have a qualified professional verify that the company's credit evaluation duties match the high-risk provisions cited."
    assert "qualified" in actions[0].lower(), "Actions are referral-voiced"

    decisions = proposer_step(data)["action_decisions"]
    kept = [d for d in decisions if d["status"] == "kept"]
    assert [d["action"] for d in kept] == [actions[0]]


def test_ungrounded_proposals_are_dropped_and_recorded_in_the_detailed_trace(live_client):
    """A proposal whose citations resolve to no kept Finding never reaches the
    Answer and never vanishes silently: it is recorded as rejected with why
    (mirrors the ClaimDecision pattern)."""
    llm = make_offline_llm()
    llm.proposals = ActionProposals(
        proposals=[
            ActionProposal(action="Verify the DORA applicability conclusion.", kind="verify_against_facts", citation_refs=["C1"]),
            ActionProposal(action="An action grounded on nothing.", kind="contingency", citation_refs=[]),
            ActionProposal(
                action="An action grounded on a label matching no kept Finding.",
                kind="contingency",
                citation_refs=["C99"],
            ),
        ]
    )
    install_fake_pipeline(llm, FakeRetriever())

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    actions = data["answer"]["actions"]
    assert actions[0] == "Verify the DORA applicability conclusion."
    assert "An action grounded on nothing." not in actions
    assert "An action grounded on a label matching no kept Finding." not in actions

    decisions = {d["action"]: d for d in proposer_step(data)["action_decisions"]}
    for rejected_text in ("An action grounded on nothing.", "An action grounded on a label matching no kept Finding."):
        decision = decisions[rejected_text]
        assert decision["status"] == "rejected"
        assert "resolve to no kept Finding" in decision["reason"]
    assert decisions["Verify the DORA applicability conclusion."]["status"] == "kept"


def test_an_action_anchored_only_by_weak_findings_is_rejected(live_client):
    """Weak Findings may enrich an anchored Action but never support one
    alone: a proposal grounding only on the weak definitions Finding's
    Citation is rejected, leaving the hand-off alone (ADR-0004)."""
    llm = make_offline_llm()
    llm.proposals = ActionProposals(
        proposals=[
            ActionProposal(action="Confirm the AI-system definition applies.", kind="contingency", citation_refs=["C2"]),
        ]
    )
    install_fake_pipeline(llm, FakeRetriever())

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    assert data["answer"]["actions"] == [SEEK_COUNSEL_ACTION]
    decisions = proposer_step(data)["action_decisions"]
    assert decisions[0]["status"] == "rejected"
    assert "weak" in decisions[0]["reason"]


def test_weak_only_findings_never_yield_standalone_actions(monkeypatch):
    """With only a weak Finding kept, even a proposal grounded on its
    Citation is rejected: the hand-off alone is served, never a standalone
    Action built on framing Evidence."""
    from src.live_workflow import DraftClaim, Plan, ResearchTarget, Verdicts

    statement = "The system qualifies as an AI system under the definitions."
    llm = make_offline_llm()
    llm.plan = Plan(targets=[ResearchTarget(query="definitions")])
    llm.claims = DraftClaims(claims=[DraftClaim(statement=statement, evidence_refs=["E1"])])
    llm.verdicts = Verdicts(verdicts=[grounded_verdict(statement, Strength.weak, ["E1"])])
    llm.proposals = ActionProposals(
        proposals=[ActionProposal(action="Act on the definition.", kind="contingency", citation_refs=["C1"])]
    )
    install_fake_pipeline(llm, FakeRetriever(per_query={"definitions": [DEFINITIONS_CHUNK]}))

    with TestClient(app) as client:
        resp = post_arbitrary_scenario(client)
    assert resp.status_code == 200
    data = resp.json()

    assert data["answer"]["actions"] == [SEEK_COUNSEL_ACTION]
    decisions = proposer_step(data)["action_decisions"]
    assert decisions[0]["status"] == "rejected"
    assert "weak" in decisions[0]["reason"]


def test_at_most_five_validated_actions_plus_the_hand_off_are_served(live_client):
    """The sheet is capped: more validated proposals than the budget are
    recorded as rejected, and exactly five Actions plus the hand-off reach
    the Answer (ADR-0004)."""
    llm = make_offline_llm()
    llm.proposals = ActionProposals(
        proposals=[
            ActionProposal(action=f"Referral action {index}.", kind="verify_against_facts", citation_refs=["C1"])
            for index in range(7)
        ]
    )
    install_fake_pipeline(llm, FakeRetriever())

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    actions = data["answer"]["actions"]
    assert len(actions) == 6, actions
    assert actions[5] == SEEK_COUNSEL_ACTION
    assert actions[:5] == [f"Referral action {index}." for index in range(5)]

    decisions = proposer_step(data)["action_decisions"]
    statuses = [d["status"] for d in decisions]
    assert statuses == ["kept"] * 5 + ["rejected"] * 2, statuses
    assert "at most 5" in decisions[5]["reason"]


def test_kind_anchor_mismatch_is_rejected_by_the_gate(monkeypatch):
    """The referral mapping is application code, not just a prompt (AC 2): a
    verify-kind proposal anchored only by a moderate Finding is rejected with
    why, while the matching contingency kind passes the same anchors."""
    from src.live_workflow import DraftClaim, DraftClaims, Plan, ResearchTarget, Verdicts

    moderate = "A moderate finding."
    llm = make_offline_llm()
    llm.plan = Plan(targets=[ResearchTarget(query="definitions")])
    llm.claims = DraftClaims(claims=[DraftClaim(statement=moderate, evidence_refs=["E1"])])
    llm.verdicts = Verdicts(verdicts=[grounded_verdict(moderate, Strength.moderate, ["E1"])])
    llm.proposals = ActionProposals(
        proposals=[
            ActionProposal(action="Verify the moderate finding.", kind="verify_against_facts", citation_refs=["C1"]),
            ActionProposal(action="Check the contingency.", kind="contingency", citation_refs=["C1"]),
        ]
    )
    install_fake_pipeline(llm, FakeRetriever(per_query={"definitions": [DEFINITIONS_CHUNK]}))

    with TestClient(app) as client:
        resp = post_arbitrary_scenario(client)
    assert resp.status_code == 200
    data = resp.json()

    assert data["answer"]["actions"] == ["Check the contingency.", SEEK_COUNSEL_ACTION]
    decisions = proposer_step(data)["action_decisions"]
    verify = next(d for d in decisions if d["action"] == "Verify the moderate finding.")
    assert verify["status"] == "rejected"
    assert "referral kind" in verify["reason"]
    assert verify["reason"].endswith("(moderate)")


def test_contingency_kind_on_a_strong_anchor_is_rejected(live_client):
    """The gate's mapping runs both ways (AC 2): a contingency-kind proposal
    grounded only on the strong Finding is rejected, leaving the hand-off."""
    llm = make_offline_llm()
    llm.proposals = ActionProposals(
        proposals=[
            ActionProposal(action="A contingency on a strong claim.", kind="contingency", citation_refs=["C1"]),
        ]
    )
    install_fake_pipeline(llm, FakeRetriever())

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    assert data["answer"]["actions"] == [SEEK_COUNSEL_ACTION]
    decisions = proposer_step(data)["action_decisions"]
    assert decisions[0]["status"] == "rejected"
    assert "referral kind" in decisions[0]["reason"]
    assert decisions[0]["reason"].endswith("(strong)")


def test_partial_grounding_drops_are_recorded_on_the_kept_decision(live_client):
    """A kept proposal citing both resolvable and unknown labels records the
    dropped labels on its decision — a partial grounding failure never
    vanishes from the detailed trace either (AC 3)."""
    llm = make_offline_llm()
    llm.proposals = ActionProposals(
        proposals=[
            ActionProposal(
                action="Verify the high-risk duties.",
                kind="verify_against_facts",
                citation_refs=["C1", "C99"],
            ),
        ]
    )
    install_fake_pipeline(llm, FakeRetriever())

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    assert data["answer"]["actions"][0] == "Verify the high-risk duties."
    decisions = proposer_step(data)["action_decisions"]
    kept = next(d for d in decisions if d["status"] == "kept")
    assert kept["dropped_refs"] == ["C99"]


def test_insufficient_evidence_path_is_unchanged_and_never_calls_the_proposer(monkeypatch):
    """Nothing retrieved → the narrowing Actions as before, and the Proposer
    makes no LLM call over no kept Findings."""
    llm = make_offline_llm()
    install_fake_pipeline(llm, FakeRetriever(chunks=[]))

    with TestClient(app) as client:
        resp = post_arbitrary_scenario(client)
    assert resp.status_code == 200
    data = resp.json()

    assert_insufficient_evidence_response(data)
    assert data["answer"]["actions"] == NARROW_THE_QUESTION_ACTIONS
    assert [call for call in llm.calls if call[2] is ActionProposals] == []
    # The trace stays honest about the skipped pass (AC 5): it says nothing
    # was distilled, not that Findings were distilled.
    step = proposer_step(data)
    assert "Insufficient-evidence path" in step["action"]
    assert "no LLM call" in step["action"]
    assert step["action_decisions"] == []


def test_no_kept_findings_serves_the_hand_off_alone_without_a_proposer_call(monkeypatch):
    """Evidence retrieved but every Claim rejected: the Proposer handles the
    no-findings edge by serving the hand-off alone, without an LLM call."""
    from src.live_workflow import Verdicts

    llm = make_offline_llm()
    llm.verdicts = Verdicts(verdicts=[])
    install_fake_pipeline(llm, FakeRetriever())

    with TestClient(app) as client:
        resp = post_arbitrary_scenario(client)
    assert resp.status_code == 200
    data = resp.json()

    assert data["answer"]["actions"] == [SEEK_COUNSEL_ACTION]
    assert [call for call in llm.calls if call[2] is ActionProposals] == []
    assert "0 referral Action(s)" in data["trace"]["summary"]
    # The node itself emitted the hand-off alone (AC 1), and the trace says so.
    step = proposer_step(data)
    assert "hand-off was served alone" in step["action"]
    assert step["action_decisions"] == []


# --- Finding-label grounding rejection (ticket #36) --------------------------


def test_proposal_using_finding_labels_is_rejected_with_invalid_grounding(live_client):
    """A proposal that references Finding labels (F1) instead of Citation
    labels (C1) cannot create an Action, and the rejection identifies the
    invalid grounding (#36)."""
    llm = make_offline_llm()
    llm.proposals = ActionProposals(
        proposals=[
            ActionProposal(
                action="Verify the high-risk classification.",
                kind="verify_against_facts",
                citation_refs=["F1"],
            ),
        ]
    )
    install_fake_pipeline(llm, FakeRetriever())

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    assert "Verify the high-risk classification." not in data["answer"]["actions"]
    assert data["answer"]["actions"] == [SEEK_COUNSEL_ACTION]

    decisions = proposer_step(data)["action_decisions"]
    assert len(decisions) == 1
    assert decisions[0]["status"] == "rejected"
    assert "F1" in decisions[0]["reason"]
    assert "Finding" in decisions[0]["reason"]


def test_proposal_using_citation_labels_produces_grounded_actions(live_client):
    """A valid Proposer response using only displayed Citation labels produces
    the expected grounded referral Actions in the Answer (#36)."""
    llm = make_offline_llm()
    llm.proposals = ActionProposals(
        proposals=[
            ActionProposal(
                action="Have a professional verify the high-risk classification.",
                kind="verify_against_facts",
                citation_refs=["C1"],
            ),
        ]
    )
    install_fake_pipeline(llm, FakeRetriever())

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    assert data["answer"]["actions"][0] == "Have a professional verify the high-risk classification."
    assert data["answer"]["actions"][-1] == SEEK_COUNSEL_ACTION

    decisions = proposer_step(data)["action_decisions"]
    kept = [d for d in decisions if d["status"] == "kept"]
    assert len(kept) == 1
    assert kept[0]["action"] == "Have a professional verify the high-risk classification."


# --- Citation rider: source_short_name from corpus metadata (#24) ------------


def test_research_target_tolerates_a_bare_string_as_shorthand_for_query():
    """Live solar-pro4 emits the Planner's targets as bare keyword strings
    despite the structural shape instruction; the schema wraps each string
    into {'query': ...} instead of failing the whole workflow on shape."""
    from src.live_workflow import Plan

    plan = Plan.model_validate(
        {"targets": ["bare keywords", {"query": "explicit object"}]}
    )

    assert [target.query for target in plan.targets] == ["bare keywords", "explicit object"]


def test_live_citations_carry_the_source_short_name_from_corpus_metadata(live_client):
    """The Live Citation rider (#24): every Citation names its source the way
    the demo's do, populated from corpus metadata via the source id."""
    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    short_names = {c["source_id"]: c.get("source_short_name") for c in data["answer"]["citations"]}
    assert short_names["ai-act"] == "EU AI Act"


def test_derive_citation_populates_the_short_name_only_for_known_sources():
    """Unknown source ids degrade to no short name — the rider never invents
    corpus metadata for a source the corpus does not carry."""
    from src.live_workflow import derive_citation

    known = derive_citation(make_chunk(source_id="ai-act", number=6))
    assert known.source_short_name == "EU AI Act"
    unknown = derive_citation(make_chunk(source_id="some-future-regulation", number=1))
    assert unknown.source_short_name is None
