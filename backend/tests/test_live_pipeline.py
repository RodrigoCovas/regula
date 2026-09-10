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
from src.llm import LlmError
from src.live_workflow import (
    LIVE_WORKFLOW_MARKER,
    NARROW_THE_QUESTION_ACTIONS,
    SEATS_PER_TARGET,
    SEEK_COUNSEL_ACTION,
    ActionProposal,
    ActionProposals,
    DraftClaim,
    DraftClaims,
    Plan,
    ProvisionSummary,
    ResearchTarget,
    Summaries,
    Verdict,
    Verdicts,
)
from src.main import app
from src.models import ProvisionKind, Strength
from src.retrieval import LEXICAL_LEG_DEPTH, VECTOR_LEG_DEPTH, HybridRetriever

from conftest import install_fake_pipeline
from fakes import (
    AUTOMATED_DECISION_CHUNK,
    DEFINITIONS_CHUNK,
    HIGH_RISK_CHUNK,
    FakeEmbedder,
    FakeRetriever,
    FakeSearchStore,
    ScriptedLlm,
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


# --- The rubrics resolve Scenario facts at the HTTP seam (ticket #70) --------


def test_scenario_settled_contingency_is_discarded_as_unsupported_and_recorded(live_client):
    """A claim whose contingency the scenario text settles is judged on the
    stated facts (ticket #70): the Verifier marks the conditional 'if the
    company were a financial entity' form out of scope, the pipeline discards
    it as an Unsupported claim, and the Execution trace records it — while
    the same duty stated as the scenario states it stays in the Answer."""
    from src.live_workflow import DraftClaim, DraftClaims, Plan, ResearchTarget, Verdict, Verdicts

    conditional = (
        "If the company were a financial entity, its major ICT incidents would fall "
        "under DORA's incident-reporting regime."
    )
    stated = "The company's major ICT incidents fall under DORA's incident-reporting regime."
    incident_chunk = make_chunk(
        source_id="dora",
        number=19,
        text="Financial entities must classify and report major ICT-related incidents.",
        title="Reporting of major ICT-related incidents",
    )
    llm = make_offline_llm()
    llm.plan = Plan(targets=[ResearchTarget(query="ICT incident reporting")])
    llm.claims = DraftClaims(claims=[
        DraftClaim(statement=conditional, evidence_refs=["E1"]),
        DraftClaim(statement=stated, evidence_refs=["E1"]),
    ])
    llm.verdicts = Verdicts(verdicts=[
        Verdict(statement=conditional, supported=False),
        Verdict(statement=stated, supported=True, strength=Strength.strong, decisive_refs=["E1"]),
    ])
    llm.proposals = ActionProposals(proposals=[])
    install_fake_pipeline(llm, FakeRetriever(per_query={"ICT incident reporting": [incident_chunk]}))
    resp = live_client.post(
        "/api/analyze",
        json={
            "scenario": {
                "id": "licensed-lender",
                "description": "An online lender evaluating loan applications automatically, licensed as a credit institution in Spain",
            },
            "question": "What incident-reporting duties bind the company?",
        },
    )
    assert resp.status_code == 200
    data = resp.json()

    statements = [f["statement"] for f in data["answer"]["findings"]]
    assert stated in statements, "the claim as the scenario states it stays in the Answer"
    assert conditional not in statements, "the conditional form of a settled fact never appears"

    discarded = data["trace"]["unsupported_claims_discarded"]
    assert conditional in discarded

    verifier_steps = [s for s in data["detailed_trace"] if s["step"] == "verifier"]
    decisions = {d["claim"]: d for d in verifier_steps[0]["claim_decisions"]}
    assert decisions[conditional]["status"] == "rejected"
    assert decisions[conditional]["reason"] == "no Evidence in the Corpus supports this claim"
    assert decisions[stated]["status"] == "kept"


def test_supported_but_immaterial_claim_is_rejected_with_the_materiality_reason(live_client):
    """The Verifier's materiality judgment (spec #94, T4): a Claim the
    Evidence supports but whose provisions do not decide what the Answer
    turns on is rejected with its own reason — recorded in both traces,
    never reaching the Answer."""
    from src.live_workflow import DraftClaim, DraftClaims, Plan, ResearchTarget, Verdict, Verdicts

    kept = (
        "The company's automated loan decisions must not be based solely on "
        "automated processing."
    )
    immaterial = (
        "The GDPR's cooperation machinery requires supervisory authorities to "
        "cooperate with each other on cross-border cases."
    )
    llm = make_offline_llm()
    llm.plan = Plan(targets=[ResearchTarget(query="automated decisions")])
    llm.claims = DraftClaims(claims=[
        DraftClaim(statement=kept, evidence_refs=["E1"]),
        DraftClaim(statement=immaterial, evidence_refs=["E1"]),
    ])
    llm.verdicts = Verdicts(verdicts=[
        Verdict(statement=kept, supported=True, material=True, strength=Strength.strong, decisive_refs=["E1"]),
        Verdict(statement=immaterial, supported=True, material=False, strength=Strength.moderate, decisive_refs=["E1"]),
    ])
    llm.proposals = ActionProposals(proposals=[])
    llm.summaries = Summaries(summaries=[
        ProvisionSummary(
            ref="P1",
            relevance=(
                "Article 22 constrains decisions based solely on automated processing, "
                "which is what the Answer's conclusion about the loan decisions turns on."
            ),
            strength=Strength.strong,
        ),
    ])
    install_fake_pipeline(llm, FakeRetriever(per_query={"automated decisions": [AUTOMATED_DECISION_CHUNK]}))
    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    statements = [f["statement"] for f in data["answer"]["findings"]]
    assert kept in statements, "the material claim stays in the Answer"
    assert immaterial not in statements, "an Immaterial claim never reaches the Answer"

    discarded = data["trace"]["unsupported_claims_discarded"]
    assert immaterial in discarded, "the Execution trace counts the immaterial rejection"

    verifier_steps = [s for s in data["detailed_trace"] if s["step"] == "verifier"]
    decisions = {d["claim"]: d for d in verifier_steps[0]["claim_decisions"]}
    assert decisions[immaterial]["status"] == "rejected"
    assert decisions[immaterial]["reason"] == "its provisions do not decide what the Answer turns on"
    assert decisions[kept]["status"] == "kept"


def test_exclusion_finding_appears_when_the_pool_holds_a_perimeter_provision(live_client):
    """The applicability target seats the regime's perimeter provision in the
    Evidence pool, and the exclusion claim it supports becomes a Finding
    (ticket #70): the Answer explains why the excluded regime is left out,
    citing the perimeter provision."""
    from src.live_workflow import DraftClaim, DraftClaims, Plan, ResearchTarget, Verdict, Verdicts

    scope_chunk = make_chunk(
        source_id="dora",
        number=2,
        text="This Regulation applies to financial entities.",
        title="Scope",
    )
    exclusion = (
        "DORA's incident regime covers financial entities only, and the company is an "
        "online shop rather than a financial entity, so the regime does not reach it."
    )
    llm = make_offline_llm()
    llm.plan = Plan(targets=[ResearchTarget(query="DORA applicability financial entities")])
    llm.claims = DraftClaims(claims=[DraftClaim(statement=exclusion, evidence_refs=["E1"])])
    llm.verdicts = Verdicts(verdicts=[
        Verdict(statement=exclusion, supported=True, strength=Strength.moderate, decisive_refs=["E1"]),
    ])
    llm.proposals = ActionProposals(proposals=[])
    llm.summaries = Summaries(summaries=[
        ProvisionSummary(
            ref="P1",
            relevance=(
                "Article 2 sets DORA's perimeter — who the regime covers — which is "
                "what the Answer's exclusion conclusion turns on."
            ),
            strength=Strength.moderate,
        ),
    ])
    install_fake_pipeline(llm, FakeRetriever(per_query={"DORA applicability financial entities": [scope_chunk]}))
    resp = live_client.post(
        "/api/analyze",
        json={
            "scenario": {
                "id": "electronics-shop",
                "description": "An online shop selling consumer electronics from Madrid",
            },
            "question": "Do DORA's ICT incident duties apply to the company?",
        },
    )
    assert resp.status_code == 200
    data = resp.json()

    # The perimeter provision rode the applicability target into the pool.
    retrieved = served_evidence(data)
    assert [(r["source_id"], r["number"]) for r in retrieved] == [("dora", 2)]

    findings = data["answer"]["findings"]
    assert len(findings) == 1
    assert findings[0]["statement"] == exclusion
    assert findings[0]["strength"] == "moderate"
    assert [(c["source_id"], c["article_number"]) for c in findings[0]["citations"]] == [("dora", 2)]

    verifier_steps = [s for s in data["detailed_trace"] if s["step"] == "verifier"]
    decisions = verifier_steps[0]["claim_decisions"]
    assert decisions[0]["status"] == "kept", "the exclusion claim is the supported form, never a discard"
    assert data["trace"]["unsupported_claims_discarded"] == []


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
    # Each LLM call is visible in the scripted client's log: planner,
    # researcher, verifier, proposer, summarizer.
    assert len(llm.calls) == 5
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


def target_chunk_prefix(target: str) -> str:
    """The source_id prefix a target's retrieval results carry — minted by
    depth_results and hand-made fixtures alike, and the convention the
    representation assertions match against."""
    return f"{target}-"


def depth_results(target):
    """One full-depth hybrid retrieval for a target: up to ``VECTOR_LEG_DEPTH +
    LEXICAL_LEG_DEPTH`` unique Chunks (ADR-0012)."""
    return [
        make_chunk(source_id=f"{target_chunk_prefix(target)}{i}", number=i + 1)
        for i in range(VECTOR_LEG_DEPTH + LEXICAL_LEG_DEPTH)
    ]


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
    assert len(retrieved) > VECTOR_LEG_DEPTH + LEXICAL_LEG_DEPTH, "the widened pool out-seats one full hybrid retrieval — #23's fix"
    assert len(retrieved) <= SEATS_PER_TARGET * len(targets), "the plan-derived cap holds"
    assert any(s.startswith(target_chunk_prefix("narrow")) for s in sources), "the second target is represented"
    assert any(s.startswith(target_chunk_prefix("broad")) for s in sources), "the first target is still represented"


def assert_every_target_keeps_representation(targets, retrieved):
    """The pool derives from the plan: every planned target keeps at least
    one of its Chunks in the served Evidence — no target is starved."""
    sources = {r["source_id"] for r in retrieved}
    for target in targets:
        assert any(s.startswith(target_chunk_prefix(target)) for s in sources), f"{target} keeps representation"


def test_evidence_pool_scales_with_the_planned_target_count(live_client):
    """Seat demand tracks the Planner's decomposition (ADR-0003): a third
    research target widens the pool again, and every target keeps its share
    of seats."""
    targets = ["creditworthiness", "automated decisions", "deployer obligations"]
    data = run_plan(live_client, targets, {target: depth_results(target) for target in targets})

    retrieved = served_evidence(data)
    assert len(retrieved) == SEATS_PER_TARGET * len(targets)
    assert_every_target_keeps_representation(targets, retrieved)


def test_evidence_pool_seats_a_full_eight_target_plan(live_client):
    """A full plan — the raised budget's eight Research targets (issue #75,
    ADR-0015) — seats SEATS_PER_TARGET × 8 Chunks, and every target keeps its
    share: the pool bound scales with the budget, no target is starved."""
    targets = [
        "engagement threshold",
        "creditworthiness",
        "automated decisions",
        "deployer obligations",
        "incident reporting",
        "continuity arrangements",
        "records of processing activities",
        "model provider duties",
    ]
    data = run_plan(live_client, targets, {target: depth_results(target) for target in targets})

    retrieved = served_evidence(data)
    assert len(retrieved) == SEATS_PER_TARGET * 8
    assert_every_target_keeps_representation(targets, retrieved)


def test_pool_derives_from_the_plan_not_from_retrieval_volume(live_client):
    """A lone broad target retrieves up to ``VECTOR_LEG_DEPTH +
    LEXICAL_LEG_DEPTH`` Chunks but seats only ``SEATS_PER_TARGET`` of them:
    the pool is the plan's seat count — extra retrieval volume alone never
    widens what the agents reason over."""
    data = run_plan(live_client, ["creditworthiness"], {"creditworthiness": depth_results("broad")})

    retrieved = served_evidence(data)
    assert len(retrieved) == SEATS_PER_TARGET == 12


def test_duplicate_drafted_statements_first_kept_later_rejected_as_duplicate(monkeypatch):
    """Exact-normalized duplicate statements (spec #94, T4): first drafted
    wins — the first copy is decided on its own verdict and kept, every later
    exact-normalized duplicate is rejected with the duplicate reason, never
    vanishing from the trace."""
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
    assert len(findings) == 1, "exactly one Finding survives — first drafted wins"

    verifier_steps = [s for s in data["detailed_trace"] if s["step"] == "verifier"]
    decisions = [d for d in verifier_steps[0]["claim_decisions"] if d["claim"] == duplicated]
    statuses = sorted(d["status"] for d in decisions)
    assert statuses == ["kept", "rejected"], statuses
    rejected = next(d for d in decisions if d["status"] == "rejected")
    assert rejected["reason"] == "an earlier kept claim states the same"

    discarded = data["trace"]["unsupported_claims_discarded"]
    assert discarded.count(duplicated) == 1


def test_exact_normalized_duplicate_statements_are_rejected_as_duplicates(live_client):
    """Dedup is exact-normalized (casefold + whitespace collapse): a re-draft
    that differs only in case or spacing from a kept statement is the same
    statement — first drafted wins, the normalized twin is rejected with the
    duplicate reason (spec #94, T4)."""
    from src.live_workflow import DraftClaim, DraftClaims, Plan, ResearchTarget, Verdicts

    first = "The System  qualifies as an AI system under the definitions."
    twin = "the system qualifies as an AI system under the definitions."
    llm = make_offline_llm()
    llm.plan = Plan(targets=[ResearchTarget(query="definitions")])
    llm.claims = DraftClaims(
        claims=[
            DraftClaim(statement=first, evidence_refs=["E1"]),
            DraftClaim(statement=twin, evidence_refs=["E1"]),
        ]
    )
    llm.verdicts = Verdicts(verdicts=[
        grounded_verdict(first, Strength.weak, ["E1"]),
        grounded_verdict(twin, Strength.weak, ["E1"]),
    ])
    llm.proposals = ActionProposals(proposals=[])
    install_fake_pipeline(llm, FakeRetriever(per_query={"definitions": [DEFINITIONS_CHUNK]}))
    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    findings = [f["statement"] for f in data["answer"]["findings"]]
    assert findings == [first], "only the first drafted copy reaches the Answer"

    verifier_steps = [s for s in data["detailed_trace"] if s["step"] == "verifier"]
    decisions = {d["claim"]: d for d in verifier_steps[0]["claim_decisions"]}
    assert decisions[first]["status"] == "kept"
    assert decisions[twin]["status"] == "rejected"
    assert decisions[twin]["reason"] == "an earlier kept claim states the same"

    discarded = data["trace"]["unsupported_claims_discarded"]
    assert discarded.count(twin) == 1


def test_duplicate_statements_are_decided_in_draft_order_first_drafted_wins(live_client):
    """'First drafted wins' is literal (spec review finding): the kept copy
    of a duplicated statement is the first-drafted one's, even when the
    Verifier returns its verdicts in the opposite order."""
    from src.live_workflow import DraftClaim, DraftClaims, Plan, ResearchTarget, Verdicts

    first = "The System  qualifies as an AI system under the definitions."
    twin = "the system qualifies as an AI system under the definitions."
    llm = make_offline_llm()
    llm.plan = Plan(targets=[ResearchTarget(query="definitions")])
    llm.claims = DraftClaims(
        claims=[
            DraftClaim(statement=first, evidence_refs=["E1"]),
            DraftClaim(statement=twin, evidence_refs=["E1"]),
        ]
    )
    llm.verdicts = Verdicts(verdicts=[
        grounded_verdict(twin, Strength.weak, ["E1"]),
        grounded_verdict(first, Strength.weak, ["E1"]),
    ])
    llm.proposals = ActionProposals(proposals=[])
    install_fake_pipeline(llm, FakeRetriever(per_query={"definitions": [DEFINITIONS_CHUNK]}))
    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    findings = [f["statement"] for f in data["answer"]["findings"]]
    assert findings == [first], "the first-drafted copy's verdict is the one kept"

    verifier_steps = [s for s in data["detailed_trace"] if s["step"] == "verifier"]
    decisions = {d["claim"]: d for d in verifier_steps[0]["claim_decisions"]}
    assert decisions[first]["status"] == "kept"
    assert decisions[twin]["status"] == "rejected"
    assert decisions[twin]["reason"] == "an earlier kept claim states the same"


def test_a_duplicate_of_a_rejected_statement_is_decided_on_its_own_verdict(live_client):
    """Dedup keys on kept statements: a later copy whose earlier twin was
    rejected is judged on its own verdict — an Unsupported first copy never
    poisons the supported later one (spec #94, T4)."""
    from src.live_workflow import DraftClaim, DraftClaims, Plan, ResearchTarget, Verdict, Verdicts

    duplicated = "The company's loan scoring data counts as special-category data."
    llm = make_offline_llm()
    llm.plan = Plan(targets=[ResearchTarget(query="special-category data")])
    llm.claims = DraftClaims(
        claims=[
            DraftClaim(statement=duplicated, evidence_refs=["E1"]),
            DraftClaim(statement=duplicated, evidence_refs=["E1"]),
        ]
    )
    llm.verdicts = Verdicts(verdicts=[
        Verdict(statement=duplicated, supported=False, material=True),
        Verdict(statement=duplicated, supported=True, material=True, strength=Strength.strong, decisive_refs=["E1"]),
    ])
    llm.proposals = ActionProposals(proposals=[])
    llm.summaries = Summaries(summaries=[
        ProvisionSummary(
            ref="P1",
            relevance=(
                "The special-category provision is what the Answer's conclusion about "
                "the loan scoring data turns on."
            ),
            strength=Strength.strong,
        ),
    ])
    install_fake_pipeline(llm, FakeRetriever(per_query={"special-category data": [HIGH_RISK_CHUNK]}))
    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    findings = [f["statement"] for f in data["answer"]["findings"]]
    assert findings == [duplicated], "the supported later copy is kept"

    verifier_steps = [s for s in data["detailed_trace"] if s["step"] == "verifier"]
    decisions = [d for d in verifier_steps[0]["claim_decisions"] if d["claim"] == duplicated]
    statuses = [d["status"] for d in decisions]
    assert statuses == ["rejected", "kept"], statuses
    assert "no Evidence" in decisions[0]["reason"]

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


# --- Insufficient evidence fires only when both legs come back empty (issue #64) ---


def test_junk_only_corpus_answers_insufficient_evidence_never_fabricated_findings():
    """Both retrieval legs come back empty — every vector hit sits beyond the
    relevance floor and the lexical leg matches nothing — so the served
    answer says so honestly: zero Findings, a Known limitation naming the
    gap, narrowing Actions — even though the scripted LLM stands ready to
    fabricate grounded-looking claims if it were ever asked to draft."""
    junk = [chunk_hit(source_id=f"junk-{i}", number=i + 1, distance=0.9) for i in range(3)]
    # The shared FakeSearchStore replays its hits whatever bound it is given
    # (the vector leg's adversarial case) and holds no lexical matches:
    # neither leg returns anything.
    retriever = HybridRetriever(store=FakeSearchStore(hits=junk), embedder=FakeEmbedder())
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


def test_lexical_only_evidence_still_produces_findings_and_citations():
    """The Insufficient-evidence path fires only when *neither* leg returns
    anything (ADR-0012): with every vector hit beyond the relevance floor,
    the lexical leg's exact-term matches alone fill the Evidence pool —
    lexically-obvious provisions are no longer hidden behind an embedding
    miss, and the answer carries Findings and Citations."""
    lexical_only = [
        make_chunk(
            source_id="gdpr",
            number=30,
            text="The controller shall maintain records of processing activities.",
        ),
    ]
    retriever = HybridRetriever(
        store=FakeSearchStore(hits=[chunk_hit(source_id="junk", number=1, distance=0.9)], lexical_hits=lexical_only),
        embedder=FakeEmbedder(),
    )
    llm = make_offline_llm()
    install_fake_pipeline(llm, retriever)

    with TestClient(app) as client:
        resp = post_arbitrary_scenario(client)
    assert resp.status_code == 200
    data = resp.json()

    findings = data["answer"]["findings"]
    assert findings, "lexical-only Evidence still produces Findings"
    assert data["answer"]["citations"], "lexical-only Evidence still produces Citations"
    assert all(c["source_id"] == "gdpr" for c in data["answer"]["citations"])
    assert [r["source_id"] for r in served_evidence(data)] == ["gdpr"], "the pool filled from the lexical leg alone"
    assert not any("nothing relevant" in line.lower() for line in data["known_limitations"])


# --- Recitals never enter Evidence or Citations (ticket #69) -------------------


def test_recital_hits_from_an_adversarial_store_never_reach_evidence_or_citations():
    """End-to-end (ticket #69): a store that ignores the pushed-down Recital
    exclusion cannot seat a Recital Chunk — the retriever re-checks both legs
    at the seam, so a Recital Chunk surfacing in the Evidence pool never
    becomes a Citation and the pool's seats go to operative provisions."""
    retriever = HybridRetriever(
        store=FakeSearchStore(
            hits=[
                chunk_hit(source_id="gdpr", kind="recital", number=71, text="Recital 71 body", distance=0.05),
                chunk_hit(source_id="gdpr", number=22, text="Automated individual decision-making.", distance=0.1),
            ],
            lexical_hits=[
                make_chunk(source_id="gdpr", kind=ProvisionKind.recital, number=70, text="Recital 70 body"),
                make_chunk(source_id="gdpr", number=13, text="Transparent information and obligations."),
            ],
        ),
        embedder=FakeEmbedder(),
    )
    llm = make_offline_llm()
    install_fake_pipeline(llm, retriever)

    with TestClient(app) as client:
        resp = post_arbitrary_scenario(client)
    assert resp.status_code == 200
    data = resp.json()

    served = served_evidence(data)
    assert served, "the store's operative chunks still fill the pool"
    assert all(item["kind"] != "recital" for item in served), "no Recital Chunk surfaces in the Evidence pool"
    assert [item["number"] for item in served] == [22, 13], "operative provisions took the seats"

    citations = data["answer"]["citations"]
    assert citations, "the operative Evidence still yields Citations"
    for citation in citations:
        assert citation["recital_number"] is None
    for finding in data["answer"]["findings"]:
        for citation in finding["citations"]:
            assert citation["recital_number"] is None


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


# --- Summarizer: Provision relevance and Citation strength (issue #47) ---------


def summarizer_step(data) -> dict:
    """The Summarizer's detailed-trace step."""
    return [s for s in data["detailed_trace"] if s["step"] == "summarizer"][0]


def test_live_answer_citations_carry_relevance_and_rated_strength(live_client):
    """The served Answer's Citations carry one entry per cited provision, each
    with its Summarizer relevance and the rated Citation strength the
    Summarizer carried (ADR-0011) — while the per-Finding Citations carry
    neither (issue #47)."""
    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    citations = data["answer"]["citations"]
    assert len(citations) == 2, "one entry per cited provision, not per Finding citation"
    by_number = {c["article_number"]: c for c in citations}
    assert by_number[HIGH_RISK_CHUNK.article_number]["strength"] == "strong"
    assert by_number[DEFINITIONS_CHUNK.article_number]["strength"] == "weak"
    assert all(c["relevance"] for c in citations), "every cited provision carries relevance"

    # The per-Finding Citations stay bare: relevance and strength are
    # answer-wide and ride the Answer's list only.
    for finding in data["answer"]["findings"]:
        for citation in finding["citations"]:
            assert citation["relevance"] is None
            assert citation["strength"] is None

    step = summarizer_step(data)
    assert step["action"]
    kept = [d for d in step["summary_decisions"] if d["status"] == "kept"]
    assert len(kept) == 2, "one kept decision per cited provision"
    assert {d["strength"] for d in kept} == {"strong", "weak"}, "kept decisions carry their rating"


def test_grounding_gate_drops_summaries_for_unknown_labels(live_client):
    """A relevance statement whose ref names no cited provision never reaches
    the Answer and is recorded as rejected in the detailed trace — the
    Summarizer's grounding gate (the Action proposals' discipline)."""
    llm = make_offline_llm()
    llm.summaries = Summaries(
        summaries=[
            ProvisionSummary(ref="P1", relevance="Grounded in the citing Findings."),
            ProvisionSummary(ref="P99", relevance="Drawn on nothing the Findings say."),
            ProvisionSummary(ref="F1", relevance="A Finding label, not a provision."),
        ]
    )
    install_fake_pipeline(llm, FakeRetriever())

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    citations = data["answer"]["citations"]
    grounded = [c for c in citations if c["relevance"]]
    assert [c["article_number"] for c in grounded] == [HIGH_RISK_CHUNK.article_number]
    assert grounded[0]["relevance"] == "Grounded in the citing Findings."

    decisions = summarizer_step(data)["summary_decisions"]
    # Both passes are recorded: the first left P2 bare, so the one corrective
    # re-prompt (issue #82) replayed the batch — its P1 repeat is rejected as
    # the duplicate it is, alongside the unknown labels from each pass.
    rejected = [d for d in decisions if d["status"] == "rejected"]
    assert {d["ref"] for d in rejected} == {"P99", "F1", "P1"}
    # The ungated statements never reached the Answer.
    served_relevance = [c["relevance"] for c in citations if c["relevance"]]
    assert "Drawn on nothing the Findings say." not in served_relevance
    assert "A Finding label, not a provision." not in served_relevance


def test_the_rating_rides_the_provision_never_the_finding_strengths(live_client):
    """Two kept Findings citing one provision, at different Strengths: the
    Answer's Citation carries the rating the Summarizer gave that provision —
    even one no citing Finding's Strength matches. Nothing is derived from
    Finding strength any more (ADR-0011)."""
    from src.live_workflow import DraftClaim, DraftClaims, Plan, ResearchTarget, Verdicts

    strong_statement = "A strong claim about creditworthiness."
    weak_statement = "A weak framing claim over the same provision."
    llm = make_offline_llm()
    llm.plan = Plan(targets=[ResearchTarget(query="creditworthiness")])
    llm.claims = DraftClaims(
        claims=[
            DraftClaim(statement=strong_statement, evidence_refs=["E1"]),
            DraftClaim(statement=weak_statement, evidence_refs=["E1"]),
        ]
    )
    llm.verdicts = Verdicts(
        verdicts=[
            grounded_verdict(strong_statement, Strength.strong, ["E1"]),
            grounded_verdict(weak_statement, Strength.weak, ["E1"]),
        ]
    )
    llm.summaries = Summaries(
        summaries=[ProvisionSummary(ref="P1", relevance="Grounded in the citing Findings.", strength=Strength.moderate)]
    )
    install_fake_pipeline(llm, FakeRetriever())

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    assert len(data["answer"]["findings"]) == 2
    citations = data["answer"]["citations"]
    assert len(citations) == 1, "both Findings cite one provision: one Answer entry"
    assert citations[0]["article_number"] == HIGH_RISK_CHUNK.article_number
    assert citations[0]["strength"] == "moderate", "the Summarizer's rating, not the max-rule"


def test_a_missing_rating_keeps_the_relevance_and_is_recorded(live_client):
    """A provision the Summarizer left unrated keeps its relevance, carries no
    strength — never a default — and the gap is recorded with a reason on the
    kept decision in the detailed trace (ADR-0011)."""
    llm = make_offline_llm()
    llm.summaries = Summaries(
        summaries=[
            ProvisionSummary(ref="P1", relevance="Grounded in the citing Findings."),
            ProvisionSummary(ref="P2", relevance="Also grounded.", strength=Strength.weak),
        ]
    )
    install_fake_pipeline(llm, FakeRetriever())

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    citations = data["answer"]["citations"]
    by_number = {c["article_number"]: c for c in citations}
    assert by_number[HIGH_RISK_CHUNK.article_number]["relevance"] == "Grounded in the citing Findings."
    assert by_number[HIGH_RISK_CHUNK.article_number]["strength"] is None
    assert by_number[DEFINITIONS_CHUNK.article_number]["strength"] == "weak"

    kept = {d["ref"]: d for d in summarizer_step(data)["summary_decisions"] if d["status"] == "kept"}
    assert kept["P1"]["strength"] is None
    assert kept["P1"]["reason"], "the missing rating is recorded, never defaulted"
    assert kept["P2"]["strength"] == "weak"


def test_an_invalid_rating_keeps_the_relevance_and_is_recorded(live_client):
    """A rating outside the three levels is no rating: the relevance ships,
    the strength does not, and the reason names the value (ADR-0011)."""
    llm = make_offline_llm()
    llm.summaries = Summaries(
        summaries=[
            ProvisionSummary(ref="P1", relevance="Grounded in the citing Findings.", strength="decisive"),
            ProvisionSummary(ref="P2", relevance="Also grounded.", strength={"level": "strong"}),
        ]
    )
    install_fake_pipeline(llm, FakeRetriever())

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    by_number = {c["article_number"]: c for c in data["answer"]["citations"]}
    assert by_number[HIGH_RISK_CHUNK.article_number]["strength"] is None
    assert by_number[DEFINITIONS_CHUNK.article_number]["strength"] == "strong", "the wrapped object is unwrapped"

    kept = {d["ref"]: d for d in summarizer_step(data)["summary_decisions"] if d["status"] == "kept"}
    assert kept["P1"]["strength"] is None
    assert "decisive" in (kept["P1"]["reason"] or "")
    assert kept["P2"]["strength"] == "strong"


def test_summarizer_failure_degrades_gracefully_never_fails_the_run(live_client):
    """A Summarizer failure is not a failed run: the Answer ships with its
    Findings and Citations intact but without relevance, plus a Known
    limitation naming the gap (#47)."""
    class SummarizerFailsLlm:
        """Serves every earlier agent, then dies at the Summaries boundary."""

        def __init__(self, base):
            self._base = base

        def complete(self, system, user, schema):
            if schema is Summaries:
                raise LlmError("OpenRouter rejected the request (HTTP 429)")
            return self._base.complete(system, user, schema)

    llm = SummarizerFailsLlm(make_offline_llm())
    install_fake_pipeline(llm, FakeRetriever())

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    assert len(data["answer"]["findings"]) == 2, "the Findings are unaffected"
    assert data["answer"]["citations"], "the Citations are unaffected"
    assert all(c["relevance"] is None for c in data["answer"]["citations"])
    assert all(c["strength"] is None for c in data["answer"]["citations"])
    assert any(
        "provision relevance is unavailable" in line.lower()
        for line in data["known_limitations"]
    ), data["known_limitations"]
    assert "without Provision relevance" in summarizer_step(data)["action"]


def test_an_all_rejected_summary_batch_degrades_like_a_failure(live_client):
    """Every statement the Summarizer offered was rejected by the gate — in
    both passes, after the one corrective re-prompt (issue #82) gave it a
    second chance: the Answer ships without relevance plus the same Known
    limitation — a silent absence would look like a clean run."""
    llm = make_offline_llm()
    llm.summaries = Summaries(
        summaries=[ProvisionSummary(ref="P99", relevance="Grounded on nothing.")]
    )
    install_fake_pipeline(llm, FakeRetriever())

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    assert all(c["relevance"] is None for c in data["answer"]["citations"])
    assert any(
        "provision relevance is unavailable" in line.lower()
        for line in data["known_limitations"]
    )
    step = summarizer_step(data)
    decisions = step["summary_decisions"]
    assert [d["status"] for d in decisions] == ["rejected", "rejected"]
    correction = step["correction"]
    assert "P1" in correction and "P2" in correction


def test_summarizer_prompt_shows_only_the_citing_findings_material(live_client):
    """The Summarizer's prompt groups the kept Findings' Citations by
    provision with the citing Findings' statements — and nothing else: the
    grounding window is exactly the citing Findings' material (#47)."""
    llm = make_offline_llm()
    install_fake_pipeline(llm, FakeRetriever())
    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200

    summarizer_system, summarizer_user = llm.calls[4][0], llm.calls[4][1]
    # The prompt is the block alone: the provisions with the citing Findings'
    # statements — no framing prose, no extra material.
    assert summarizer_user.startswith("[P1]")
    assert "[P2]" in summarizer_user
    assert "cited by" in summarizer_user
    # The citing Findings' statements ride along as the grounding material.
    assert "Creditworthiness evaluation is a high-risk use case." in summarizer_user
    # The rating contract rides the system prompt: the three levels, judged
    # only from the citing Findings (ADR-0011).
    assert "Citation strength" in summarizer_system
    for level in ("'strong'", "'moderate'", "'weak'"):
        assert level in summarizer_system
    # Nothing outside the citing Findings: not the question, not the Evidence
    # text — CONTEXT.md's grounding boundary is the prompt's boundary.
    assert "Does automated loan scoring trigger high-risk obligations?" not in summarizer_user
    assert "Does automated loan scoring trigger high-risk obligations?" not in summarizer_system
    assert "Automated individual decision-making" not in summarizer_user


def test_partial_summary_coverage_ships_what_survived_plus_a_known_limitation(live_client):
    """The Summarizer ran but covered only some cited provisions — and the
    one corrective re-prompt (issue #82) did not close the gap: the grounded
    statements it did produce still ship, and a Known limitation names the gap
    instead of letting partial coverage pass silently (#47)."""
    llm = make_offline_llm()
    llm.summaries = Summaries(
        summaries=[ProvisionSummary(ref="P1", relevance="Grounded in the citing Findings.")]
    )
    install_fake_pipeline(llm, FakeRetriever())

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    citations = data["answer"]["citations"]
    assert len(citations) == 2
    covered = [c for c in citations if c["relevance"]]
    bare = [c for c in citations if not c["relevance"]]
    assert len(covered) == 1 and len(bare) == 1
    assert any(
        "provision relevance is incomplete" in line.lower()
        and "1 of 2" in line
        for line in data["known_limitations"]
    ), data["known_limitations"]
    assert "covered 1 of 2" in summarizer_step(data)["action"]
    assert "covered 1 of 2 cited" in data["trace"]["summary"]
    # The bound held: exactly one corrective re-prompt, never a loop.
    assert len(summaries_calls(llm)) == 2
    assert "P2" in summarizer_step(data)["correction"]


# --- Summarizer partial-coverage corrective re-prompt (issue #82) --------------


def summaries_calls(llm) -> list[tuple[str, str, type]]:
    """The Summarizer boundary's calls in the scripted client's log."""
    return [call for call in llm.calls if call[2] is Summaries]


def test_partial_coverage_triggers_one_corrective_re_prompt_and_the_second_pass_ships(live_client):
    """A first pass that leaves a cited provision without Provision relevance
    earns exactly one corrective re-prompt naming that ref; the second pass
    fills the gap while the first pass's statements stand, and the correction
    is recorded in the detailed trace (issue #82 — the insurance case's
    unrated citations stop flooring fidelity pairs)."""
    llm = make_offline_llm()
    llm.summaries = Summaries(
        summaries=[
            ProvisionSummary(ref="P1", relevance="First pass covered P1.", strength=Strength.strong)
        ]
    )
    llm.summaries_retry = Summaries(
        summaries=[
            ProvisionSummary(
                ref="P2",
                relevance="Second pass covers the definitions provision.",
                strength=Strength.weak,
            ),
            ProvisionSummary(
                ref="P1",
                relevance="A repeat of what the first pass already covered.",
                strength=Strength.strong,
            ),
        ]
    )
    install_fake_pipeline(llm, FakeRetriever())

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    # Exactly one corrective re-prompt: two Summaries calls, never three.
    assert len(summaries_calls(llm)) == 2
    # The re-prompt repeats the grounding block and names the uncovered ref.
    _, retry_user, _ = summaries_calls(llm)[1]
    assert "[P1]" in retry_user and "[P2]" in retry_user, "the grounding block rides again"
    assert "P2" in retry_user
    # The second pass's statement ships; the first pass's stands.
    by_number = {c["article_number"]: c for c in data["answer"]["citations"]}
    assert by_number[HIGH_RISK_CHUNK.article_number]["relevance"] == "First pass covered P1."
    assert by_number[DEFINITIONS_CHUNK.article_number]["relevance"] == (
        "Second pass covers the definitions provision."
    )
    assert by_number[DEFINITIONS_CHUNK.article_number]["strength"] == "weak"
    # The correction is recorded in the detailed trace, listing the ref.
    step = summarizer_step(data)
    assert "P2" in step["correction"]
    assert "re-prompt" in step["correction"]
    # The repeated statement is recorded as the duplicate it is.
    duplicates = [d for d in step["summary_decisions"] if d["ref"] == "P1" and d["status"] == "rejected"]
    assert duplicates and "already carries" in duplicates[0]["reason"]
    # Complete coverage after the retry: no Known limitation for the gap.
    assert not any("provision relevance is incomplete" in line.lower() for line in data["known_limitations"])


def test_a_failing_re_prompt_still_ships_the_partial_summary(live_client):
    """The corrective re-prompt's call itself fails: the first pass's grounded
    statements ship with the incomplete-coverage Known limitation, and the
    correction is still recorded — a run never fails on summarizer trouble
    (issue #82)."""
    base = make_offline_llm()
    base.summaries = Summaries(
        summaries=[
            ProvisionSummary(ref="P1", relevance="First pass covered P1.", strength=Strength.strong)
        ]
    )

    class RetryFailsLlm:
        """Serves the first pass, then dies at the corrective re-prompt."""

        def __init__(self, inner):
            self._inner = inner
            self.summaries_calls = 0

        def complete(self, system, user, schema):
            if schema is Summaries:
                self.summaries_calls += 1
                if self.summaries_calls == 2:
                    raise LlmError("OpenRouter rejected the request (HTTP 429)")
            return self._inner.complete(system, user, schema)

    llm = RetryFailsLlm(base)
    install_fake_pipeline(llm, FakeRetriever())

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    assert llm.summaries_calls == 2
    by_number = {c["article_number"]: c for c in data["answer"]["citations"]}
    assert by_number[HIGH_RISK_CHUNK.article_number]["relevance"] == "First pass covered P1."
    assert by_number[DEFINITIONS_CHUNK.article_number]["relevance"] is None
    assert any(
        "provision relevance is incomplete" in line.lower() and "1 of 2" in line
        for line in data["known_limitations"]
    ), data["known_limitations"]
    assert "P2" in summarizer_step(data)["correction"]


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


# --- Planner grounding: the Corpus inventory and the 1-50 budget (issues #62, #75) ---


def test_planner_prompt_carries_the_corpus_inventory_and_the_target_budget(live_client):
    """The Planner is grounded in the Corpus's own table of contents (issue #62,
    ADR-0012): the request-time prompt shows the titled provisions grouped by
    source, and the system prompt declares the 1-50 budget (ADR-0018, raised
    from issue #75's 1-8) with the per-regulation and per-duty-area discipline
    and the vocabulary-lifting instruction."""
    llm = make_offline_llm()
    install_fake_pipeline(llm, FakeRetriever())
    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200

    planner_system, planner_user, _ = llm.calls[0]
    # The budget and the decomposition discipline ride the system prompt.
    assert "1-50" in planner_system
    assert "per regulation" in planner_system
    assert "duty area" in planner_system
    assert "never merge" in planner_system
    assert "guidance, not a constraint" in planner_system
    assert "exact vocabulary" in planner_system
    # The user prompt carries the full inventory block: titled provisions
    # grouped by source — the corpus's own table of contents, from chunk
    # metadata.
    assert "Corpus inventory" in planner_user
    assert "EU AI Act:" in planner_user
    assert "DORA:" in planner_user
    assert "GDPR:" in planner_user
    assert "Article 3: Definitions" in planner_user
    assert "Article 6: ICT risk management framework" in planner_user
    assert "Annex 1: List of Union Harmonisation Legislation" in planner_user
    # The scenario and the question still travel in the same user prompt.
    assert "Regulatory question: Does automated loan scoring trigger high-risk obligations?" in planner_user


# --- Planner Research-target budget enforcement (ticket #37) -----------------


# A full plan at the raised budget (ADR-0018): the fifty Research targets a
# maximal accepted plan carries, plus the fifty-first a provider response may
# add — the one the one-shot truncation correction drops.
FULL_PLAN_TARGETS = [
    "creditworthiness",
    "automated decisions",
    "deployer obligations",
    "incident reporting",
    "records of processing activities",
    "human oversight",
    "contract clauses",
    "continuity arrangements",
    *(f"extra target {i}" for i in range(1, 43)),
]
OVER_BUDGET_TARGET = "over-budget target"


def planner_step(data) -> dict:
    """The Planner's detailed-trace step."""
    return [s for s in data["detailed_trace"] if s["step"] == "planner"][0]


def run_with_plan(live_client, targets, per_query=None):
    """Run a scenario with the given ResearchTarget objects, returning the response data."""
    from src.live_workflow import Plan

    llm = make_offline_llm()
    llm.plan = Plan(targets=targets)
    install_fake_pipeline(llm, FakeRetriever(per_query=per_query or {}))
    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    return resp.json()


def test_planner_response_with_three_targets_is_accepted_unchanged(live_client):
    """A Planner response with three Research targets is accepted
    without correction — within the declared 1-50 budget."""
    from src.live_workflow import ResearchTarget

    data = run_with_plan(live_client, [
        ResearchTarget(query="creditworthiness"),
        ResearchTarget(query="automated decisions"),
        ResearchTarget(query="deployer obligations"),
    ])

    step = planner_step(data)
    assert len(step["research_targets"]) == 3
    assert step["research_targets"] == ["creditworthiness", "automated decisions", "deployer obligations"]
    assert step.get("correction") is None


def test_planner_response_with_one_target_is_accepted_unchanged(live_client):
    """A Planner response with one Research target is accepted — the lower
    end of the 1-50 budget works unchanged."""
    from src.live_workflow import ResearchTarget

    data = run_with_plan(live_client, [ResearchTarget(query="creditworthiness")])

    step = planner_step(data)
    assert len(step["research_targets"]) == 1
    assert step["research_targets"] == ["creditworthiness"]
    assert step.get("correction") is None


def test_planner_response_with_zero_targets_falls_through_to_insufficient_evidence(live_client):
    """A Planner response with zero Research targets is under-limit, not
    over-limit: no correction is applied, and the workflow naturally reaches
    the Insufficient-evidence path (nothing to research). The acceptance
    criteria focus on the upper bound; an empty plan is a degenerate case
    that the system handles honestly."""
    from src.live_workflow import Plan

    data = run_with_plan(live_client, [])

    step = planner_step(data)
    assert len(step["research_targets"]) == 0
    assert step.get("correction") is None
    assert_insufficient_evidence_response(data)


def test_planner_response_over_budget_is_truncated_to_fifty(live_client):
    """A Planner response with more than fifty Research targets is corrected
    once by truncation: only the first fifty targets are accepted, the rest
    are silently dropped — the Evidence pool never expands beyond the
    accepted plan (issue #75: the positional semantics are unchanged at the
    raised budget, ADR-0018)."""
    from src.live_workflow import ResearchTarget

    data = run_with_plan(
        live_client,
        [ResearchTarget(query=t) for t in [*FULL_PLAN_TARGETS, OVER_BUDGET_TARGET]],
    )

    step = planner_step(data)
    assert len(step["research_targets"]) == 50
    assert step["research_targets"][:3] == ["creditworthiness", "automated decisions", "deployer obligations"]
    assert step["correction"] is not None
    assert "truncated" in step["correction"].lower()
    assert "51" in step["correction"]
    assert "50" in step["correction"]


def test_evidence_pool_is_bounded_by_accepted_plan_not_provider_response(live_client):
    """The Evidence pool derives from the accepted plan, not the provider's
    over-limit response: even when the Planner returns fifty-one targets,
    only fifty seats' worth of Evidence is retrieved — a full plan seats
    SEATS_PER_TARGET × 50 (ADR-0018's raised budget)."""
    from src.live_workflow import ResearchTarget, SEATS_PER_TARGET

    targets = [*FULL_PLAN_TARGETS, OVER_BUDGET_TARGET]
    per_query = {t: depth_results(t) for t in targets}
    data = run_with_plan(
        live_client,
        [ResearchTarget(query=t) for t in targets],
        per_query=per_query,
    )

    retrieved = served_evidence(data)
    assert len(retrieved) == SEATS_PER_TARGET * 50, "pool bounded by accepted plan, not provider response"
    assert_every_target_keeps_representation(FULL_PLAN_TARGETS, retrieved)
    sources = {r["source_id"] for r in retrieved}
    assert not any(
        s.startswith(target_chunk_prefix(OVER_BUDGET_TARGET)) for s in sources
    ), "the fifty-first target was dropped"


def test_planner_correction_is_recorded_in_the_execution_trace(live_client):
    """An over-limit Planner response leaves a correction record in the
    detailed trace — the correction is transparent, never silent."""
    from src.live_workflow import ResearchTarget

    data = run_with_plan(
        live_client,
        [ResearchTarget(query=t) for t in [*FULL_PLAN_TARGETS, OVER_BUDGET_TARGET]],
    )

    step = planner_step(data)
    assert step["correction"] is not None
    assert "51" in step["correction"]
    assert "50" in step["correction"]


# --- Scenario-relevance discipline for the Researcher and Verifier (issue #63) ---


def test_researcher_prompt_ties_claims_to_the_scenario(live_client):
    """The Researcher drafts only claims that bear on the question asked for
    this scenario (issue #63) — what the scenario's actors must do, must not
    do, or how the provisions' rules reach them — and never restates a
    provision's content in the abstract. The grounding and evidence-ref
    discipline is unchanged."""
    llm = make_offline_llm()
    install_fake_pipeline(llm, FakeRetriever())
    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200

    researcher_system, _, _ = llm.calls[1]
    # The scenario-tie instruction: only claims bearing on the question.
    assert "question asked for this scenario" in researcher_system
    assert "must do" in researcher_system
    assert "must not do" in researcher_system
    assert "how the provisions' rules reach them" in researcher_system
    # The no-restatement rule.
    assert "never restate a provision's content in the abstract" in researcher_system
    # Grounding and evidence-ref discipline unchanged.
    assert "grounded ONLY in the listed evidence" in researcher_system
    assert "never reference a label that was not given to you" in researcher_system


def test_verifier_supported_bar_demands_scenario_relevance(live_client):
    """The Verifier's supported bar (issue #63): some listed provision bears
    on answering the question asked for this scenario — false when the claim
    merely restates provisions without bearing on the scenario, or when none
    bears either way. The strength levels and the bare-string contract are
    unchanged (ticket #70 turns the moderate clause toward the Scenario)."""
    llm = make_offline_llm()
    install_fake_pipeline(llm, FakeRetriever())
    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200

    verifier_system, _, _ = llm.calls[2]
    # The new supported bar.
    assert "bears on answering the question asked for this scenario" in verifier_system
    assert "merely restates provisions without bearing on the scenario" in verifier_system
    assert "none bears either way" in verifier_system
    # The strength rubric keeps its levels and the bare-string contract.
    assert "'strong'" in verifier_system
    assert "'moderate'" in verifier_system
    assert "'weak'" in verifier_system
    assert "directly and explicitly establish the claim" in verifier_system
    assert "framing only (definitions, vocabulary)" in verifier_system


# --- The rubrics resolve Scenario facts (ticket #70) -------------------------


def recorded_system_prompts(live_client) -> dict[str, str]:
    """Run one offline scenario and return each agent's system prompt keyed
    by agent: planner, researcher, verifier, proposer, summarizer — the
    ScriptedLlm's call order."""
    llm = make_offline_llm()
    install_fake_pipeline(llm, FakeRetriever())
    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    agent_names = ["planner", "researcher", "verifier", "proposer", "summarizer"]
    return dict(zip(agent_names, (call[0] for call in llm.calls)))


@pytest.fixture
def planner_rubric() -> dict[str, str]:
    """The one place the Planner rubric's load-bearing phrases live (prior
    art: the Summarizer's rubric fixture): each test pins its rule through
    this mapping, so a wording edit that preserves intent updates this dict
    and nothing else."""
    return {
        "exclusion gate": "only for regulations the scenario's facts do not exclude",
        "applicability target": "one applicability target per regulation",
        "unconfirmed": "plausibly implicates but does not confirm",
        "perimeter provision": "perimeter provision",
        "exclusion citable": "citable for an exclusion Finding",
        "confirmed regime": "For each confirmed regulation",
        "per duty area": "per operational duty area involved",
        "never merge": "never merge two duty areas into one target",
        "duty path": "classification, continuity, post-incident review, and contract provisions",
        "budget": "1-50",
        "budget priority": "keep every applicability target",
        "budget competition": "compete for the remaining seats",
        "engagement threshold": "Reserve one engagement-threshold target per confirmed regulation",
        "threshold provision": "the provision that decides whether the regime engages at all",
        "threshold notions": "a breach notion, a profiling notion, a financial-entity perimeter",
        "threshold principles": "the principles constraining the contested processing",
        "threshold listed first": "list those reserved targets first",
        "threshold marked reserved": 'marking each "reserved": true',
        "threshold survives truncation": "truncation correction can never discard",
        "regime coverage": "every regulation the Regulatory question or the scenario's facts name",
        "coverage earns": "earns at least one Research target",
        "coverage keyed off both": "both the question and the scenario description",
        "no crowding": "crowd a named regime out",
        "vendor model trigger": "names or implies a generative AI model supplied by a vendor",
        "model provider duty area": "model-provider duty area",
        "vendor duties": "documentation, downstream information, and policies",
        "ai-act marker rule": "a confirmed high-risk AI regime earns the AI-Act marker duty areas",
        "ai-act markers": (
            "the provider-side obligations and the compliance markers the deployer "
            "verifies in its vendor (conformity assessment, CE marking, EU-database registration)"
        ),
        "ai-act post-market": "the post-market monitoring and serious-incident reporting channel",
        "ai-act literacy": "AI literacy for the staff operating or supervising the system",
        "ai-act application date": "the application date that decides whether the duties are in force",
        "gdpr completeness rule": "a confirmed GDPR engagement earns the completeness duty areas",
        "gdpr completeness": (
            "information and access rights, records of processing activities, data "
            "protection by design and by default with security of processing, and accountability"
        ),
        "gdpr breach authority": (
            "a breach scenario adds the notification-authority detail "
            "(which supervisory authority receives the breach notification)"
        ),
        "gdpr breach complaints": (
            "the complaint exposure (the data subjects' complaint right and the authority's corrective powers)"
        ),
        "gdpr transfer trigger": "where inference or storage may sit outside the EEA",
        "gdpr transfer target": "a third-country transfer target",
        "dora machinery rule": "a confirmed DORA engagement earns the reporting-machinery duty areas",
        "dora reporting machinery": "the incident-reporting templates, channels, and the authority's feedback",
        "dora review strategy": "the post-incident review with its communication strategy",
        "dora size status check": (
            "the size and status check that decides between the full and the simplified ICT risk-management framework"
        ),
        "dora proportionality": "the proportionality that scales the duties to the entity",
        "operative only": "(articles, annexes)",
        "exact vocabulary": "exact vocabulary",
        "guidance not constraint": "guidance, not a constraint",
    }


def test_planner_rubric_plans_only_for_regulations_the_scenario_does_not_exclude(live_client, planner_rubric):
    """The Planner lets the Scenario's facts draw the perimeter (ticket #70,
    spec #68): Research targets go only to regulations the scenario's facts
    do not exclude — the full-DORA-for-a-retailer drift dies at planning."""
    planner_system = recorded_system_prompts(live_client)["planner"]
    assert planner_rubric["exclusion gate"] in planner_system
    # The budget discipline survives the rewrite.
    assert planner_rubric["budget"] in planner_system


def test_planner_prompt_points_at_operative_provisions_only(live_client, planner_rubric):
    """Live-mode retrieval never returns a Recital (the store read-path
    exclusion, ticket #69), so the Planner must not be told to locate them —
    its target vocabulary names the operative provisions only."""
    planner_system = recorded_system_prompts(live_client)["planner"]
    assert planner_rubric["operative only"] in planner_system
    assert "recitals" not in planner_system


def test_planner_rubric_spends_the_budget_on_applicability_targets_first(live_client, planner_rubric):
    """Applicability targets and per-duty-area targets compete for the same
    1-50 seats: the rubric settles the competition — every applicability
    target is kept and the confirmed regulations' duty areas take what
    remains — so an unconfirmed regime's exclusion can never be priced out."""
    planner_system = recorded_system_prompts(live_client)["planner"]
    assert planner_rubric["budget priority"] in planner_system
    assert planner_rubric["budget competition"] in planner_system


def test_planner_rubric_reserves_one_engagement_threshold_target_per_confirmed_regulation(live_client, planner_rubric):
    """The reserved engagement-threshold target (issue #75, ADR-0015): each
    confirmed Regulation earns one target aimed at the provision that decides
    whether the regime engages at all — a breach notion, a profiling notion,
    a financial-entity perimeter — listed first so the one-shot truncation
    correction can never discard it (CONTEXT.md, Engagement-threshold
    target)."""
    planner_system = recorded_system_prompts(live_client)["planner"]
    assert planner_rubric["engagement threshold"] in planner_system
    assert planner_rubric["threshold provision"] in planner_system
    assert planner_rubric["threshold notions"] in planner_system
    assert planner_rubric["threshold principles"] in planner_system
    assert planner_rubric["threshold listed first"] in planner_system
    assert planner_rubric["threshold survives truncation"] in planner_system
    # The budget competition keeps every reserved target, too.
    assert "every engagement-threshold target" in planner_system


def test_planner_rubric_marks_each_reserved_target_reserved(live_client, planner_rubric):
    """The reservation is programmatic, not just prose (issue #84): the
    Planner marks each engagement-threshold target `\"reserved\": true`, so the
    application code can find the reserved targets the backstop protects."""
    planner_system = recorded_system_prompts(live_client)["planner"]
    assert planner_rubric["threshold marked reserved"] in planner_system


def test_planner_rubric_enforces_regime_coverage_for_every_named_regulation(live_client, planner_rubric):
    """The regime-coverage rule (issue #75, ADR-0015): every Regulation the
    Regulatory question or the Scenario's facts name earns at least one
    Research target, keyed off both the question and the description the
    Planner already reads — a dominant regime can never crowd a named regime
    out of the plan (the cloud-attack case's zero-data-protection drift dies
    here)."""
    planner_system = recorded_system_prompts(live_client)["planner"]
    assert planner_rubric["regime coverage"] in planner_system
    assert planner_rubric["coverage earns"] in planner_system
    assert planner_rubric["coverage keyed off both"] in planner_system
    assert planner_rubric["no crowding"] in planner_system


def test_planner_rubric_enumerates_the_model_provider_duty_area_for_vendor_models(live_client, planner_rubric):
    """The model-provider enumeration rule (issue #75, ADR-0015): when the
    scenario names or implies a vendor-supplied generative model, the
    model-provider duty area earns its own target — documentation, downstream
    information, policies — so the vendor-side duties are researched as
    deliberately as the deployer's instead of drifting to whichever provider
    chunks retrieval surfaces (ADR-0014's deferred heuristic, adopted)."""
    planner_system = recorded_system_prompts(live_client)["planner"]
    assert planner_rubric["vendor model trigger"] in planner_system
    assert planner_rubric["model provider duty area"] in planner_system
    assert planner_rubric["vendor duties"] in planner_system


def test_planner_rubric_adds_one_applicability_target_per_unconfirmed_regulation(live_client, planner_rubric):
    """A regulation the scenario plausibly implicates but does not confirm
    earns one applicability target for the regime's perimeter provision, so
    that provision is retrieved and an exclusion Finding can cite it."""
    planner_system = recorded_system_prompts(live_client)["planner"]
    assert planner_rubric["applicability target"] in planner_system
    assert planner_rubric["unconfirmed"] in planner_system
    assert planner_rubric["perimeter provision"] in planner_system
    assert planner_rubric["exclusion citable"] in planner_system


def test_planner_rubric_keeps_one_target_per_duty_area_of_a_confirmed_regulation(live_client, planner_rubric):
    """The per-duty-area instruction survives (issue #62), sharpened with the
    confirmed-regime framing: each duty area of a confirmed regulation gets
    its own target — never merged — so the regime's whole duty path is
    researched (the DORA incident path: classification, continuity,
    post-incident review, contract provisions)."""
    planner_system = recorded_system_prompts(live_client)["planner"]
    assert planner_rubric["confirmed regime"] in planner_system
    assert planner_rubric["per duty area"] in planner_system
    assert planner_rubric["never merge"] in planner_system
    assert planner_rubric["duty path"] in planner_system
    # The vocabulary-lifting discipline survives the rewrite.
    assert planner_rubric["exact vocabulary"] in planner_system
    assert planner_rubric["guidance not constraint"] in planner_system


def test_planner_rubric_adds_the_ai_act_marker_duty_areas_for_a_confirmed_high_risk_regime(live_client, planner_rubric):
    """The AI-Act markers rule (issue #87): per confirmed high-risk AI regime,
    the marker duty areas earn their own targets — the provider-side
    obligations and compliance markers, the post-market monitoring and
    serious-incident reporting channel, AI literacy, and the application date
    — so the AI scenarios stop missing the systematic blocks the eval's
    ground truth expects."""
    planner_system = recorded_system_prompts(live_client)["planner"]
    assert planner_rubric["ai-act marker rule"] in planner_system
    assert planner_rubric["ai-act markers"] in planner_system
    assert planner_rubric["ai-act post-market"] in planner_system
    assert planner_rubric["ai-act literacy"] in planner_system
    assert planner_rubric["ai-act application date"] in planner_system


def test_planner_rubric_adds_the_gdpr_completeness_duty_areas_for_a_confirmed_engagement(live_client, planner_rubric):
    """The GDPR completeness rule (issue #87): per confirmed GDPR engagement,
    the mid-weight duties that drop case to case — information and access
    rights, records, DPbDD and security, accountability — earn their own
    targets, with the breach additions (notification-authority detail,
    complaint exposure) and the third-country transfer target where
    inference or storage may sit outside the EEA on top."""
    planner_system = recorded_system_prompts(live_client)["planner"]
    assert planner_rubric["gdpr completeness rule"] in planner_system
    assert planner_rubric["gdpr completeness"] in planner_system
    assert planner_rubric["gdpr breach authority"] in planner_system
    assert planner_rubric["gdpr breach complaints"] in planner_system
    assert planner_rubric["gdpr transfer trigger"] in planner_system
    assert planner_rubric["gdpr transfer target"] in planner_system


def test_planner_rubric_adds_the_dora_machinery_duty_areas_for_a_confirmed_engagement(live_client, planner_rubric):
    """The DORA machinery rule (issue #87): per confirmed DORA engagement, the
    reporting machinery — templates, channels, the authority's feedback — the
    post-incident review with its communication strategy, the size and status
    check, and proportionality earn their own targets, so the template
    provisions stop being missed everywhere."""
    planner_system = recorded_system_prompts(live_client)["planner"]
    assert planner_rubric["dora machinery rule"] in planner_system
    assert planner_rubric["dora reporting machinery"] in planner_system
    assert planner_rubric["dora review strategy"] in planner_system
    assert planner_rubric["dora size status check"] in planner_system
    assert planner_rubric["dora proportionality"] in planner_system


@pytest.fixture
def researcher_rubric() -> dict[str, str]:
    """The Researcher rubric's load-bearing phrases (ticket #70)."""
    return {
        "definite claims": "Draft definite claims",
        "resolve scenario facts": (
            "resolve what the scenario text states about the company's nature, "
            "roles, and jurisdiction into the claim itself"
        ),
        "no if where stated": "'if X' claim where the scenario states X",
        "scenario tie": "question asked for this scenario",
        "no restatement": "never restate a provision's content in the abstract",
        "defined terms": "a personal data breach, profiling, an ICT-related incident, a financial entity",
        "definitional citation": "it cites the provision that defines it",
        "engagement not restatement": "deciding engagement is not restatement in the abstract",
        "pool threshold rule": "When the evidence pool holds a regime's engagement or definitional provision",
        "must draft anchor": "the applicability or engagement claim citing it is never optional filler",
        "no excluded branch": "Never draft a claim whose contingency the scenario's facts exclude",
        "stated facts instead": "draft the claim the stated facts support instead",
        "open stays draftable": "A contingency the scenario text genuinely leaves open is still drafted as today",
        "open visible downstream": "moderate Finding with its referral Actions",
        "vendor verification rule": "the deployer's vendor-verification claims belong in the draft",
        "vendor markers": (
            "the provider's conformity assessment, CE marking, and EU-database registration "
            "are the compliance markers the deployer verifies before adopting the system"
        ),
        "decisive citations": "name only the provisions that decide it",
        "framing rides on framing claims": "rides only when the claim is about that framing itself",
        "no embellishments": "Never pad a claim with cross-reference embellishments",
        "no plumbing": "generic plumbing citations that decide nothing",
        "grounded only": "grounded ONLY in the listed evidence",
        "given labels only": "never reference a label that was not given to you",
    }


def test_researcher_rubric_demands_definite_claims_resolving_the_scenario_facts(live_client, researcher_rubric):
    """The Researcher resolves what the Scenario text states about the
    company's nature, roles, and jurisdiction into the claim itself — the
    claim states the fact, it does not hedge it (ticket #70)."""
    researcher_system = recorded_system_prompts(live_client)["researcher"]
    assert researcher_rubric["definite claims"] in researcher_system
    assert researcher_rubric["resolve scenario facts"] in researcher_system


def test_researcher_rubric_never_drafts_an_if_claim_where_the_scenario_states_it(live_client, researcher_rubric):
    """Never an 'if X' claim where the scenario states X: the conditional
    form of a settled fact is exactly the hedging the eval penalised."""
    researcher_system = recorded_system_prompts(live_client)["researcher"]
    assert researcher_rubric["no if where stated"] in researcher_system
    # The scenario-tie and grounding discipline survive the rewrite.
    assert researcher_rubric["scenario tie"] in researcher_system
    assert researcher_rubric["no restatement"] in researcher_system
    assert researcher_rubric["grounded only"] in researcher_system
    assert researcher_rubric["given labels only"] in researcher_system


def test_researcher_rubric_cites_the_defining_provision_for_a_claim_turning_on_a_defined_term(live_client, researcher_rubric):
    """When a claim turns on a defined term — a personal data breach,
    profiling, an ICT-related incident, a financial entity — it cites the
    provision that defines it, because the 2026-09-03 re-run's largest
    single recall loss was a definitional provision the Answers presupposed
    instead of cited: deciding engagement is not restatement in the abstract
    (issue #76, ADR-0015)."""
    researcher_system = recorded_system_prompts(live_client)["researcher"]
    assert researcher_rubric["defined terms"] in researcher_system
    assert researcher_rubric["definitional citation"] in researcher_system
    assert researcher_rubric["engagement not restatement"] in researcher_system


def test_researcher_rubric_never_drafts_a_branch_the_scenario_facts_exclude(live_client, researcher_rubric):
    """A contingency the Scenario's facts exclude is never drafted — the
    claim the stated facts support goes in its place — while a contingency
    the scenario genuinely leaves open stays draftable exactly as today, so
    the open question stays visible downstream as a moderate Finding with
    its referral Actions (issue #76, ADR-0015)."""
    researcher_system = recorded_system_prompts(live_client)["researcher"]
    assert researcher_rubric["no excluded branch"] in researcher_system
    assert researcher_rubric["stated facts instead"] in researcher_system
    assert researcher_rubric["open stays draftable"] in researcher_system
    assert researcher_rubric["open visible downstream"] in researcher_system


def test_researcher_rubric_must_draft_the_engagement_claim_when_the_pool_holds_the_threshold(live_client, researcher_rubric):
    """The must-draft rule (issue #84): when the Evidence pool holds a regime's
    engagement or definitional provision, drafting the applicability/engagement
    claim citing it is never optional filler — the weight-50 anchor miss (gdpr
    Art 5 in employee-productivity, 68% of that case's recall gap) dies here."""
    researcher_system = recorded_system_prompts(live_client)["researcher"]
    assert researcher_rubric["pool threshold rule"] in researcher_system
    assert researcher_rubric["must draft anchor"] in researcher_system
    # The defined-terms citation rule and the grounding discipline survive.
    assert researcher_rubric["definitional citation"] in researcher_system
    assert researcher_rubric["grounded only"] in researcher_system


def test_researcher_rubric_drafts_the_deployer_vendor_verification_claims(live_client, researcher_rubric):
    """The vendor-verification wording (issue #87): when the scenario has the
    company deploying a provider-supplied system, the deployer's claims about
    verifying the provider's conformity assessment, CE marking, and
    EU-database registration belong in the draft — what the company verifies
    about its vendor, not only the duties it owes itself."""
    researcher_system = recorded_system_prompts(live_client)["researcher"]
    assert researcher_rubric["vendor verification rule"] in researcher_system
    assert researcher_rubric["vendor markers"] in researcher_system
    # The grounding discipline survives the addition.
    assert researcher_rubric["grounded only"] in researcher_system
    assert researcher_rubric["given labels only"] in researcher_system


def test_researcher_rubric_cites_only_the_provisions_that_decide_the_claim(live_client, researcher_rubric):
    """The citation-discipline rule (issue #88): a claim's evidence labels name
    only the provisions that decide it — the ones its truth turns on; a
    framing-only reference rides only when the claim is about the framing
    itself, and cross-reference embellishments, auxiliary elaborations, and
    generic plumbing citations are never padded on. Prompt language only — the
    deterministic per-finding cap stays a contingency."""
    researcher_system = recorded_system_prompts(live_client)["researcher"]
    assert researcher_rubric["decisive citations"] in researcher_system
    assert researcher_rubric["framing rides on framing claims"] in researcher_system
    assert researcher_rubric["no embellishments"] in researcher_system
    assert researcher_rubric["no plumbing"] in researcher_system
    # The rules the discipline must not contradict survive beside it.
    assert researcher_rubric["definitional citation"] in researcher_system
    assert researcher_rubric["must draft anchor"] in researcher_system
    assert researcher_rubric["grounded only"] in researcher_system


@pytest.fixture
def verifier_rubric() -> dict[str, str]:
    """The Verifier rubric's load-bearing phrases (ticket #70)."""
    return {
        "open contingency moderate": "contingent on facts the scenario text leaves open",
        "settled contingency": "whose contingency the scenario text settles",
        "in scope as stated": "supported only as stated",
        "out of scope": "unsupported when they fall out of scope",
        "perimeter form": "perimeter provision",
        "exclusion supported": "does not reach the scenario is the supported form",
        "excluded contingency": "a contingency the scenario's facts exclude",
        "settled the other way": "where the stated facts settle the matter the other way",
        "never moderate": "is unsupported, never moderate",
        "example setup": "a stated private customer base does not make a company a financial entity",
        "example consequence": "developing the full financial-entity regime under that contingency is unsupported",
        "within-regime example": "The same test works within one regime",
        "qualifying use unsupported": "a qualifying use the scenario never describes is unsupported",
        "open classification moderate": "whether the described use falls within an Annex III category at all",
        "supported bar": "bears on answering the question asked for this scenario",
        "strong": "directly and explicitly establish the claim",
        "weak": "framing only (definitions, vocabulary)",
        "bare string": "never an object or rationale",
        "refs decisive": "list only the provisions that decide the claim",
        "refs framing rides": "rides only when the claim is about that framing itself",
        "refs drop embellishments": (
            "cross-reference embellishments, auxiliary elaborations, and generic plumbing citations"
        ),
        "refs decisive survive": "while the decisive citations",
    }


def test_verifier_rubric_keeps_scenario_open_contingencies_eligible_for_moderate(live_client, verifier_rubric):
    """A claim contingent on facts the Scenario text leaves open stays
    eligible for 'moderate' — only a professional can settle those facts
    (CONTEXT.md, Strength: moderate)."""
    verifier_system = recorded_system_prompts(live_client)["verifier"]
    assert verifier_rubric["open contingency moderate"] in verifier_system


def test_verifier_rubric_decides_a_settled_contingency_as_in_scope_or_out(live_client, verifier_rubric):
    """A claim whose contingency the Scenario text settles is judged on the
    stated facts alone: supported only as stated (in scope) or unsupported
    (out of scope) — the conditional drift dies at this gate."""
    verifier_system = recorded_system_prompts(live_client)["verifier"]
    assert verifier_rubric["settled contingency"] in verifier_system
    assert verifier_rubric["in scope as stated"] in verifier_system
    assert verifier_rubric["out of scope"] in verifier_system


def test_verifier_rubric_names_the_exclusion_claim_as_the_supported_form(live_client, verifier_rubric):
    """The exclusion Finding is the supported form where the perimeter
    provision is in Evidence: a claim that the regime does not reach the
    scenario, citing who the regime covers, is kept — never discarded."""
    verifier_system = recorded_system_prompts(live_client)["verifier"]
    assert verifier_rubric["perimeter form"] in verifier_system
    assert verifier_rubric["exclusion supported"] in verifier_system
    # The supported bar and the bare-string contract survive the rewrite.
    assert verifier_rubric["supported bar"] in verifier_system
    assert verifier_rubric["strong"] in verifier_system
    assert verifier_rubric["weak"] in verifier_system
    assert verifier_rubric["bare string"] in verifier_system


def test_verifier_rubric_makes_the_scenario_excluded_contingency_explicitly_unsupported(live_client, verifier_rubric):
    """A contingency the Scenario's facts exclude — where the stated facts
    settle the matter the other way — is unsupported, never moderate: the
    explicit case the glossary's Unsupported-claim definition already states,
    anchored by one abstract worked example deliberately not drawn from the
    eval cases (issue #77, ADR-0015)."""
    verifier_system = recorded_system_prompts(live_client)["verifier"]
    assert verifier_rubric["excluded contingency"] in verifier_system
    assert verifier_rubric["settled the other way"] in verifier_system
    assert verifier_rubric["never moderate"] in verifier_system
    assert verifier_rubric["example setup"] in verifier_system
    assert verifier_rubric["example consequence"] in verifier_system


def test_verifier_rubric_gains_the_worked_within_regime_annex_iii_example(live_client, verifier_rubric):
    """The within-regime worked example (issue #86), both directions: the
    high-risk classification provisions support a claim only when the
    Scenario describes the qualifying use — 'could be high-risk if it did X'
    limbs are unsupported when the Scenario never describes X — while a
    contingency the text genuinely leaves open stays moderate."""
    verifier_system = recorded_system_prompts(live_client)["verifier"]
    assert verifier_rubric["within-regime example"] in verifier_system
    assert verifier_rubric["qualifying use unsupported"] in verifier_system
    assert verifier_rubric["open classification moderate"] in verifier_system
    assert "Annex III" in verifier_system
    # The cross-regime example and the strength contract survive the addition.
    assert verifier_rubric["example setup"] in verifier_system
    assert verifier_rubric["exclusion supported"] in verifier_system


def test_verifier_rubric_keeps_only_the_decisive_provisions_in_evidence_refs(live_client, verifier_rubric):
    """The citation-discipline rule on the refs the Verifier keeps (issue #88):
    evidence_refs names only the provisions that decide the claim — the ones
    its truth turns on; a framing-only reference rides only when the claim is
    about the framing itself, and cross-reference embellishments, auxiliary
    elaborations, and generic plumbing citations are dropped while the
    decisive citations — and every moderate or strong anchor — stay."""
    verifier_system = recorded_system_prompts(live_client)["verifier"]
    assert verifier_rubric["refs decisive"] in verifier_system
    assert verifier_rubric["refs framing rides"] in verifier_system
    assert verifier_rubric["refs drop embellishments"] in verifier_system
    assert verifier_rubric["refs decisive survive"] in verifier_system
    # The supported bar and the strength contract survive the addition.
    assert verifier_rubric["supported bar"] in verifier_system
    assert verifier_rubric["bare string"] in verifier_system


# --- Reserved-anchor backstop: trace + one corrective re-prompt (issue #84) ----


def claims_calls(llm) -> list[tuple[str, str, type]]:
    """The DraftClaims boundary's calls in the scripted client's log."""
    return [call for call in llm.calls if call[2] is DraftClaims]


def researcher_step(data) -> dict:
    """The Researcher's detailed-trace step."""
    return [s for s in data["detailed_trace"] if s["step"] == "researcher"][0]


PRINCIPLES_CHUNK = make_chunk(
    source_id="gdpr",
    number=5,
    text="Processing must be lawful, fair and transparent; data minimisation applies.",
    title="Principles relating to processing",
)
NOTIFICATION_CHUNK = make_chunk(
    source_id="gdpr",
    number=33,
    text="Notification of a personal data breach to the supervisory authority.",
    title="Notification of a breach",
)
GDPR_DEFINITIONS_CHUNK = make_chunk(
    source_id="gdpr",
    number=4,
    text="For the purposes of this Regulation, the definitions apply.",
    title="Definitions",
)
ANCHOR_RETRIEVALS = {
    "principles relating to processing": [PRINCIPLES_CHUNK],
    "breach notification duties": [NOTIFICATION_CHUNK],
}
# The reserved target's retrieval seats the threshold provision first and a
# neighbouring chunk behind it — the shape the union check could not see past.
MIXED_RETRIEVALS = {
    "principles relating to processing": [PRINCIPLES_CHUNK, GDPR_DEFINITIONS_CHUNK],
    "breach notification duties": [NOTIFICATION_CHUNK],
}


def anchor_llm() -> "ScriptedLlm":
    """A scripted run whose reserved engagement-threshold target retrieves the
    principles provision the first pass never cites — the employee-productivity
    case's weight-50 gdpr Article 5 miss. The backstop's corrective re-prompt
    drafts the engagement claim, and its verification keeps it."""
    from src.live_workflow import Plan, ResearchTarget

    return ScriptedLlm(
        plan=Plan(targets=[
            ResearchTarget(query="principles relating to processing", reserved=True),
            ResearchTarget(query="breach notification duties"),
        ]),
        claims=DraftClaims(claims=[
            DraftClaim(
                statement="The breach must be notified to the authority without undue delay.",
                evidence_refs=["E2"],
            ),
        ]),
        verdicts=Verdicts(verdicts=[
            grounded_verdict(
                "The breach must be notified to the authority without undue delay.",
                Strength.moderate,
                ["E2"],
            ),
        ]),
        claims_retry=DraftClaims(claims=[
            DraftClaim(
                statement="The processing of the employee data must respect the data-protection principles.",
                evidence_refs=["E1"],
            ),
        ]),
        verdicts_retry=Verdicts(verdicts=[
            grounded_verdict(
                "The processing of the employee data must respect the data-protection principles.",
                Strength.strong,
                ["E1"],
            ),
        ]),
        proposals=ActionProposals(proposals=[
            ActionProposal(
                action="Verify the processing against the principles the company must respect.",
                kind="verify_against_facts",
                citation_refs=["C2"],
            ),
        ]),
        summaries=Summaries(summaries=[
            ProvisionSummary(
                ref="P1",
                relevance="The notification duty is what the question turns on.",
                strength=Strength.moderate,
            ),
            ProvisionSummary(
                ref="P2",
                relevance=(
                    "The principles provision decides whether the processing is lawful "
                    "at all — the threshold the Answer cites."
                ),
                strength=Strength.strong,
            ),
        ]),
    )


def test_uncited_reserved_anchor_is_recorded_and_one_corrective_re_prompt_lands_it(live_client):
    """The weight-50 anchor lands (issue #84): a first pass that never cites the
    reserved engagement-threshold evidence earns exactly one corrective
    re-prompt naming that Evidence; the second pass's engagement claim is
    verified, kept, and the Answer cites the threshold provision with its
    rated strength."""
    llm = anchor_llm()
    install_fake_pipeline(llm, FakeRetriever(per_query=ANCHOR_RETRIEVALS))

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    # Exactly one corrective re-prompt: two DraftClaims calls, never three.
    assert len(claims_calls(llm)) == 2
    # The re-prompt repeats the grounding material and names the uncited anchor.
    _, retry_user, _ = claims_calls(llm)[1]
    assert "E1" in retry_user, "the anchor's Evidence label is named"
    assert "principles relating to processing" in retry_user
    assert "Article 5" in retry_user
    # The correction is recorded in the detailed trace, naming the anchor.
    step = researcher_step(data)
    assert "re-prompt" in step["correction"]
    assert "Article 5" in step["correction"]
    # The anchor lands: the strong engagement Finding cites gdpr Article 5,
    # and the Answer's citation carries the rated weight-50 strength.
    strong = next(f for f in data["answer"]["findings"] if f["strength"] == "strong")
    assert strong["citations"][0]["source_id"] == "gdpr"
    assert strong["citations"][0]["article_number"] == PRINCIPLES_CHUNK.article_number
    by_number = {c["article_number"]: c for c in data["answer"]["citations"]}
    assert by_number[PRINCIPLES_CHUNK.article_number]["strength"] == "strong"
    assert by_number[PRINCIPLES_CHUNK.article_number]["relevance"]
    # The first pass's Finding still ships alongside it.
    assert by_number[NOTIFICATION_CHUNK.article_number]["relevance"]


def test_a_cited_reserved_anchor_earns_no_corrective_re_prompt(live_client):
    """When the first pass already cites the reserved engagement-threshold
    evidence, the backstop stays silent: one DraftClaims call, and no
    correction is recorded (issue #84)."""
    llm = anchor_llm()
    llm.claims = DraftClaims(claims=[
        DraftClaim(
            statement="The processing of the employee data must respect the data-protection principles.",
            evidence_refs=["E1"],
        ),
    ])
    llm.verdicts = Verdicts(verdicts=[
        grounded_verdict(
            "The processing of the employee data must respect the data-protection principles.",
            Strength.strong,
            ["E1"],
        ),
    ])
    install_fake_pipeline(llm, FakeRetriever(per_query=ANCHOR_RETRIEVALS))

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    assert len(claims_calls(llm)) == 1
    assert researcher_step(data).get("correction") is None


def test_a_failing_anchor_re_prompt_still_ships_the_first_pass(live_client):
    """The corrective re-prompt's call itself fails: the first pass's Findings
    ship unchanged with the correction recorded — the backstop never fails
    the run (issue #84)."""
    base = anchor_llm()

    class RetryFailsLlm:
        """Serves the first pass, then dies at the corrective re-prompt."""

        def __init__(self, inner):
            self._inner = inner
            self.attempts = 0

        def complete(self, system, user, schema):
            if schema is DraftClaims:
                self.attempts += 1
                if self.attempts == 2:
                    raise LlmError("OpenRouter rejected the request (HTTP 429)")
            return self._inner.complete(system, user, schema)

    llm = RetryFailsLlm(base)
    install_fake_pipeline(llm, FakeRetriever(per_query=ANCHOR_RETRIEVALS))

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    assert llm.attempts == 2
    # The first pass's Finding ships; no principles Finding appears.
    numbers = [c["article_number"] for c in data["answer"]["citations"]]
    assert NOTIFICATION_CHUNK.article_number in numbers
    assert PRINCIPLES_CHUNK.article_number not in numbers
    assert "re-prompt" in researcher_step(data)["correction"]


def test_a_neighbour_citation_does_not_silence_the_uncited_anchor(live_client):
    """The union hole (issue #84): a first pass that cites only the reserved
    target's neighbouring chunk — not the threshold provision its retrieval
    ranked first — still owes the corrective re-prompt, and the weight-50
    anchor lands through it."""
    llm = anchor_llm()
    llm.claims = DraftClaims(claims=[
        DraftClaim(
            statement="The definitions article applies to the processing.",
            evidence_refs=["E2"],
        ),
    ])
    llm.verdicts = Verdicts(verdicts=[
        grounded_verdict(
            "The definitions article applies to the processing.",
            Strength.moderate,
            ["E2"],
        ),
    ])
    install_fake_pipeline(llm, FakeRetriever(per_query=MIXED_RETRIEVALS))

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    # The backstop fired: exactly one corrective re-prompt.
    assert len(claims_calls(llm)) == 2
    _, retry_user, _ = claims_calls(llm)[1]
    assert "E1" in retry_user, "the threshold provision's label is named"
    assert "Article 5" in retry_user
    # The weight-50 anchor lands: the strong engagement Finding cites it.
    strong = next(f for f in data["answer"]["findings"] if f["strength"] == "strong")
    assert strong["citations"][0]["source_id"] == "gdpr"
    assert strong["citations"][0]["article_number"] == PRINCIPLES_CHUNK.article_number
    assert "Article 5" in researcher_step(data)["correction"]


def test_the_ranked_first_provision_cited_earns_no_re_prompt(live_client):
    """The anchor lands the moment a kept Finding cites the top-ranked
    provision: a neighbour's absence never earns a re-prompt (issue #84)."""
    llm = anchor_llm()
    llm.claims = DraftClaims(claims=[
        DraftClaim(
            statement="The processing of the employee data must respect the data-protection principles.",
            evidence_refs=["E1"],
        ),
    ])
    llm.verdicts = Verdicts(verdicts=[
        grounded_verdict(
            "The processing of the employee data must respect the data-protection principles.",
            Strength.strong,
            ["E1"],
        ),
    ])
    install_fake_pipeline(llm, FakeRetriever(per_query=MIXED_RETRIEVALS))

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    assert len(claims_calls(llm)) == 1
    assert researcher_step(data).get("correction") is None


# --- The engagement gate: closed-wins over the kept set (issue #86) ------------


def verifier_step(data) -> dict:
    """The Verifier's detailed-trace step — where the claim decisions and the
    engagement gate's per-Regulation record live."""
    return [s for s in data["detailed_trace"] if s["step"] == "verifier"][0]


DORA_SCOPE_CHUNK = make_chunk(
    source_id="dora",
    number=2,
    text="This Regulation applies to financial entities as defined in Article 3.",
    title="Scope",
)
DORA_MANAGEMENT_CHUNK = make_chunk(
    source_id="dora",
    number=5,
    text="The management body of the financial entity shall bear ultimate responsibility for ICT risk.",
    title="Governance and organisation",
)
DORA_FRAMEWORK_CHUNK = make_chunk(
    source_id="dora",
    number=6,
    text="Financial entities shall have in place a sound, comprehensive ICT risk-management framework.",
    title="ICT risk management framework",
)
DORA_REPORTING_CHUNK = make_chunk(
    source_id="dora",
    number=19,
    text="Financial entities shall classify and report major ICT-related incidents.",
    title="Reporting of major ICT-related incidents",
)
GDPR_SECURITY_CHUNK = make_chunk(
    source_id="gdpr",
    number=32,
    text="The controller shall implement appropriate technical and organisational measures.",
    title="Security of processing",
)


def gate_llm(*claim_pairs: tuple[str, "str | list[str]"], strength: Strength = Strength.moderate) -> "ScriptedLlm":
    """A scripted run over the given (statement, evidence-labels) pairs, each
    verified supported at one uniform Strength — the raw material the
    engagement gate reads. The duty-blocks' ``duty_llm`` with the strength
    pinned across the claims."""
    return duty_llm(*((statement, refs, strength) for statement, refs in claim_pairs))


RETAILER_RETRIEVALS = {
    "DORA applicability financial entities": [DORA_SCOPE_CHUNK],
    "ICT incident reporting": [DORA_REPORTING_CHUNK],
    "security of processing": [GDPR_SECURITY_CHUNK],
}


def post_gate_scenario(live_client) -> dict:
    """One retailer-flavoured analysis run: an online shop asking whether
    DORA's ICT incident duties bind it."""
    resp = live_client.post(
        "/api/analyze",
        json={
            "scenario": {
                "id": "electronics-shop",
                "description": "An online shop selling consumer electronics from Madrid",
            },
            "question": "Do DORA's ICT incident duties apply to the company?",
        },
    )
    assert resp.status_code == 200
    return resp.json()


def test_duty_area_findings_scoped_only_to_a_closed_regulation_are_dropped_with_recorded_reasons(live_client):
    """The engagement gate (issue #86): the kept Exclusion Finding citing
    DORA's perimeter provision settles the regime away, so the duty-area
    Finding scoped only to DORA is rejected with a recorded reason — the
    Perimeter Citation surfaces as the Exclusion Finding, the expected form,
    and the finding scoped to the undecided Regulation is untouched."""
    from src.live_workflow import Plan, ResearchTarget

    exclusion = (
        "DORA's incident regime covers financial entities only, and the company is an "
        "online shop rather than a financial entity, so the regime does not reach it."
    )
    duty = "The company must report major ICT-related incidents under DORA's reporting regime."
    other_regime = "The company must implement security measures appropriate to the breach risk."
    llm = gate_llm((exclusion, "E1"), (duty, "E2"), (other_regime, "E3"))
    llm.plan = Plan(targets=[
        ResearchTarget(query="DORA applicability financial entities"),
        ResearchTarget(query="ICT incident reporting"),
        ResearchTarget(query="security of processing"),
    ])
    install_fake_pipeline(llm, FakeRetriever(per_query=RETAILER_RETRIEVALS))

    data = post_gate_scenario(live_client)

    statements = [f["statement"] for f in data["answer"]["findings"]]
    assert exclusion in statements, "the Exclusion Finding carries the Perimeter Citation"
    exclusion_finding = next(f for f in data["answer"]["findings"] if f["statement"] == exclusion)
    assert [(c["source_id"], c["article_number"]) for c in exclusion_finding["citations"]] == [("dora", DORA_SCOPE_CHUNK.article_number)]
    assert other_regime in statements, "a finding scoped to an undecided Regulation is untouched"
    assert duty not in statements, "the duty limb scoped only to the closed regime is dropped"

    assert duty in data["trace"]["unsupported_claims_discarded"]

    decisions = {d["claim"]: d for d in verifier_step(data)["claim_decisions"]}
    assert decisions[duty]["status"] == "rejected"
    assert "DORA Article 2" in decisions[duty]["reason"]
    assert "not reaching the scenario" in decisions[duty]["reason"]
    assert decisions[exclusion]["status"] == "kept"
    assert decisions[other_regime]["status"] == "kept"

    gate_records = verifier_step(data)["engagement_states"]
    assert gate_records == [{
        "source_id": "dora",
        "state": "closed",
        "conflict": False,
        "open": [],
        "closed": [exclusion],
    }]


def test_conflicting_engagement_evidence_resolves_closed_wins_and_is_recorded(live_client):
    """When the kept set holds both an exclusion Finding and an open-form
    engagement Finding citing the same perimeter provision, closed-wins wins
    and the conflict is recorded in the detailed trace — the duty-area
    Finding scoped only to the closed regime still drops (issue #86)."""
    from src.live_workflow import Plan, ResearchTarget

    exclusion = (
        "DORA's incident regime covers financial entities only, and the company is an "
        "online shop, so the regime does not reach it."
    )
    open_question = (
        "Whether the company's payment arm makes it a financial entity under DORA's "
        "perimeter is an open question the scenario leaves undecided."
    )
    duty = "The company must report major ICT-related incidents under DORA's reporting regime."
    llm = gate_llm((exclusion, "E1"), (open_question, "E1"), (duty, "E2"))
    llm.plan = Plan(targets=[
        ResearchTarget(query="DORA applicability financial entities"),
        ResearchTarget(query="ICT incident reporting"),
    ])
    install_fake_pipeline(llm, FakeRetriever(per_query={
        "DORA applicability financial entities": [DORA_SCOPE_CHUNK],
        "ICT incident reporting": [DORA_REPORTING_CHUNK],
    }))

    data = post_gate_scenario(live_client)

    statements = [f["statement"] for f in data["answer"]["findings"]]
    assert exclusion in statements
    assert open_question in statements, "both perimeter-citing Findings survive the closed-wins call"
    assert duty not in statements

    gate_records = verifier_step(data)["engagement_states"]
    assert gate_records == [{
        "source_id": "dora",
        "state": "closed",
        "conflict": True,
        "open": [open_question],
        "closed": [exclusion],
    }]
    decisions = {d["claim"]: d for d in verifier_step(data)["claim_decisions"]}
    assert "closed-wins" in decisions[duty]["reason"]


def test_the_conditional_dora_block_survives_the_gate(live_client):
    """The fintech tripwire (issue #86): DORA's entity-status question is
    genuinely open, the engagement Finding cites the perimeter provision as
    an open question, so the conditional duty Findings scoped only to DORA
    survive the gate untouched."""
    from src.live_workflow import Plan, ResearchTarget

    engagement = (
        "Whether the fintech's authorization brings it within DORA's financial-entity "
        "perimeter is an open question; if it does, its management body becomes "
        "ultimately responsible for ICT risk for the loan platform."
    )
    duty_five = (
        "Once it qualifies as a financial entity, the fintech's management body must "
        "bear ultimate responsibility for ICT risk under DORA's management-body rule."
    )
    duty_six = (
        "On the same condition the fintech must operate a documented ICT "
        "risk-management framework for the loan platform."
    )
    llm = gate_llm((engagement, "E1"), (duty_five, "E2"), (duty_six, "E3"))
    llm.plan = Plan(targets=[
        ResearchTarget(query="DORA applicability financial entities"),
        ResearchTarget(query="governance and organisation"),
        ResearchTarget(query="ICT risk management framework"),
    ])
    install_fake_pipeline(llm, FakeRetriever(per_query={
        "DORA applicability financial entities": [DORA_SCOPE_CHUNK],
        "governance and organisation": [DORA_MANAGEMENT_CHUNK],
        "ICT risk management framework": [DORA_FRAMEWORK_CHUNK],
    }))

    resp = live_client.post(
        "/api/analyze",
        json={
            "scenario": {
                "id": "fintech-loans",
                "description": "A Spanish fintech startup operating a loan-recommendation platform",
            },
            "question": "What ICT risk duties bind the loan platform?",
        },
    )
    assert resp.status_code == 200
    data = resp.json()

    statements = [f["statement"] for f in data["answer"]["findings"]]
    assert engagement in statements
    assert duty_five in statements, "the conditional DORA block survives the gate"
    assert duty_six in statements
    assert data["trace"]["unsupported_claims_discarded"] == []

    gate_records = verifier_step(data)["engagement_states"]
    assert gate_records == [{
        "source_id": "dora",
        "state": "open",
        "conflict": False,
        "open": [engagement],
        "closed": [],
    }]


def test_a_regulation_with_no_kept_perimeter_citation_is_undecided_and_untouched(live_client):
    """No kept Finding cites DORA's perimeter provision, so its engagement
    stays undecided: duty-area Findings scoped only to it ship untouched, and
    the detailed trace records no engagement state at all (issue #86)."""
    from src.live_workflow import Plan, ResearchTarget

    duty = "The company must report major ICT-related incidents under DORA's reporting regime."
    security = "The company must implement security measures appropriate to the breach risk."
    llm = gate_llm((duty, "E1"), (security, "E2"))
    llm.plan = Plan(targets=[
        ResearchTarget(query="ICT incident reporting"),
        ResearchTarget(query="security of processing"),
    ])
    install_fake_pipeline(llm, FakeRetriever(per_query={
        "ICT incident reporting": [DORA_REPORTING_CHUNK],
        "security of processing": [GDPR_SECURITY_CHUNK],
    }))

    data = post_gate_scenario(live_client)

    statements = [f["statement"] for f in data["answer"]["findings"]]
    assert duty in statements
    assert security in statements
    assert data["trace"]["unsupported_claims_discarded"] == []
    assert "engagement_states" not in verifier_step(data)


def test_a_finding_scoped_across_an_undecided_and_a_closed_regulation_survives(live_client):
    """A Finding citing a closed Regulation and an undecided one is not
    scoped only to the closed one: it survives the gate with its mixed
    citations intact (issue #86)."""
    from src.live_workflow import Plan, ResearchTarget

    exclusion = (
        "DORA's incident regime covers financial entities only, and the company is an "
        "online shop, so the regime does not reach it."
    )
    mixed = (
        "The company must secure the breached data and classify and report the "
        "incident under the security and reporting provisions read together."
    )
    llm = gate_llm((exclusion, "E1"), (mixed, ["E2", "E3"]))
    llm.plan = Plan(targets=[
        ResearchTarget(query="DORA applicability financial entities"),
        ResearchTarget(query="security and reporting duties"),
    ])
    install_fake_pipeline(llm, FakeRetriever(per_query={
        "DORA applicability financial entities": [DORA_SCOPE_CHUNK],
        "security and reporting duties": [GDPR_SECURITY_CHUNK, DORA_REPORTING_CHUNK],
    }))

    data = post_gate_scenario(live_client)

    statements = [f["statement"] for f in data["answer"]["findings"]]
    assert exclusion in statements
    assert mixed in statements, "a finding not scoped only to the closed regime survives"


def test_duty_limbs_on_an_ai_act_regime_settled_away_are_dropped(live_client):
    """The ransomware/telecom-style pathology across regimes (issue #86): the
    gate is regulation-agnostic — the kept exclusion Finding citing the AI
    Act's perimeter provision settles that regime away, so the duty-area
    Finding scoped only to the AI Act drops with its recorded reason while
    the undecided regime's finding survives."""
    from src.live_workflow import Plan, ResearchTarget

    exclusion = (
        "The company deploys no AI system at all, so the AI Act does not reach "
        "its operations."
    )
    duty = "The provider must register the system and pass the conformity assessment."
    other_regime = "The company must notify the personal data breach to the authority."
    llm = gate_llm((exclusion, "E1"), (duty, "E2"), (other_regime, "E3"))
    llm.plan = Plan(targets=[
        ResearchTarget(query="AI Act applicability providers"),
        ResearchTarget(query="provider registration duties"),
        ResearchTarget(query="breach notification duties"),
    ])
    install_fake_pipeline(llm, FakeRetriever(per_query={
        "AI Act applicability providers": [make_chunk(
            source_id="ai-act", number=2,
            text="This Regulation applies to providers placing on the market or putting into service AI systems.",
            title="Scope",
        )],
        "provider registration duties": [make_chunk(
            source_id="ai-act", number=49,
            text="Providers shall register high-risk AI systems in the EU database.",
            title="Registration",
        )],
        "breach notification duties": [GDPR_SECURITY_CHUNK],
    }))

    data = post_gate_scenario(live_client)

    statements = [f["statement"] for f in data["answer"]["findings"]]
    assert exclusion in statements
    assert duty not in statements, "the AI-Act duty limb on the settled-away regime drops"
    assert other_regime in statements

    decisions = {d["claim"]: d for d in verifier_step(data)["claim_decisions"]}
    assert decisions[duty]["status"] == "rejected"
    assert "EU AI Act Article 2" in decisions[duty]["reason"]
    gate_records = verifier_step(data)["engagement_states"]
    assert [r["state"] for r in gate_records if r["source_id"] == "ai-act"] == ["closed"]


# --- The duty-area blocks: additive targets for confirmed regimes (issue #87) ---


def duty_llm(*claim_triples: tuple[str, "str | list[str]", Strength]) -> "ScriptedLlm":
    """A scripted run over (statement, evidence-labels, strength) triples, each
    verified supported — the duty-area blocks' raw material, the engagement
    gate's ``gate_llm`` with a per-claim Strength."""
    from src.live_workflow import Verdict

    claims = [
        DraftClaim(statement=statement, evidence_refs=refs if isinstance(refs, list) else [refs])
        for statement, refs, _ in claim_triples
    ]
    verdicts = [
        Verdict(
            statement=statement,
            supported=True,
            strength=strength,
            decisive_refs=refs if isinstance(refs, list) else [refs],
        )
        for statement, refs, strength in claim_triples
    ]
    llm = make_offline_llm()
    llm.claims = DraftClaims(claims=claims)
    llm.verdicts = Verdicts(verdicts=verdicts)
    llm.proposals = ActionProposals(proposals=[])
    return llm


def plan_with_reserved_first(targets: list[str]) -> Plan:
    """The duty-block plans' shape: the first target is the reserved
    engagement-threshold one, the rest the block's duty areas in plan order."""
    return Plan(targets=[
        ResearchTarget(query=targets[0], reserved=True),
        *(ResearchTarget(query=query) for query in targets[1:]),
    ])


def script_summaries(llm: "ScriptedLlm", relevance: list[str]) -> None:
    """Script one Provision relevance per cited provision (P1..Pn in the
    Summarizer's first-mention order), every rating 'weak' — the blocks'
    tests assert landing, not ratings — so a duty-block run needs no
    Summarizer corrective re-prompt and the five-call script stays exact."""
    llm.summaries = Summaries(summaries=[
        ProvisionSummary(ref=f"P{index}", relevance=text, strength=Strength.weak)
        for index, text in enumerate(relevance, 1)
    ])


AI_CLASSIFICATION_CHUNK = make_chunk(
    source_id="ai-act", number=6,
    text="AI systems referred to in Annex III shall be considered high-risk AI systems.",
    title="Classification rules for high-risk AI systems",
)
AI_PROVIDER_MARKERS_CHUNK = make_chunk(
    source_id="ai-act", number=16,
    text="Providers of high-risk AI systems shall have the system undergo the conformity assessment, affix the CE marking, and register it in the EU database.",
    title="Obligations of providers",
)
AI_POST_MARKET_CHUNK = make_chunk(
    source_id="ai-act", number=72,
    text="Providers shall establish a post-market monitoring system and report serious incidents.",
    title="Post-market monitoring by providers",
)
AI_LITERACY_CHUNK = make_chunk(
    source_id="ai-act", number=4,
    text="Providers and deployers of AI systems shall ensure a sufficient level of AI literacy of their staff.",
    title="AI literacy",
)
AI_APPLICATION_DATE_CHUNK = make_chunk(
    source_id="ai-act", number=113,
    text="Chapters III and IV apply from 2 August 2026.",
    title="Application of this Regulation",
)

# The AI-Act markers block's targets (issue #87): the reserved
# engagement-threshold target plus one target per marker duty area, each
# retrieving the provision that carries it. The keys are the plan's targets
# in order, the reserved engagement-threshold target first.
RECRUITMENT_RETRIEVALS = {
    "high-risk classification": [AI_CLASSIFICATION_CHUNK],
    "provider obligations and compliance markers": [AI_PROVIDER_MARKERS_CHUNK],
    "post-market monitoring and serious-incident reporting": [AI_POST_MARKET_CHUNK],
    "AI literacy": [AI_LITERACY_CHUNK],
    "application date": [AI_APPLICATION_DATE_CHUNK],
}


def test_the_ai_act_marker_block_lands_for_a_confirmed_high_risk_regime(live_client):
    """The AI-Act markers block (issue #87): a confirmed high-risk AI regime's
    plan carries the marker duty areas as their own targets, and the Answer
    carries a Finding for each — the engagement claim, the provider-side
    compliance markers, the post-market monitoring and serious-incident
    reporting channel, AI literacy, and the application date — with the AI
    Act's engagement open, so nothing is touched by the gate."""
    engagement = (
        "The CV-screening system is a high-risk AI system, so the AI Act's full "
        "Chapter III regime applies to its deployment alongside the GDPR."
    )
    markers = (
        "Before adopting the screener the company verifies the provider's conformity "
        "assessment, CE marking, and EU-database registration."
    )
    post_market = (
        "After deployment the provider keeps post-market monitoring of the screener, "
        "and the company reports serious incidents to the provider."
    )
    literacy = "The staff operating and supervising the screener must have sufficient AI literacy."
    date = "The screener's high-risk deployer duties apply from 2 August 2026."
    llm = duty_llm(
        (engagement, "E1", Strength.moderate),
        (markers, "E2", Strength.moderate),
        (post_market, "E3", Strength.weak),
        (literacy, "E4", Strength.weak),
        (date, "E5", Strength.weak),
    )
    llm.plan = plan_with_reserved_first(list(RECRUITMENT_RETRIEVALS))
    script_summaries(llm, [
        "Article 6 classifies the screening system as high-risk, which engages the Chapter III regime the other findings develop.",
        "Article 16's provider duties define the compliance markers of conformity assessment, CE marking, and registration the deployer verifies.",
        "Article 72 keeps the provider monitoring the system after deployment, the channel through which the deployer's serious-incident reports flow.",
        "Article 4 obliges the deployer to ensure its staff have sufficient AI literacy to oversee the screener.",
        "Article 113 fixes the date the high-risk duties apply from, confirming they are in force at the time of use.",
    ])
    retriever = FakeRetriever(per_query=RECRUITMENT_RETRIEVALS)
    install_fake_pipeline(llm, retriever)

    resp = live_client.post(
        "/api/analyze",
        json={
            "scenario": {
                "id": "hr-ai-cv-screening",
                "description": "A company uses an AI system to screen job applications, scoring each applicant's CV and recorded video interview so recruiters can prioritise candidates.",
            },
            "question": "What regulatory requirements should the company consider before using this AI system for recruitment?",
        },
    )
    assert resp.status_code == 200
    data = resp.json()

    # The block's targets land: every planned duty-area target reached retrieval.
    assert retriever.queries == list(RECRUITMENT_RETRIEVALS)

    statements = [f["statement"] for f in data["answer"]["findings"]]
    for statement in (engagement, markers, post_market, literacy, date):
        assert statement in statements, "the marker block's finding lands in the Answer"

    by_statement = {f["statement"]: f for f in data["answer"]["findings"]}
    assert [(c["source_id"], c["article_number"]) for c in by_statement[engagement]["citations"]] == [("ai-act", 6)]
    assert [(c["source_id"], c["article_number"]) for c in by_statement[markers]["citations"]] == [("ai-act", 16)]
    assert [(c["source_id"], c["article_number"]) for c in by_statement[post_market]["citations"]] == [("ai-act", 72)]
    assert [(c["source_id"], c["article_number"]) for c in by_statement[literacy]["citations"]] == [("ai-act", 4)]
    assert [(c["source_id"], c["article_number"]) for c in by_statement[date]["citations"]] == [("ai-act", 113)]

    assert data["trace"]["unsupported_claims_discarded"] == []
    assert len(llm.calls) == 5, "the block lands on its own: no corrective re-prompt fired"
    assert verifier_step(data)["engagement_states"] == [{
        "source_id": "ai-act",
        "state": "open",
        "conflict": False,
        "open": [engagement],
        "closed": [],
    }]


GDPR_INFORMATION_CHUNK = make_chunk(
    source_id="gdpr", number=13,
    text="The controller shall inform the data subject of the processing's purposes, legal basis, retention and rights.",
    title="Information to be provided where personal data are collected",
)
GDPR_RECORDS_CHUNK = make_chunk(
    source_id="gdpr", number=30,
    text="Each controller shall maintain records of processing activities under its responsibility.",
    title="Records of processing activities",
)
GDPR_DPBYD_CHUNK = make_chunk(
    source_id="gdpr", number=25,
    text="The controller shall implement data protection by design and by default.",
    title="Data protection by design and by default",
)
GDPR_ACCOUNTABILITY_CHUNK = make_chunk(
    source_id="gdpr", number=24,
    text="The controller shall implement measures capable of demonstrating that processing complies with the Regulation.",
    title="Responsibility of the controller",
)
GDPR_COMPETENCE_CHUNK = make_chunk(
    source_id="gdpr", number=55,
    text="Each supervisory authority is competent for the territory of its own Member State.",
    title="Competence",
)
GDPR_COMPLAINT_CHUNK = make_chunk(
    source_id="gdpr", number=77,
    text="Every data subject has the right to lodge a complaint with a supervisory authority.",
    title="Right to lodge a complaint",
)
GDPR_TRANSFER_CHUNK = make_chunk(
    source_id="gdpr", number=44,
    text="Any transfer of personal data to a third country shall take place only if the conditions of this Chapter are complied with.",
    title="General principle for transfers",
)

# The GDPR completeness block's targets (issue #87) for the retailer breach:
# the reserved principles threshold, the four completeness duty areas, the
# breach additions — notification-authority detail and complaint exposure —
# the third-country transfer target, and the applicability target for DORA,
# whose retrieval seats the perimeter provision first and the reporting
# provision behind it — the pool the drafted DORA machinery limb must be kept
# honest against. The keys are the plan's targets in order, the reserved
# engagement-threshold target first; the plan fits the 1-50 budget.
RETAILER_COMPLETENESS_RETRIEVALS = {
    "principles relating to processing": [PRINCIPLES_CHUNK],
    "information and access rights": [GDPR_INFORMATION_CHUNK],
    "records of processing activities": [GDPR_RECORDS_CHUNK],
    "data protection by design and by default and security": [GDPR_DPBYD_CHUNK],
    "accountability": [GDPR_ACCOUNTABILITY_CHUNK],
    "notification authority and complaint exposure": [GDPR_COMPETENCE_CHUNK, GDPR_COMPLAINT_CHUNK],
    "third-country transfer safeguards": [GDPR_TRANSFER_CHUNK],
    "DORA applicability financial entities": [DORA_SCOPE_CHUNK, DORA_REPORTING_CHUNK],
}


def test_the_gdpr_completeness_block_lands_while_the_gate_drops_the_closed_regimes_machinery_limb(live_client):
    """The GDPR completeness block plus the gate's honesty (issue #87): the
    confirmed GDPR engagement's completeness targets land — information and
    access rights, records, DPbDD and security, accountability, the breach
    additions, and the third-country transfer analysis — while the DORA
    machinery limb drafted for the regime the kept Exclusion Finding settles
    away is rejected with a recorded reason: the added findings inherit the
    engagement gate, so no conditional-limb noise is reintroduced."""
    exclusion = (
        "DORA's incident regime covers financial entities only, and the company is an "
        "online shop rather than a financial entity, so the regime does not reach it."
    )
    principles = "The retailer's processing of the breached customer data must respect the data-protection principles."
    information = (
        "Affected customers must be told about the breach and can exercise their access "
        "rights over the processing."
    )
    records = "The breach response must appear in the retailer's records of processing activities."
    dpbdd = (
        "The retailer's database processing must embed data protection by design and by "
        "default with security measures proportionate to the risk."
    )
    accountability = "The retailer must be able to demonstrate compliance through documented measures."
    authority = (
        "The notification goes to the competent supervisory authority, and affected "
        "customers can lodge complaints the retailer must anticipate."
    )
    transfer = (
        "Where the breached customer data are hosted outside the EEA, any transfer to the "
        "third country rests on the GDPR's transfer safeguards."
    )
    machinery_limb = "The company must report the incident through DORA's templates and channels."
    llm = duty_llm(
        (exclusion, "E9", Strength.moderate),
        (principles, "E1", Strength.weak),
        (information, "E2", Strength.moderate),
        (records, "E3", Strength.weak),
        (dpbdd, "E4", Strength.moderate),
        (accountability, "E5", Strength.weak),
        (authority, ["E6", "E7"], Strength.moderate),
        (transfer, "E8", Strength.weak),
        (machinery_limb, "E10", Strength.moderate),
    )
    llm.plan = plan_with_reserved_first(list(RETAILER_COMPLETENESS_RETRIEVALS))
    script_summaries(llm, [
        "Article 2 sets DORA's perimeter — who the regime covers — which is what the exclusion conclusion turns on.",
        "Article 5's principles govern the retailer's handling of the breached customer data.",
        "Article 13 obliges the retailer to tell affected customers about the breach and answer their access requests.",
        "Article 30 requires the breach response to appear in the records of processing activities.",
        "Article 25 requires the database processing to embed data protection by design and by default.",
        "Article 24 obliges the retailer to demonstrate compliance through documented measures.",
        "Article 55 identifies the competent supervisory authority the notification goes to.",
        "Article 77 gives affected customers the complaint right the retailer must anticipate.",
        "Article 44 conditions any transfer of the breached customer data to a third country on the Chapter V safeguards.",
    ])
    retriever = FakeRetriever(per_query=RETAILER_COMPLETENESS_RETRIEVALS)
    install_fake_pipeline(llm, retriever)

    data = post_gate_scenario(live_client)

    # The completeness block's targets land: every planned target reached retrieval.
    assert retriever.queries == list(RETAILER_COMPLETENESS_RETRIEVALS)

    statements = [f["statement"] for f in data["answer"]["findings"]]
    for statement in (exclusion, principles, information, records, dpbdd, accountability, authority, transfer):
        assert statement in statements, "the completeness block's finding lands in the Answer"
    assert machinery_limb not in statements, "the closed regime's machinery limb is kept honest"

    by_statement = {f["statement"]: f for f in data["answer"]["findings"]}
    assert [(c["source_id"], c["article_number"]) for c in by_statement[exclusion]["citations"]] == [("dora", 2)]
    assert [(c["source_id"], c["article_number"]) for c in by_statement[authority]["citations"]] == [("gdpr", 55), ("gdpr", 77)]

    assert machinery_limb in data["trace"]["unsupported_claims_discarded"]
    decisions = {d["claim"]: d for d in verifier_step(data)["claim_decisions"]}
    assert decisions[machinery_limb]["status"] == "rejected"
    assert "DORA Article 2" in decisions[machinery_limb]["reason"]
    assert "not reaching the scenario" in decisions[machinery_limb]["reason"]

    gate_records = verifier_step(data)["engagement_states"]
    assert gate_records == [{
        "source_id": "dora",
        "state": "closed",
        "conflict": False,
        "open": [],
        "closed": [exclusion],
    }]
    assert len(llm.calls) == 5, "no corrective re-prompt fired: the dropped limb never reached the Summarizer"


DORA_FEEDBACK_CHUNK = make_chunk(
    source_id="dora", number=22,
    text="The competent authority shall acknowledge receipt of the notification and give feedback on the reports.",
    title="Responsibility of financial entities",
)
DORA_REVIEW_CHUNK = make_chunk(
    source_id="dora", number=13,
    text="Financial entities shall conduct a post-incident review after a major ICT-related incident.",
    title="Post-incident review",
)
DORA_COMMUNICATION_CHUNK = make_chunk(
    source_id="dora", number=14,
    text="Financial entities shall put in place a communication strategy for ICT-related incidents.",
    title="Communication",
)
DORA_SIMPLIFIED_CHUNK = make_chunk(
    source_id="dora", number=16,
    text="Small and non-interconnected investment entities may follow the simplified ICT risk-management framework.",
    title="Simplified ICT risk-management framework",
)
DORA_PROPORTIONALITY_CHUNK = make_chunk(
    source_id="dora", number=4,
    text="Financial entities shall size their ICT risk framework in proportion to their nature, scale and risk profile.",
    title="Proportionality",
)

# The DORA machinery block's targets (issue #87) for the bank outage: the
# reserved financial-entity threshold, the reporting machinery, the
# post-incident review with its communication strategy, the size/status
# check, and proportionality. The keys are the plan's targets in order, the
# reserved engagement-threshold target first.
BANK_MACHINERY_RETRIEVALS = {
    "financial entity perimeter": [DORA_SCOPE_CHUNK],
    "incident reporting templates and channels": [DORA_REPORTING_CHUNK],
    "authority feedback on incident reports": [DORA_FEEDBACK_CHUNK],
    "post-incident review and communication strategy": [DORA_REVIEW_CHUNK, DORA_COMMUNICATION_CHUNK],
    "size and status check": [DORA_SIMPLIFIED_CHUNK],
    "proportionality": [DORA_PROPORTIONALITY_CHUNK],
}


def test_the_dora_machinery_block_lands_for_a_confirmed_dora_engagement(live_client):
    """The DORA machinery block (issue #87): a confirmed DORA engagement's
    plan carries the machinery duty areas as their own targets, and the Answer
    carries a Finding for each — the engagement claim, the reporting
    machinery, the authority's feedback, the post-incident review with its
    communication strategy, the size and status check, and proportionality —
    with DORA's engagement open, so the block survives the gate untouched."""
    engagement = (
        "As a credit institution the bank falls within DORA, and the cloud failure is an "
        "ICT-related incident affecting a critical function."
    )
    machinery = (
        "Once the outage is classified major, the bank reports it through initial, "
        "intermediate, and final reports within the harmonised templates and time limits."
    )
    feedback = "The authority's acknowledgement and feedback on the reports close the supervisory loop."
    review = (
        "The major incident triggers a post-incident review, and a communication strategy "
        "frames what clients and stakeholders are told."
    )
    size_status = (
        "The size and status check applies: as a credit institution the bank cannot use the "
        "simplified framework and must comply with the general framework of Articles 5 to 15."
    )
    proportionality = (
        "The bank's incident-response duties scale to its size and risk profile under the "
        "proportionality principle."
    )
    llm = duty_llm(
        (engagement, "E1", Strength.moderate),
        (machinery, "E2", Strength.strong),
        (feedback, "E3", Strength.weak),
        (review, ["E4", "E5"], Strength.weak),
        (size_status, "E6", Strength.weak),
        (proportionality, "E7", Strength.weak),
    )
    llm.plan = plan_with_reserved_first(list(BANK_MACHINERY_RETRIEVALS))
    script_summaries(llm, [
        "Article 2 brings the bank within DORA as a credit institution, engaging the incident regime the other findings develop.",
        "Article 19 obliges the bank to report the major incident through initial, intermediate, and final reports within the templates and time limits.",
        "Article 22 provides the authority's acknowledgement and feedback on the reports, closing the supervisory loop.",
        "Article 13 requires a post-incident review once the major incident has disrupted core activities.",
        "Article 14's communication-strategy duty frames what clients and stakeholders are told.",
        "Article 16 requires the size and status check that routes the bank to the general framework.",
        "Article 4 scales the bank's incident-response duties to its size and risk profile.",
    ])
    retriever = FakeRetriever(per_query=BANK_MACHINERY_RETRIEVALS)
    install_fake_pipeline(llm, retriever)

    resp = live_client.post(
        "/api/analyze",
        json={
            "scenario": {
                "id": "bank-cloud-outage",
                "description": "A bank relies on an external cloud provider to host critical systems used for online banking. A major technical failure at the provider makes the bank's online banking services unavailable to customers for several hours.",
            },
            "question": "What regulatory obligations should the bank consider in relation to this incident?",
        },
    )
    assert resp.status_code == 200
    data = resp.json()

    # The machinery block's targets land: every planned duty-area target reached retrieval.
    assert retriever.queries == list(BANK_MACHINERY_RETRIEVALS)

    statements = [f["statement"] for f in data["answer"]["findings"]]
    for statement in (engagement, machinery, feedback, review, size_status, proportionality):
        assert statement in statements, "the machinery block's finding lands in the Answer"

    by_statement = {f["statement"]: f for f in data["answer"]["findings"]}
    assert [(c["source_id"], c["article_number"]) for c in by_statement[engagement]["citations"]] == [("dora", 2)]
    assert [(c["source_id"], c["article_number"]) for c in by_statement[machinery]["citations"]] == [("dora", 19)]
    assert [(c["source_id"], c["article_number"]) for c in by_statement[feedback]["citations"]] == [("dora", 22)]
    assert [(c["source_id"], c["article_number"]) for c in by_statement[review]["citations"]] == [("dora", 13), ("dora", 14)]
    assert [(c["source_id"], c["article_number"]) for c in by_statement[size_status]["citations"]] == [("dora", 16)]
    assert [(c["source_id"], c["article_number"]) for c in by_statement[proportionality]["citations"]] == [("dora", 4)]

    assert data["trace"]["unsupported_claims_discarded"] == []
    assert len(llm.calls) == 5, "the block lands on its own: no corrective re-prompt fired"
    assert verifier_step(data)["engagement_states"] == [{
        "source_id": "dora",
        "state": "open",
        "conflict": False,
        "open": [engagement],
        "closed": [],
    }]


AI_SCOPE_CHUNK = make_chunk(
    source_id="ai-act", number=2,
    text="This Regulation applies to providers placing on the market or putting into service AI systems.",
    title="Scope",
)
GDPR_HOUSEHOLD_SCOPE_CHUNK = make_chunk(
    source_id="gdpr", number=2,
    text="This Regulation does not apply to processing by a natural person in the course of a purely personal or household activity.",
    title="Material scope",
)

# The noise path the gate keeps honest (issue #87): one applicability target
# whose broad retrieval seats the perimeter provision first and the block's
# duty provisions behind it — the pool the drafted block findings must be
# dropped from when the Exclusion Finding settles the regime away.


def test_the_ai_act_marker_block_inherits_the_gate(live_client):
    """The AI-Act markers block inherits the engagement gate (issue #87): the
    marker findings drafted over an applicability retrieval for a regime the
    kept Exclusion Finding settles away are rejected with recorded reasons —
    the block cannot reintroduce the conditional-limb noise."""
    exclusion = (
        "The company deploys no AI system at all, so the AI Act does not reach "
        "its operations."
    )
    markers = (
        "The provider must pass the conformity assessment, affix the CE marking, "
        "and register the system in the EU database."
    )
    post_market = "The provider keeps post-market monitoring of the system and reports serious incidents."
    literacy = "The staff operating and supervising the system must have sufficient AI literacy."
    date = "The high-risk deployer duties apply from 2 August 2026."
    llm = duty_llm(
        (exclusion, "E1", Strength.moderate),
        (markers, "E2", Strength.moderate),
        (post_market, "E3", Strength.weak),
        (literacy, "E4", Strength.weak),
        (date, "E5", Strength.weak),
    )
    llm.plan = Plan(targets=[ResearchTarget(query="AI Act applicability providers")])
    script_summaries(llm, [
        "Article 2 sets the AI Act's perimeter — who the regime covers — which is what the exclusion conclusion turns on.",
    ])
    retriever = FakeRetriever(per_query={
        "AI Act applicability providers": [
            AI_SCOPE_CHUNK,
            AI_PROVIDER_MARKERS_CHUNK,
            AI_POST_MARKET_CHUNK,
            AI_LITERACY_CHUNK,
            AI_APPLICATION_DATE_CHUNK,
        ],
    })
    install_fake_pipeline(llm, retriever)

    resp = live_client.post(
        "/api/analyze",
        json={
            "scenario": {
                "id": "invoice-consultancy",
                "description": "A small consultancy reviews invoices and manages payroll by hand, with no AI system in use.",
            },
            "question": "What EU regulatory obligations should the company consider?",
        },
    )
    assert resp.status_code == 200
    data = resp.json()

    assert retriever.queries == ["AI Act applicability providers"]

    statements = [f["statement"] for f in data["answer"]["findings"]]
    assert exclusion in statements, "the Exclusion Finding carries the Perimeter Citation"
    by_statement = {f["statement"]: f for f in data["answer"]["findings"]}
    assert [(c["source_id"], c["article_number"]) for c in by_statement[exclusion]["citations"]] == [("ai-act", 2)]
    for statement in (markers, post_market, literacy, date):
        assert statement not in statements, "the marker finding on the settled-away regime is dropped"
        assert statement in data["trace"]["unsupported_claims_discarded"]

    decisions = {d["claim"]: d for d in verifier_step(data)["claim_decisions"]}
    for statement in (markers, post_market, literacy, date):
        assert decisions[statement]["status"] == "rejected"
        assert "EU AI Act Article 2" in decisions[statement]["reason"]
        assert "not reaching the scenario" in decisions[statement]["reason"]

    gate_records = verifier_step(data)["engagement_states"]
    assert gate_records == [{
        "source_id": "ai-act",
        "state": "closed",
        "conflict": False,
        "open": [],
        "closed": [exclusion],
    }]
    assert len(llm.calls) == 5, "no corrective re-prompt fired: the dropped findings never reached the Summarizer"


def test_the_gdpr_completeness_block_inherits_the_gate(live_client):
    """The GDPR completeness block inherits the engagement gate (issue #87):
    the completeness findings drafted over an applicability retrieval for a
    regime the household-exception Exclusion Finding settles away are
    rejected with recorded reasons — no conditional-limb noise."""
    exclusion = (
        "The processing is purely personal or household activity, so the GDPR "
        "does not reach it."
    )
    information = (
        "The individuals whose data are processed must be told about the processing "
        "and can exercise their access rights."
    )
    records = "The processing must appear in the records of processing activities."
    dpbdd = (
        "The processing must embed data protection by design and by default with "
        "security measures proportionate to the risk."
    )
    accountability = "The company must be able to demonstrate compliance through documented measures."
    llm = duty_llm(
        (exclusion, "E1", Strength.moderate),
        (information, "E2", Strength.moderate),
        (records, "E3", Strength.weak),
        (dpbdd, "E4", Strength.moderate),
        (accountability, "E5", Strength.weak),
    )
    llm.plan = Plan(targets=[ResearchTarget(query="GDPR applicability household exception")])
    script_summaries(llm, [
        "Article 2 sets the GDPR's perimeter — who the regulation covers — which is what the exclusion conclusion turns on.",
    ])
    retriever = FakeRetriever(per_query={
        "GDPR applicability household exception": [
            GDPR_HOUSEHOLD_SCOPE_CHUNK,
            GDPR_INFORMATION_CHUNK,
            GDPR_RECORDS_CHUNK,
            GDPR_DPBYD_CHUNK,
            GDPR_ACCOUNTABILITY_CHUNK,
        ],
    })
    install_fake_pipeline(llm, retriever)

    resp = live_client.post(
        "/api/analyze",
        json={
            "scenario": {
                "id": "personal-household-activity",
                "description": "An individual keeps a private family blog and manages a household address book by hand, with no professional or commercial processing.",
            },
            "question": "What EU regulatory obligations apply to this processing?",
        },
    )
    assert resp.status_code == 200
    data = resp.json()

    assert retriever.queries == ["GDPR applicability household exception"]

    statements = [f["statement"] for f in data["answer"]["findings"]]
    assert exclusion in statements, "the Exclusion Finding carries the Perimeter Citation"
    by_statement = {f["statement"]: f for f in data["answer"]["findings"]}
    assert [(c["source_id"], c["article_number"]) for c in by_statement[exclusion]["citations"]] == [("gdpr", 2)]
    for statement in (information, records, dpbdd, accountability):
        assert statement not in statements, "the completeness finding on the settled-away regime is dropped"
        assert statement in data["trace"]["unsupported_claims_discarded"]

    decisions = {d["claim"]: d for d in verifier_step(data)["claim_decisions"]}
    for statement in (information, records, dpbdd, accountability):
        assert decisions[statement]["status"] == "rejected"
        assert "GDPR Article 2" in decisions[statement]["reason"]
        assert "not reaching the scenario" in decisions[statement]["reason"]

    gate_records = verifier_step(data)["engagement_states"]
    assert gate_records == [{
        "source_id": "gdpr",
        "state": "closed",
        "conflict": False,
        "open": [],
        "closed": [exclusion],
    }]
    assert len(llm.calls) == 5, "no corrective re-prompt fired: the dropped findings never reached the Summarizer"


# --- Citation discipline: decisive citations only (issue #88) -------------------


DISCIPLINE_RETRIEVALS = {
    "principles relating to processing": [PRINCIPLES_CHUNK],
    "security of processing": [GDPR_SECURITY_CHUNK, GDPR_DEFINITIONS_CHUNK],
}


def discipline_llm() -> "ScriptedLlm":
    """A scripted run under the citation discipline (issue #88): the duty
    claim's draft pads itself with the definitions Article's label, but the
    disciplined Verdict keeps only the provision that decides it; the framing
    reference rides only on the claim that is about the framing, and the
    reserved engagement anchor's decisive citation survives untouched."""
    engagement = (
        "The company's processing of applicant data must respect the data-protection "
        "principles the GDPR sets for it."
    )
    duty = (
        "The company must implement appropriate technical and organisational security "
        "measures for the applicant data."
    )
    framing = "The company acts as a controller of the applicant data within the GDPR's defined roles."
    return ScriptedLlm(
        plan=Plan(targets=[
            ResearchTarget(query="principles relating to processing", reserved=True),
            ResearchTarget(query="security of processing"),
        ]),
        claims=DraftClaims(claims=[
            DraftClaim(statement=engagement, evidence_refs=["E1"]),
            DraftClaim(statement=duty, evidence_refs=["E2", "E3"]),
            DraftClaim(statement=framing, evidence_refs=["E3"]),
        ]),
        verdicts=Verdicts(verdicts=[
            grounded_verdict(engagement, Strength.strong, ["E1"]),
            grounded_verdict(duty, Strength.moderate, ["E2"]),
            grounded_verdict(framing, Strength.weak, ["E3"]),
        ]),
        proposals=ActionProposals(proposals=[]),
        summaries=Summaries(summaries=[
            ProvisionSummary(
                ref="P1",
                relevance="Article 5's principles are the threshold the processing must respect.",
                strength=Strength.strong,
            ),
            ProvisionSummary(
                ref="P2",
                relevance="Article 32 imposes the security measures the duty finding turns on.",
                strength=Strength.moderate,
            ),
            ProvisionSummary(
                ref="P3",
                relevance="Article 4 supplies the controller role the framing finding decides with.",
                strength=Strength.weak,
            ),
        ]),
    )


def test_citation_discipline_drops_framing_only_citations_and_keeps_decisive_ones_and_anchors(live_client):
    """The discipline's scripted outcome (issue #88): the duty finding cites
    only the provision that decides it — the framing-only reference it drafted
    is gone from the Answer even though the definitions chunk sat in the pool;
    the decisive citations survive; and the reserved engagement anchor rides
    the framing provision it decides with, no corrective re-prompt owed."""
    llm = discipline_llm()
    install_fake_pipeline(llm, FakeRetriever(per_query=DISCIPLINE_RETRIEVALS))

    resp = live_client.post(
        "/api/analyze",
        json={
            "scenario": {
                "id": "lending-app-data",
                "description": "A Spanish fintech evaluating loan applications automatically from applicant data",
            },
            "question": "What data-protection duties bind the company when it processes applicant data?",
        },
    )
    assert resp.status_code == 200
    data = resp.json()

    engagement = (
        "The company's processing of applicant data must respect the data-protection "
        "principles the GDPR sets for it."
    )
    duty = (
        "The company must implement appropriate technical and organisational security "
        "measures for the applicant data."
    )
    framing = "The company acts as a controller of the applicant data within the GDPR's defined roles."

    by_statement = {f["statement"]: f for f in data["answer"]["findings"]}
    assert set(by_statement) == {engagement, duty, framing}, "every disciplined verdict stayed"
    # Framing-only citation dropped: the duty finding cites only the provision
    # that decides it, though the definitions chunk sat in the Evidence pool.
    assert [(c["source_id"], c["article_number"]) for c in by_statement[duty]["citations"]] == [("gdpr", 32)]
    # The decisive citation survives: the reserved engagement anchor rides the
    # principles provision it decides with.
    assert [(c["source_id"], c["article_number"]) for c in by_statement[engagement]["citations"]] == [("gdpr", 5)]
    # The framing reference rides only on the claim that is about the framing.
    assert [(c["source_id"], c["article_number"]) for c in by_statement[framing]["citations"]] == [("gdpr", 4)]

    # The Answer's citation list: one entry per cited provision, the definitions
    # one present exactly once.
    citations = data["answer"]["citations"]
    assert sorted(c["article_number"] for c in citations) == [4, 5, 32]

    assert data["trace"]["unsupported_claims_discarded"] == []
    assert len(llm.calls) == 5, "the anchor landed in the first pass: no corrective re-prompt owed"


# --- Deterministic citation trims with protected refs (spec #94, T5) ------------


TRIM_RETRIEVALS = {
    "security of processing": [GDPR_SECURITY_CHUNK, NOTIFICATION_CHUNK],
}


TRIM_DUTY_STATEMENT = (
    "The company must implement appropriate technical and organisational security "
    "measures for the applicant data."
)


def trim_llm(*verdicts: "Verdict") -> "ScriptedLlm":
    """A scripted run over the security-duty claim with one split Verdict —
    the raw material the deterministic citation trim reads."""
    from src.live_workflow import DraftClaim, DraftClaims, Plan, ResearchTarget

    duty = TRIM_DUTY_STATEMENT
    return ScriptedLlm(
        plan=Plan(targets=[ResearchTarget(query="security of processing")]),
        claims=DraftClaims(claims=[
            DraftClaim(statement=duty, evidence_refs=["E1", "E2"]),
        ]),
        verdicts=Verdicts(verdicts=list(verdicts)),
        proposals=ActionProposals(proposals=[]),
        summaries=Summaries(summaries=[
            ProvisionSummary(
                ref="P1",
                relevance="Article 32 imposes the security measures the duty finding turns on.",
                strength=Strength.strong,
            ),
        ]),
    )


def assert_trim_step(data, statement: str) -> dict:
    """The kept decision for one statement, from the Verifier's detailed-trace step."""
    decisions = {d["claim"]: d for d in verifier_step(data)["claim_decisions"]}
    return decisions[statement]


def test_auxiliary_refs_are_trimmed_from_kept_findings_and_every_drop_is_recorded(live_client):
    """The deterministic citation trim (spec #94, T5): the Verdict's auxiliary
    references are dropped from the kept Finding's Citations by application
    code — the rule survives prompt drift — and every drop is recorded on the
    kept decision in the detailed trace and counted in the Execution trace."""

    duty = TRIM_DUTY_STATEMENT
    llm = trim_llm(
        Verdict(
            statement=duty,
            supported=True,
            material=True,
            strength=Strength.strong,
            decisive_refs=["E1"],
            auxiliary_refs=["E2"],
        ),
    )
    install_fake_pipeline(llm, FakeRetriever(per_query=TRIM_RETRIEVALS))

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    findings = [f for f in data["answer"]["findings"] if f["statement"] == duty]
    assert len(findings) == 1, "the claim stays kept — only its auxiliary Citation goes"
    # The auxiliary reference is dropped: the Finding cites only the provision
    # that decides it, though the notification chunk sat in the Evidence pool.
    assert [(c["source_id"], c["article_number"]) for c in findings[0]["citations"]] == [("gdpr", 32)]
    assert all(c["article_number"] != 33 for c in data["answer"]["citations"])

    decision = assert_trim_step(data, duty)
    assert decision["status"] == "kept"
    assert decision["dropped_refs"] == ["E2"], "every drop is recorded, never silent"

    assert "trimmed 1 auxiliary citation" in data["trace"]["summary"].lower(), (
        "the Execution trace accounts for the trim"
    )


def test_a_finding_never_falls_below_one_resolvable_ref(live_client):
    """The trim's floor (spec #94, T5): when every decisive reference fails to
    resolve and the auxiliary drop would leave no Citation at all, the first
    resolvable reference survives — a Finding never falls below one resolvable
    ref, and the still-dropped remainder stays recorded."""

    duty = TRIM_DUTY_STATEMENT
    llm = trim_llm(
        Verdict(
            statement=duty,
            supported=True,
            material=True,
            strength=Strength.strong,
            decisive_refs=["E99"],
            auxiliary_refs=["E1", "E2"],
        ),
    )
    install_fake_pipeline(llm, FakeRetriever(per_query=TRIM_RETRIEVALS))

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    findings = [f for f in data["answer"]["findings"] if f["statement"] == duty]
    assert len(findings) == 1, "the floor keeps the Finding alive"
    assert [(c["source_id"], c["article_number"]) for c in findings[0]["citations"]] == [("gdpr", 32)], (
        "the first resolvable reference survives"
    )

    decision = assert_trim_step(data, duty)
    assert decision["dropped_refs"] == ["E2"], "the remainder is dropped and recorded"


def test_a_supported_claim_whose_refs_resolve_to_nothing_is_still_rejected(live_client):
    """The trim never fabricates a Citation: when no listed reference resolves
    — decisive or auxiliary — the claim is rejected with the no-citable-
    evidence reason, exactly as before the trim existed."""

    duty = TRIM_DUTY_STATEMENT
    llm = trim_llm(
        Verdict(
            statement=duty,
            supported=True,
            material=True,
            strength=Strength.strong,
            decisive_refs=["E98"],
            auxiliary_refs=["E99"],
        ),
    )
    install_fake_pipeline(llm, FakeRetriever(per_query=TRIM_RETRIEVALS))

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    assert all(f["statement"] != duty for f in data["answer"]["findings"])
    assert duty in data["trace"]["unsupported_claims_discarded"]
    decision = assert_trim_step(data, duty)
    assert decision["status"] == "rejected"
    assert decision["reason"] == "the selected evidence resolves to no citable provision"


def test_an_unresolvable_auxiliary_ref_rides_the_unknown_label_warning(live_client):
    """An auxiliary reference matching no Chunk never becomes a Citation, so
    it is not a recorded trim drop: it rides the same unknown-label treatment
    the decisive references always got, and ``dropped_refs`` carries only
    references the trim actually removed."""
    duty = TRIM_DUTY_STATEMENT
    llm = trim_llm(
        Verdict(
            statement=duty,
            supported=True,
            material=True,
            strength=Strength.strong,
            decisive_refs=["E1"],
            auxiliary_refs=["E2", "E99"],
        ),
    )
    install_fake_pipeline(llm, FakeRetriever(per_query=TRIM_RETRIEVALS))

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    findings = [f for f in data["answer"]["findings"] if f["statement"] == duty]
    assert [(c["source_id"], c["article_number"]) for c in findings[0]["citations"]] == [("gdpr", 32)]
    decision = assert_trim_step(data, duty)
    assert decision["status"] == "kept"
    assert decision["dropped_refs"] == ["E2"], "only the resolved auxiliary ref is a trim drop"


def test_a_label_listed_in_both_roles_rides_one_citation_and_records_no_phantom_drop(live_client):
    """A label the Verdict lists as both decisive and auxiliary keeps its one
    decisive Citation — the auxiliary copy drops nothing, so no drop is
    recorded for it."""
    duty = TRIM_DUTY_STATEMENT
    llm = trim_llm(
        Verdict(
            statement=duty,
            supported=True,
            material=True,
            strength=Strength.strong,
            decisive_refs=["E1"],
            auxiliary_refs=["E1", "E2"],
        ),
    )
    install_fake_pipeline(llm, FakeRetriever(per_query=TRIM_RETRIEVALS))

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    findings = [f for f in data["answer"]["findings"] if f["statement"] == duty]
    assert [(c["source_id"], c["article_number"]) for c in findings[0]["citations"]] == [("gdpr", 32)], (
        "the label rides its decisive Citation exactly once"
    )
    decision = assert_trim_step(data, duty)
    assert decision["dropped_refs"] == ["E2"], "no phantom drop for the dual-listed label"


def test_a_perimeter_provision_ref_survives_the_auxiliary_drop_and_keeps_the_gate_honest(live_client):
    """Drop immunity (spec #94, T5): a ref matching a Perimeter provision
    survives the auxiliary drop regardless of the Verifier's role judgment —
    the Engagement state machine depends on it — so the Exclusion Finding
    keeps its Perimeter Citation and the engagement gate still closes the
    regime over the trimmed set."""

    exclusion = (
        "DORA's incident regime covers financial entities only, and the company is an "
        "online shop rather than a financial entity, so the regime does not reach it."
    )
    duty = "The company must report major ICT-related incidents under DORA's reporting regime."
    llm = make_offline_llm()
    llm.plan = Plan(targets=[
        ResearchTarget(query="DORA applicability financial entities"),
        ResearchTarget(query="ICT incident reporting"),
        ResearchTarget(query="security of processing"),
    ])
    llm.claims = DraftClaims(claims=[
        DraftClaim(statement=exclusion, evidence_refs=["E1", "E2", "E3"]),
        DraftClaim(statement=duty, evidence_refs=["E2"]),
    ])
    llm.verdicts = Verdicts(verdicts=[
        Verdict(
            statement=exclusion,
            supported=True,
            material=True,
            strength=Strength.moderate,
            decisive_refs=["E2"],
            auxiliary_refs=["E1", "E3"],
        ),
        Verdict(
            statement=duty,
            supported=True,
            material=True,
            strength=Strength.moderate,
            decisive_refs=["E2"],
        ),
    ])
    llm.proposals = ActionProposals(proposals=[])
    llm.summaries = Summaries(summaries=[
        ProvisionSummary(
            ref="P1",
            relevance="Article 19 imposes the incident-reporting duties the exclusion carves the company out of.",
            strength=Strength.moderate,
        ),
        ProvisionSummary(
            ref="P2",
            relevance="Article 2 decides who DORA covers, which is what the exclusion turns on.",
            strength=Strength.moderate,
        ),
    ])
    install_fake_pipeline(llm, FakeRetriever(per_query=RETAILER_RETRIEVALS))

    data = post_gate_scenario(live_client)

    by_statement = {f["statement"]: f for f in data["answer"]["findings"]}
    exclusion_finding = by_statement[exclusion]
    # Decisive first, then the protected perimeter ref — kept despite its
    # auxiliary role, so the Exclusion Finding stays evidence-backed.
    assert [(c["source_id"], c["article_number"]) for c in exclusion_finding["citations"]] == [
        ("dora", 19),
        ("dora", 2),
    ]
    decision = assert_trim_step(data, exclusion)
    assert decision["status"] == "kept"
    assert decision["dropped_refs"] == ["E3"], "the unguarded auxiliary ref is still trimmed"

    # The Engagement gate reads the trimmed set: the surviving perimeter
    # Citation closes DORA, so the duty limb scoped only to it drops.
    assert duty not in by_statement
    assert duty in data["trace"]["unsupported_claims_discarded"]
    duty_decision = assert_trim_step(data, duty)
    assert "DORA Article 2" in duty_decision["reason"]
    assert "not reaching the scenario" in duty_decision["reason"]
    gate_records = verifier_step(data)["engagement_states"]
    assert gate_records == [{
        "source_id": "dora",
        "state": "closed",
        "conflict": False,
        "open": [],
        "closed": [exclusion],
    }]


def test_a_reserved_anchor_cited_as_auxiliary_survives_the_trim(live_client):
    """Drop immunity for Reserved anchors (spec #94, T5): a kept Finding
    citing the reserved engagement-threshold provision keeps it even when the
    Verdict rated the reference auxiliary — the anchor backstop depends on
    it, so no corrective re-prompt is owed."""

    engagement = (
        "The breach must be notified to the authority without undue delay, and the "
        "processing of the employee data must respect the data-protection principles."
    )
    llm = make_offline_llm()
    llm.plan = Plan(targets=[
        ResearchTarget(query="principles relating to processing", reserved=True),
        ResearchTarget(query="breach notification duties"),
    ])
    llm.claims = DraftClaims(claims=[DraftClaim(statement=engagement, evidence_refs=["E1", "E2"])])
    llm.verdicts = Verdicts(verdicts=[
        Verdict(
            statement=engagement,
            supported=True,
            material=True,
            strength=Strength.strong,
            decisive_refs=["E2"],
            auxiliary_refs=["E1"],
        ),
    ])
    llm.proposals = ActionProposals(proposals=[])
    llm.summaries = Summaries(summaries=[
        ProvisionSummary(
            ref="P1",
            relevance="The notification duty is what the question turns on.",
            strength=Strength.moderate,
        ),
        ProvisionSummary(
            ref="P2",
            relevance="The principles provision decides whether the processing is lawful at all.",
            strength=Strength.strong,
        ),
    ])
    install_fake_pipeline(llm, FakeRetriever(per_query=ANCHOR_RETRIEVALS))

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    # The auxiliary-rated anchor survived: the strong engagement Finding cites
    # the threshold provision alongside its decisive citation.
    strong = next(f for f in data["answer"]["findings"] if f["statement"] == engagement)
    assert [(c["source_id"], c["article_number"]) for c in strong["citations"]] == [
        ("gdpr", 33),
        ("gdpr", 5),
    ]
    decision = assert_trim_step(data, engagement)
    assert decision["status"] == "kept"
    assert decision["dropped_refs"] == [], "the anchor ref is immune, never dropped"
    # The anchor backstop depends on the surviving ref: no re-prompt owed.
    assert len(claims_calls(llm)) == 1
    assert researcher_step(data).get("correction") is None


def test_the_corrective_pass_anchor_survives_an_auxiliary_role(live_client):
    """Drop immunity holds in the backstop's merged decision too (spec #94,
    T5): the corrective re-prompt's engagement Claim lands with its anchor
    ref even when the retry Verdict rated it auxiliary — the second
    claim-decision pass protects the same refs."""

    llm = anchor_llm()
    llm.verdicts_retry = Verdicts(verdicts=[
        Verdict(
            statement="The processing of the employee data must respect the data-protection principles.",
            supported=True,
            material=True,
            strength=Strength.strong,
            decisive_refs=["E2"],
            auxiliary_refs=["E1"],
        ),
    ])
    install_fake_pipeline(llm, FakeRetriever(per_query=ANCHOR_RETRIEVALS))

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    assert len(claims_calls(llm)) == 2, "the backstop still owes its one corrective re-prompt"
    strong = next(f for f in data["answer"]["findings"] if f["strength"] == "strong")
    assert [(c["source_id"], c["article_number"]) for c in strong["citations"]] == [
        ("gdpr", 33),
        ("gdpr", 5),
    ], "the anchor survives its auxiliary role in the corrective pass"


def test_the_proposer_and_summarizer_see_the_trimmed_set(live_client):
    """Trims run before the downstream consumers (spec #94, T5): a proposal
    grounded on a trimmed Citation's label resolves to nothing and is
    rejected with a recorded reason, and a relevance statement referencing a
    trimmed provision's label names no cited provision — both read exactly
    the labels the trimmed set left behind."""

    duty = TRIM_DUTY_STATEMENT
    llm = trim_llm(
        Verdict(
            statement=duty,
            supported=True,
            material=True,
            strength=Strength.strong,
            decisive_refs=["E1"],
            auxiliary_refs=["E2"],
        ),
    )
    llm.proposals = ActionProposals(proposals=[
        ActionProposal(
            action="Have a qualified professional verify the security measures against the company's actual setup.",
            kind="verify_against_facts",
            citation_refs=["C1"],
        ),
        ActionProposal(
            action="Have a qualified professional check the breach-notification duty against the company's processes.",
            kind="verify_against_facts",
            citation_refs=["C2"],
        ),
    ])
    llm.summaries = Summaries(summaries=[
        ProvisionSummary(
            ref="P1",
            relevance="Article 32 imposes the security measures the duty finding turns on.",
            strength=Strength.strong,
        ),
        ProvisionSummary(
            ref="P2",
            relevance="Article 33 imposes the notification duty the trimmed reference pointed at.",
            strength=Strength.moderate,
        ),
    ])
    install_fake_pipeline(llm, FakeRetriever(per_query=TRIM_RETRIEVALS))

    resp = post_arbitrary_scenario(live_client)
    assert resp.status_code == 200
    data = resp.json()

    # The Proposer saw the trimmed set: C2 — the trimmed reference's label —
    # resolves to nothing, so that proposal is rejected and recorded.
    proposer_steps = [s for s in data["detailed_trace"] if s["step"] == "proposer"]
    by_action = {d["action"]: d for d in proposer_steps[0]["action_decisions"]}
    kept = by_action[
        "Have a qualified professional verify the security measures against the company's actual setup."
    ]
    assert kept["status"] == "kept"
    rejected = by_action[
        "Have a qualified professional check the breach-notification duty against the company's processes."
    ]
    assert rejected["status"] == "rejected"
    assert rejected["reason"] == "its citations resolve to no kept Finding"
    assert rejected["dropped_refs"] == ["C2"]

    # The Summarizer saw the trimmed set too: P2 — the trimmed provision —
    # names no cited provision, and its statement is rejected and recorded.
    summarizer_steps = [s for s in data["detailed_trace"] if s["step"] == "summarizer"]
    summary_decisions = {d["ref"]: d for d in summarizer_steps[0]["summary_decisions"]}
    assert summary_decisions["P1"]["status"] == "kept"
    assert summary_decisions["P2"]["status"] == "rejected"
    assert summary_decisions["P2"]["reason"] == "the label names no cited provision"

    assert all(c["article_number"] != 33 for c in data["answer"]["citations"])
    assert len(llm.calls) == 5, "no corrective re-prompt fired: P1 was covered in the first pass"
