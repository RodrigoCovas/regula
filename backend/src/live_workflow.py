"""The Live workflow: Planner → Researcher → Verifier as a LangGraph graph.

One pass answers an arbitrary Scenario: the Planner decomposes the
Regulatory question into research targets; the Researcher gathers Evidence
exclusively through the retrieval tool under the locked single-pass budget
(~8 Chunks); the Verifier checks every produced Claim against the retrieved
Evidence, tags its Strength, and discards Unsupported claims into the
Execution trace.

Two rules are enforced by application code, never trusted to the LLM:

- Citations are derived deterministically from Chunk provision metadata —
  the LLM only selects which Chunks support a Claim by their label, and a
  Claim whose references resolve to no Chunk cannot become a Finding.
- A Claim no verdict supports is an Unsupported claim: absent from the
  Answer, recorded in the Execution trace.

When retrieval returns nothing relevant enough (no Chunk clears the
relevance threshold), no LLM call drafts or verifies claims: the response
is the Insufficient-evidence path — empty Findings, a Known limitation
explaining the gap, and Actions suggesting how to narrow the question.
"""

import json
import logging
import time
from collections import Counter
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .availability import ENGLISH_ONLY_LIMITATION
from .llm import Llm
from .models import AnalyzeRequest, AnalyzeResponse, Answer, Citation, ClaimDecision, Chunk, Finding, ProvisionKind, Strength, Trace, PROVISION_NUMBER_FIELDS, quote_snippet
from .retrieval import MAX_RETRIEVED_CHUNKS, Retriever

logger = logging.getLogger(__name__)

# The workflow marker the Execution trace carries for every served Live answer.
LIVE_WORKFLOW_MARKER = "planner -> researcher -> verifier"

# Prompt-side bound: each retrieved Chunk contributes at most this many
# characters of Evidence to a prompt, so eight Chunks stay within a modest,
# predictable context regardless of how long the stored provisions run.
_MAX_CHUNK_CHARS = 1500

INSUFFICIENT_EVIDENCE_LIMITATION = (
    "Known limitation: the Corpus holds nothing relevant enough to answer this "
    "question, so no Findings were produced."
)

NARROW_THE_QUESTION_ACTIONS = [
    "Rephrase the Regulatory question using the vocabulary of the Corpus — EU AI Act, GDPR, or DORA terms such as 'high-risk AI system', 'automated decision-making', or 'ICT third-party risk'.",
    "Check whether the obligation you are asking about falls outside the Corpus (e.g. national requirements of a single member state).",
]


# --- Workflow boundaries: every agent input/output crosses as a validated schema ---


class ResearchTarget(BaseModel):
    """One retrieval query the Planner wants researched."""

    query: str


class Plan(BaseModel):
    """The Planner's decomposition of the Regulatory question."""

    targets: list[ResearchTarget] = Field(default_factory=list)


class DraftClaim(BaseModel):
    """A Candidate claim the Researcher drafted from retrieved Evidence."""

    statement: str
    evidence_refs: list[str] = Field(default_factory=list)


class DraftClaims(BaseModel):
    claims: list[DraftClaim] = Field(default_factory=list)


class Verdict(BaseModel):
    """The Verifier's decision about one drafted Claim."""

    statement: str
    supported: bool
    strength: Strength = Strength.moderate
    evidence_refs: list[str] = Field(default_factory=list)


class Verdicts(BaseModel):
    verdicts: list[Verdict] = Field(default_factory=list)


class LabeledEvidence(BaseModel):
    """One retrieved Chunk with the stable label the agents reference it by."""

    label: str
    chunk: Chunk


class LiveState(BaseModel):
    """The LangGraph state: one field per boundary between the three agents."""

    scenario_description: str = ""
    question: str = ""
    plan: list[ResearchTarget] = Field(default_factory=list)
    evidence: list[LabeledEvidence] = Field(default_factory=list)
    retrievals: list[dict] = Field(default_factory=list)  # one record per retrieval-tool call
    drafted: DraftClaims = Field(default_factory=DraftClaims)
    verdicts: Verdicts = Field(default_factory=Verdicts)


# --- Prompts: JSON-only instructions; the client repeats the schema contract ---


_PLANNER_SYSTEM = (
    "You are the Planner of a regulatory research assistant. Decompose the regulatory "
    "question into 1-3 short keyword retrieval queries that would locate the relevant "
    "provisions (articles, recitals, annexes) in a corpus of EU regulations: the AI Act, "
    "GDPR, and DORA. Name concepts and likely provision subjects, not full sentences."
)

_RESEARCHER_SYSTEM = (
    "You are the Researcher of a regulatory research assistant. You are given numbered "
    "evidence excerpts retrieved from the corpus. Draft candidate claims that answer the "
    "question, each grounded ONLY in the listed evidence. For every claim list the evidence "
    "labels it relies on; never reference a label that was not given to you. If the evidence "
    "supports nothing relevant, return an empty claims list."
)

_VERIFIER_SYSTEM = (
    "You are the Verifier of a regulatory research assistant. You are given numbered evidence "
    "excerpts and drafted claims. Decide for each claim: supported — true only if some listed "
    "provision bears on the statement, false when none does either way; strength — 'strong' "
    "when the provisions directly and explicitly establish the claim, 'moderate' when derived "
    "from provisions read together or contingent on facts the corpus cannot settle, 'weak' "
    "when the provisions supply framing only (definitions, vocabulary); evidence_refs — the "
    "labels of the supporting provisions, empty for unsupported claims."
)


def _evidence_block(evidence: list[LabeledEvidence]) -> str:
    parts = []
    for item in evidence:
        chunk = item.chunk
        text = chunk.text[:_MAX_CHUNK_CHARS]
        parts.append(f"[{item.label}] {chunk.source_id} {chunk.kind.value} {chunk.title or ''}\n{text}")
    return "\n\n".join(parts)


# --- Deterministic Citation derivation from Chunk metadata ---

_PROVISION_NOUNS = {
    ProvisionKind.article: "Article",
    ProvisionKind.recital: "Recital",
    ProvisionKind.annex: "Annex",
}


def _provision_label(chunk: Chunk) -> str:
    return f"{_PROVISION_NOUNS[chunk.kind]} {chunk.provision_number}"


def derive_citation(chunk: Chunk) -> Citation:
    """Build the Citation for one Chunk from its stored provision metadata."""
    return Citation.model_validate({
        "source_id": chunk.source_id,
        "section": chunk.title,
        "provision": _provision_label(chunk),
        "quote": quote_snippet(chunk.text),
        # The exactly-one-target field comes from the Chunk's validated metadata.
        PROVISION_NUMBER_FIELDS[chunk.kind]: chunk.provision_number,
    })


def _resolve_refs(refs: list[str], evidence_by_label: dict[str, Chunk]) -> tuple[list[Citation], list[str]]:
    """Map evidence labels back to Citations, reporting labels that match no Chunk."""
    resolved, unknown = [], []
    for ref in refs:
        chunk = evidence_by_label.get(ref)
        if chunk is None:
            unknown.append(ref)
        else:
            resolved.append(derive_citation(chunk))
    return resolved, unknown


_NO_CITABLE_EVIDENCE_REASON = "the selected evidence resolves to no citable provision"
_NO_VERDICT_REASON = "the Verifier returned no decision for this claim"


def _decide_claims(
    drafted: DraftClaims, verdicts: Verdicts, evidence_by_label: dict[str, Chunk]
) -> tuple[list[Finding], list[str], list[ClaimDecision]]:
    """Turn verdicts into kept Findings, discarded Unsupported claims, and per-claim decisions.

    A Claim becomes a Finding only when the verdict supports it AND its evidence
    references resolve to real Chunks — otherwise it is an Unsupported claim:
    never in the Answer, recorded in the Execution trace. Every drafted claim
    must end up decided: a claim the Verifier never returned a verdict for is
    recorded as rejected too — nothing drafted may vanish silently. Verdicts
    are matched to drafts one-for-one per statement, so duplicate statements
    each consume their own verdict.
    """
    findings: list[Finding] = []
    discarded: list[str] = []
    decisions: list[ClaimDecision] = []
    outstanding = Counter(verdict.statement for verdict in verdicts.verdicts)
    for verdict in verdicts.verdicts:
        if verdict.supported:
            citations, unknown_refs = _resolve_refs(verdict.evidence_refs, evidence_by_label)
            if unknown_refs:
                logger.warning(
                    "verifier cited evidence labels matching no Chunk (%s); dropped from the claim's Citations",
                    ", ".join(unknown_refs),
                )
        else:
            citations = []
        if verdict.supported and citations:
            findings.append(Finding(statement=verdict.statement, strength=verdict.strength, citations=citations))
            decisions.append(ClaimDecision(claim=verdict.statement, status="kept"))
        else:
            reason = "no Evidence in the Corpus supports this claim"
            if verdict.supported:
                reason = _NO_CITABLE_EVIDENCE_REASON
            discarded.append(verdict.statement)
            decisions.append(ClaimDecision(claim=verdict.statement, status="rejected", reason=reason))
    for claim in drafted.claims:
        if outstanding[claim.statement] > 0:
            outstanding[claim.statement] -= 1
        else:
            discarded.append(claim.statement)
            decisions.append(ClaimDecision(claim=claim.statement, status="rejected", reason=_NO_VERDICT_REASON))
    return findings, discarded, decisions


# --- The graph: START → planner → researcher → verifier → END ---


def _build_graph(llm: Llm, retriever: Retriever):
    def planner(state: LiveState) -> dict:
        started = time.perf_counter()
        user = f"Company/product scenario: {state.scenario_description or '(not described)'}\nRegulatory question: {state.question}"
        plan = llm.complete(system=_PLANNER_SYSTEM, user=user, schema=Plan)
        logger.info("planner: %d target(s) in %.2fs", len(plan.targets), time.perf_counter() - started)
        return {"plan": plan.targets}

    def researcher(state: LiveState) -> dict:
        """Gather Evidence exclusively through the retrieval tool, then draft Claims."""
        retrievals: list[dict] = []
        per_target: list[list[Chunk]] = []
        seen: set[tuple] = set()
        for target in state.plan:
            found = retriever.retrieve(target.query)
            retrievals.append(
                {
                    "tool": "retrieve_chunks",
                    "input": {"query": target.query},
                    "chunks_returned": len(found),
                }
            )
            fresh: list[Chunk] = []
            for chunk in found:
                identity = (
                    chunk.source_id,
                    chunk.kind.value,
                    chunk.provision_number,
                    chunk.chunk_index,
                )
                if identity in seen:
                    continue
                seen.add(identity)
                fresh.append(chunk)
            per_target.append(fresh)

        # Fair-share fill under the locked single-pass budget: every target
        # claims its share of the pool before any target's broad results can
        # backfill it, so a greedy first query cannot starve the rest.
        evidence: list[LabeledEvidence] = []

        def take(chunk: Chunk) -> None:
            evidence.append(LabeledEvidence(label=f"E{len(evidence) + 1}", chunk=chunk))

        if per_target:
            fair_share = max(1, MAX_RETRIEVED_CHUNKS // len(per_target))
            for fresh in per_target:
                for chunk in fresh[:fair_share]:
                    take(chunk)
            if len(evidence) < MAX_RETRIEVED_CHUNKS:
                for fresh in per_target:
                    for chunk in fresh[fair_share:]:
                        take(chunk)
                        if len(evidence) >= MAX_RETRIEVED_CHUNKS:
                            break
                    if len(evidence) >= MAX_RETRIEVED_CHUNKS:
                        break

        drafted = DraftClaims()
        if evidence:
            user = (
                f"Regulatory question: {state.question}\n\nRetrieved evidence:\n\n"
                f"{_evidence_block(evidence)}"
            )
            started = time.perf_counter()
            drafted = llm.complete(system=_RESEARCHER_SYSTEM, user=user, schema=DraftClaims)
            logger.info("researcher: %d draft(s) over %d chunk(s) in %.2fs", len(drafted.claims), len(evidence), time.perf_counter() - started)
        return {"drafted": drafted, "evidence": evidence, "retrievals": retrievals}

    def verifier(state: LiveState) -> dict:
        if not state.evidence:
            # Nothing retrieved cleared the relevance threshold: drafting and
            # verifying claims against no Evidence would be theatre.
            return {}
        user = (
            f"Regulatory question: {state.question}\n\nRetrieved evidence:\n\n"
            f"{_evidence_block(state.evidence)}\n\n"
            f"Drafted claims:\n{json.dumps([claim.model_dump() for claim in state.drafted.claims])}"
        )
        started = time.perf_counter()
        verdicts = llm.complete(system=_VERIFIER_SYSTEM, user=user, schema=Verdicts)
        logger.info("verifier: %d decision(s) in %.2fs", len(verdicts.verdicts), time.perf_counter() - started)
        return {"verdicts": verdicts}

    builder = StateGraph(LiveState)
    builder.add_node("planner", planner)
    builder.add_node("researcher", researcher)
    builder.add_node("verifier", verifier)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "verifier")
    builder.add_edge("verifier", END)
    return builder.compile()


def run_live_analysis(request: AnalyzeRequest, *, llm: Llm, retriever: Retriever) -> AnalyzeResponse:
    """Answer an arbitrary Scenario through the Live workflow.

    The composition root injects the provider protocols; this function
    stays the one seam the API layer calls.
    """
    scenario_description = request.scenario.description or request.scenario.title or ""
    result = _build_graph(llm, retriever).invoke(
        LiveState(scenario_description=scenario_description, question=request.question)
    )
    state = result if isinstance(result, LiveState) else LiveState.model_validate(result)

    evidence_by_label = {item.label: item.chunk for item in state.evidence}
    findings, discarded, decisions = _decide_claims(state.drafted, state.verdicts, evidence_by_label)
    all_citations = [citation for finding in findings for citation in finding.citations]
    retrieved_chunks = [
        {
            "label": item.label,
            "source_id": item.chunk.source_id,
            "kind": item.chunk.kind.value,
            "number": item.chunk.provision_number,
            "section": item.chunk.title,
            "provision": _provision_label(item.chunk),
            "text": item.chunk.text[:1000],
        }
        for item in state.evidence
    ]

    insufficient = not state.evidence
    known_limitations = [ENGLISH_ONLY_LIMITATION] + ([INSUFFICIENT_EVIDENCE_LIMITATION] if insufficient else [])
    actions = list(NARROW_THE_QUESTION_ACTIONS) if insufficient else []

    queries = [t.query for t in state.plan]
    plan_summary = ", ".join(queries) if queries else "no research targets"
    if insufficient:
        summary = (
            f"Planner identified research targets ({plan_summary}); retrieval returned no Chunks "
            "relevant enough, so no Claims were drafted or verified — the Corpus holds nothing "
            "for this question."
        )
    else:
        summary = (
            f"Planner identified research targets ({plan_summary}); Researcher retrieved "
            f"{len(retrieved_chunks)} Chunk(s) via the retrieval tool and drafted "
            f"{len(state.drafted.claims)} claim(s); Verifier kept {len(findings)} Finding(s) "
            f"and recorded {len(discarded)} Unsupported claim(s) as rejected."
        )

    detailed_trace: list[dict[str, Any]] = [
        {"step": "planner", "action": "decompose the Regulatory question into research targets", "research_targets": queries},
        {
            "step": "researcher",
            "action": "retrieve Evidence exclusively via the retrieve_chunks tool, then draft Claims grounded in it",
            "retrieved": retrieved_chunks,
            "tool_calls": state.retrievals,
        },
        {
            "step": "verifier",
            "action": "check each Claim against the retrieved Evidence, tag its Strength, discard Unsupported claims",
            "claim_decisions": [decision.model_dump() for decision in decisions],
        },
    ]

    return AnalyzeResponse(
        answer=Answer(findings=findings, actions=actions, citations=all_citations),
        trace=Trace(workflow=LIVE_WORKFLOW_MARKER, summary=summary, unsupported_claims_discarded=discarded),
        detailed_trace=detailed_trace,
        known_limitations=known_limitations,
    )
