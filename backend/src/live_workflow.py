"""The Live workflow: Planner → Researcher → Verifier → Proposer as a LangGraph graph.

One pass answers an arbitrary Scenario: the Planner decomposes the
Regulatory question into research targets; the Researcher gathers Evidence
exclusively through the retrieval tool under an Evidence pool derived from
its own decomposition (``SEATS_PER_TARGET`` seats per Research target); the
Verifier checks every produced Claim against the retrieved Evidence, tags
its Strength, and discards Unsupported claims into the Execution trace;
the Proposer distills the kept Findings into referral Actions, each
grounded in a kept Finding's Citations (ADR-0004).

Three rules are enforced by application code, never trusted to the LLM:

- Citations are derived deterministically from Chunk provision metadata —
  the LLM only selects which Chunks support a Claim by their label, and a
  Claim whose references resolve to no Chunk cannot become a Finding.
- A Claim no verdict supports is an Unsupported claim: absent from the
  Answer, recorded in the Execution trace.
- A proposed Action whose citation references do not resolve to a
  moderate- or strong-anchored kept Finding is dropped and recorded in
  the detailed trace — an ungrounded referral never reaches the Answer,
  and the standing seek-counsel hand-off keeps the sheet never empty.

When retrieval returns nothing relevant enough (no Chunk clears the
relevance threshold), no LLM call drafts, verifies, or proposes anything:
the response is the Insufficient-evidence path — empty Findings, a Known
limitation explaining the gap, and Actions suggesting how to narrow the
question.
"""

import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Optional

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, field_validator

from .availability import ENGLISH_ONLY_LIMITATION, PROTOTYPE_LIMITATION
from .llm import Llm
from .models import AnalyzeRequest, AnalyzeResponse, Answer, Citation, ClaimDecision, Chunk, Finding, ProvisionKind, Strength, Trace, PROVISION_NUMBER_FIELDS, quote_snippet
from .query_log import RequestObservation
from .retrieval import Retriever

logger = logging.getLogger(__name__)

# The workflow marker the Execution trace carries for every served Live answer.
LIVE_WORKFLOW_MARKER = "planner -> researcher -> verifier -> proposer"

# Prompt-side bound: each retrieved Chunk contributes at most this many
# characters of Evidence to a prompt, so even a full pool stays within a
# modest, predictable context regardless of how long stored provisions run.
_MAX_CHUNK_CHARS = 1500

# The Evidence pool derives from the plan (ADR-0003): every Research target
# claims this many seats, so seat demand scales with the Planner's
# decomposition — breadth arriving as more targets, depth as finer-grained
# ones — and sharpening queries automatically widens the pool. Per-target
# retrieval depth stays a separate concern (retrieval.PER_TARGET_DEPTH):
# widening this pool never deepens crawling. Adjust only on scored evidence.
SEATS_PER_TARGET = 6

INSUFFICIENT_EVIDENCE_LIMITATION = (
    "Known limitation: the Corpus holds nothing relevant enough to answer this "
    "question, so no Findings were produced."
)

NARROW_THE_QUESTION_ACTIONS = [
    "Rephrase the Regulatory question using the vocabulary of the Corpus — EU AI Act, GDPR, or DORA terms such as 'high-risk AI system', 'automated decision-making', or 'ICT third-party risk'.",
    "Check whether the obligation you are asking about falls outside the Corpus (e.g. national requirements of a single member state).",
]

# The standing hand-off appended last to every Live answer that carries
# Findings or proposals: an Action in referral voice, never a Known
# limitation (ADR-0004) — it points at the professional, not at the system.
SEEK_COUNSEL_ACTION = (
    "Have a qualified legal professional verify these findings against the company's "
    "actual situation before acting on them."
)

# The most validated Actions an Answer serves, plus the hand-off last
# (ADR-0004): the sheet is capped, never empty.
MAX_ACTIONS = 5


# --- Workflow boundaries: every agent input/output crosses as a validated schema ---


class ResearchTarget(BaseModel):
    """One retrieval query the Planner wants researched."""

    query: str


class Plan(BaseModel):
    """The Planner's decomposition of the Regulatory question."""

    targets: list[ResearchTarget] = Field(default_factory=list)

    @field_validator("targets", mode="before")
    @classmethod
    def _coerce_strings(cls, v: list) -> list:
        return [{"query": t} if isinstance(t, str) else t for t in v]


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


class ActionProposal(BaseModel):
    """One candidate Action the Proposer distilled from kept Findings.

    ``citation_refs`` names the Citations of kept Findings that ground the
    Action; a proposal that resolves to no moderate- or strong-anchored
    kept Finding is rejected by application code, never the LLM.
    """

    action: str
    citation_refs: list[str] = Field(default_factory=list)


class ActionProposals(BaseModel):
    proposals: list[ActionProposal] = Field(default_factory=list)


class ActionDecision(BaseModel):
    """One Proposer decision in the detailed trace: kept, or rejected with why.

    Mirrors ``ClaimDecision``: rejected proposals must not vanish silently.
    """

    action: str
    status: Literal["kept", "rejected"]
    reason: Optional[str] = None


class LabeledEvidence(BaseModel):
    """One retrieved Chunk with the stable label the agents reference it by."""

    label: str
    chunk: Chunk


class LiveState(BaseModel):
    """The LangGraph state: one field per boundary between the agents."""

    scenario_description: str = ""
    question: str = ""
    plan: list[ResearchTarget] = Field(default_factory=list)
    evidence: list[LabeledEvidence] = Field(default_factory=list)
    retrievals: list[dict] = Field(default_factory=list)  # one record per retrieval-tool call
    drafted: DraftClaims = Field(default_factory=DraftClaims)
    verdicts: Verdicts = Field(default_factory=Verdicts)
    proposals: ActionProposals = Field(default_factory=ActionProposals)
    # The Proposer's one application-code derivation of the Verifier's output:
    # kept Findings, per-claim decisions, and the prompt's citation labels all
    # come from this single computation, so the grounding gate validates
    # exactly the labels the Proposer was shown.
    kept_findings: list[Finding] = Field(default_factory=list)
    claim_decisions: list[ClaimDecision] = Field(default_factory=list)
    citation_by_label: dict[str, Citation] = Field(default_factory=dict)
    anchor_by_label: dict[str, Strength] = Field(default_factory=dict)


# --- Prompts: JSON-only instructions; the client repeats the schema contract ---


_PLANNER_SYSTEM = (
    "You are the Planner of a regulatory research assistant. Decompose the regulatory "
    "question into 1-3 short keyword retrieval queries that would locate the relevant "
    "provisions (articles, recitals, annexes) in a corpus of EU regulations: the AI Act, "
    "GDPR, and DORA. Name concepts and likely provision subjects, not full sentences. "
    "When the question asks what applies or what obligations exist, do not stop at "
    "regime-level classification: give the operational duties each regulation imposes in "
    "this scenario their own query — e.g. 'AI Act deployer obligations human oversight "
    "logging' or 'automated decision information duties' — so duty-bearing provisions "
    "surface alongside the classification ones."
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

_PROPOSER_SYSTEM = (
    "You are the Proposer of a regulatory research assistant. You are given the Findings kept "
    "for an Answer, each badged with its Strength and carrying the Citations of the provisions "
    "that support it. "
    f"Propose at most {MAX_ACTIONS} referral Actions for legal professionals: each names "
    "something only a professional can settle — a contingency the Findings cannot determine, "
    "or verification of a cited provision against the company's actual situation. Ground each "
    "Action in the citation labels of the Findings it leans on: moderate Findings anchor "
    "contingency referrals, strong Findings anchor verify-against-facts referrals. Weak "
    "Findings may enrich an Action's wording but never support one alone. Never propose a "
    "compliance task, never presume a Regulation applies, never advise on your own authority."
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


# The corpus documents the short names are read from — same layout the demo
# path loads at startup; a fixed, repo-local set (ADR-0002).
_CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "regulations"

_source_short_names: Optional[dict[str, str]] = None


def _load_source_short_names() -> dict[str, str]:
    """Source id → short name, from the corpus documents' metadata.

    Loaded lazily once per process and cached: the lookup is a cheap dict
    hit after that, and a missing or unreadable corpus degrades to no short
    names — never an error on the request path.
    """
    global _source_short_names
    if _source_short_names is None:
        names: dict[str, str] = {}
        if _CORPUS_DIR.is_dir():
            for path in sorted(_CORPUS_DIR.glob("*.json")):
                try:
                    document = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                metadata = document.get("metadata") or {}
                if metadata.get("id") and metadata.get("shortName"):
                    names[str(metadata["id"])] = str(metadata["shortName"])
        _source_short_names = names
    return _source_short_names


def derive_citation(chunk: Chunk) -> Citation:
    """Build the Citation for one Chunk from its stored provision metadata."""
    return Citation.model_validate({
        "source_id": chunk.source_id,
        "source_short_name": _load_source_short_names().get(chunk.source_id),
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


# --- Proposer grounding: labels over the kept Findings' Citations ---


def _labelled_findings(
    findings: list[Finding],
) -> tuple[str, dict[str, Citation], dict[str, Strength]]:
    """The kept Findings block the Proposer reads, plus the label maps the
    grounding gate validates against.

    One enumeration serves both, so a label shown to the LLM always resolves
    the same way in validation. ``C`` labels are flat across Findings;
    ``anchor_by_label`` records the Strength of the Finding owning each
    Citation — the gate's anchor rule reads it.
    """
    citations: dict[str, Citation] = {}
    anchors: dict[str, Strength] = {}
    lines: list[str] = []
    index = 0
    for position, finding in enumerate(findings, 1):
        lines.append(f"[F{position} strength={finding.strength.value}] {finding.statement}")
        for citation in finding.citations:
            index += 1
            label = f"C{index}"
            citations[label] = citation
            anchors[label] = finding.strength
            name = citation.source_short_name or citation.source_id
            lines.append(f"  [{label}] {name} {citation.provision or ''}".rstrip())
    return "\n".join(lines), citations, anchors


_NO_RESOLVABLE_CITATION_REASON = "its citations resolve to no kept Finding"
_WEAK_ANCHOR_REASON = (
    "its citations resolve only to weak Findings; an Action needs a moderate or strong anchor"
)
_TOO_MANY_ACTIONS_REASON = f"the Answer serves at most {MAX_ACTIONS} Actions"


def _validate_proposals(
    proposals: ActionProposals,
    citation_by_label: dict[str, Citation],
    anchor_by_label: dict[str, Strength],
) -> tuple[list[str], list[ActionDecision]]:
    """The grounding gate: only anchored referral Actions reach the Answer.

    A proposal is kept when at least one of its citation references resolves
    to a kept Finding's Citation AND that Citation anchors a moderate or
    strong Finding — weak Findings never anchor an Action alone (ADR-0004).
    Unknown references are dropped from the grounding; proposals left with
    none are rejected and recorded in the detailed trace, so nothing
    proposed may vanish silently. At most ``MAX_ACTIONS`` validated Actions
    are served, the rest recorded as rejected.
    """
    served: list[str] = []
    decisions: list[ActionDecision] = []
    for proposal in proposals.proposals:
        unknown = [ref for ref in proposal.citation_refs if ref not in citation_by_label]
        resolved = [ref for ref in proposal.citation_refs if ref in citation_by_label]
        if unknown:
            logger.warning(
                "proposer cited labels matching no kept Finding's Citation (%s); dropped from the Action's grounding",
                ", ".join(unknown),
            )
        if not resolved:
            decisions.append(
                ActionDecision(
                    action=proposal.action,
                    status="rejected",
                    reason=_NO_RESOLVABLE_CITATION_REASON,
                )
            )
            continue
        if not any(anchor_by_label[ref] in (Strength.moderate, Strength.strong) for ref in resolved):
            decisions.append(
                ActionDecision(action=proposal.action, status="rejected", reason=_WEAK_ANCHOR_REASON)
            )
            continue
        if len(served) >= MAX_ACTIONS:
            decisions.append(
                ActionDecision(action=proposal.action, status="rejected", reason=_TOO_MANY_ACTIONS_REASON)
            )
            continue
        served.append(proposal.action)
        decisions.append(ActionDecision(action=proposal.action, status="kept"))
    return served, decisions


# --- The graph: START → planner → researcher → verifier → proposer → END ---


def _build_graph(llm: Llm, retriever: Retriever, observation: RequestObservation | None):
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

        # Fair-share fill over the plan-derived pool: every target claims its
        # share of seats before any target's broad results backfill the
        # remainder, so a greedy first query cannot starve the rest.
        evidence: list[LabeledEvidence] = []

        def take(chunk: Chunk) -> None:
            evidence.append(LabeledEvidence(label=f"E{len(evidence) + 1}", chunk=chunk))

        if per_target:
            # The pool is SEATS_PER_TARGET × targets, so every target's fair
            # share of round one is exactly SEATS_PER_TARGET seats.
            fair_share = SEATS_PER_TARGET
            pool_size = SEATS_PER_TARGET * len(per_target)
            for fresh in per_target:
                for chunk in fresh[:fair_share]:
                    take(chunk)
            if len(evidence) < pool_size:
                for fresh in per_target:
                    for chunk in fresh[fair_share:]:
                        take(chunk)
                        if len(evidence) >= pool_size:
                            break
                    if len(evidence) >= pool_size:
                        break

        drafted = DraftClaims()
        if observation is not None:
            # Report the pool as soon as it exists, not after the workflow
            # completes: a request that dies later still logs what it
            # retrieved — and spent tokens retrieving.
            observation.retrieved_chunks = len(evidence)
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

    def proposer(state: LiveState) -> dict:
        """Distill the kept Findings into referral Actions grounded in their Citations.

        The one application-code derivation of the Verifier's output happens
        here: kept Findings, per-claim decisions, and the prompt's citation
        labels all come from this single computation, carried through the
        state so the grounding gate validates exactly the labels the LLM saw.
        """
        evidence_by_label = {item.label: item.chunk for item in state.evidence}
        findings, _, decisions = _decide_claims(state.drafted, state.verdicts, evidence_by_label)
        if not findings:
            # No kept Finding anchors anything: the hand-off alone is served,
            # and no LLM call is made over nothing.
            return {
                "proposals": ActionProposals(),
                "kept_findings": findings,
                "claim_decisions": decisions,
            }
        block, citation_by_label, anchor_by_label = _labelled_findings(findings)
        user = (
            f"Regulatory question: {state.question}\n\n"
            f"Kept Findings with their supporting Citations:\n\n{block}"
        )
        started = time.perf_counter()
        proposals = llm.complete(system=_PROPOSER_SYSTEM, user=user, schema=ActionProposals)
        logger.info(
            "proposer: %d proposal(s) over %d finding(s) in %.2fs",
            len(proposals.proposals),
            len(findings),
            time.perf_counter() - started,
        )
        return {
            "proposals": proposals,
            "kept_findings": findings,
            "claim_decisions": decisions,
            "citation_by_label": citation_by_label,
            "anchor_by_label": anchor_by_label,
        }

    builder = StateGraph(LiveState)
    builder.add_node("planner", planner)
    builder.add_node("researcher", researcher)
    builder.add_node("verifier", verifier)
    builder.add_node("proposer", proposer)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "verifier")
    builder.add_edge("verifier", "proposer")
    builder.add_edge("proposer", END)
    return builder.compile()


def run_live_analysis(
    request: AnalyzeRequest,
    *,
    llm: Llm,
    retriever: Retriever,
    observation: RequestObservation | None = None,
) -> AnalyzeResponse:
    """Answer an arbitrary Scenario through the Live workflow.

    The composition root injects the provider protocols; this function
    stays the one seam the API layer calls. ``observation`` receives the
    per-request observability counts — the size of the Evidence pool this
    pass actually reasoned over.
    """
    scenario_description = request.scenario.description or request.scenario.title or ""
    result = _build_graph(llm, retriever, observation).invoke(
        LiveState(scenario_description=scenario_description, question=request.question)
    )
    state = result if isinstance(result, LiveState) else LiveState.model_validate(result)

    # The Verifier's output was decided once, by the Proposer: kept Findings,
    # per-claim decisions, and the citation labels the grounding gate reads
    # all travel through the state from that single computation.
    findings = state.kept_findings
    decisions = state.claim_decisions
    discarded = [decision.claim for decision in decisions if decision.status == "rejected"]
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
    known_limitations = [ENGLISH_ONLY_LIMITATION, PROTOTYPE_LIMITATION] + (
        [INSUFFICIENT_EVIDENCE_LIMITATION] if insufficient else []
    )

    proposal_decisions: list[ActionDecision] = []
    if insufficient:
        # The Insufficient-evidence path is unchanged: narrowing suggestions,
        # and the Proposer made no LLM call over nothing.
        actions = list(NARROW_THE_QUESTION_ACTIONS)
    else:
        actions, proposal_decisions = _validate_proposals(
            state.proposals, state.citation_by_label, state.anchor_by_label
        )
        # Never empty (ADR-0004): the standing seek-counsel hand-off closes
        # the sheet, whether proposals survived the gate or none were made.
        actions = actions + [SEEK_COUNSEL_ACTION]

    queries = [t.query for t in state.plan]
    plan_summary = ", ".join(queries) if queries else "no research targets"
    if insufficient:
        summary = (
            f"Planner identified research targets ({plan_summary}); retrieval returned no Chunks "
            "relevant enough, so no Claims were drafted or verified — the Corpus holds nothing "
            "for this question."
        )
    else:
        rejected_proposals = sum(1 for d in proposal_decisions if d.status == "rejected")
        summary = (
            f"Planner identified research targets ({plan_summary}); Researcher retrieved "
            f"{len(retrieved_chunks)} Chunk(s) via the retrieval tool and drafted "
            f"{len(state.drafted.claims)} claim(s); Verifier kept {len(findings)} Finding(s) "
            f"and recorded {len(discarded)} Unsupported claim(s) as rejected; Proposer "
            f"distilled {len(actions) - 1} referral Action(s) from the kept Findings "
            f"and recorded {rejected_proposals} ungrounded proposal(s) as rejected."
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
        {
            "step": "proposer",
            "action": "distill the kept Findings into referral Actions, each grounded in a kept Finding's Citations",
            "action_decisions": [decision.model_dump() for decision in proposal_decisions],
        },
    ]

    return AnalyzeResponse(
        answer=Answer(findings=findings, actions=actions, citations=all_citations),
        trace=Trace(workflow=LIVE_WORKFLOW_MARKER, summary=summary, unsupported_claims_discarded=discarded),
        detailed_trace=detailed_trace,
        known_limitations=known_limitations,
    )
