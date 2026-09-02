"""The Live workflow: Planner → Researcher → Verifier → Proposer → Summarizer as a LangGraph graph.

One pass answers an arbitrary Scenario: the Planner decomposes the
Regulatory question into research targets; the Researcher gathers Evidence
exclusively through the retrieval tool under an Evidence pool derived from
its own decomposition (``SEATS_PER_TARGET`` seats per Research target); the
Verifier checks every produced Claim against the retrieved Evidence, tags
its Strength, and discards Unsupported claims into the Execution trace;
the Proposer distills the kept Findings into referral Actions, each
grounded in a kept Finding's Citations (ADR-0004); and the Summarizer
turns the kept, cited Findings into Provision relevance — one grounded
statement per cited provision, drawing only on the content of the Findings
citing it (#47) — and rates each cited provision's Citation strength
(ADR-0011).

Four rules are enforced by application code, never trusted to the LLM:

- Citations are derived deterministically from Chunk provision metadata —
  the LLM only selects which Chunks support a Claim by their label, and a
  Claim whose references resolve to no Chunk cannot become a Finding.
- A Claim no verdict supports is an Unsupported claim: absent from the
  Answer, recorded in the Execution trace.
- A proposed Action whose citation references do not resolve to a
  moderate- or strong-anchored kept Finding — or whose declared referral
  kind contradicts its strongest anchor — is dropped and recorded in
  the detailed trace: an ungrounded referral never reaches the Answer,
  and the standing seek-counsel hand-off keeps the sheet never empty.
- A relevance statement whose reference resolves to no cited provision is
  dropped and recorded: Provision relevance aggregates only the Findings
  citing a provision, never material outside them. The rating the
  Summarizer gives each cited provision's Citation strength rides only when
  it is one of the three levels — a missing or invalid rating keeps the
  relevance, carries no strength, and is recorded with a reason (ADR-0011).
  On Summarizer failure the Answer ships without summaries plus a Known
  limitation — the run does not fail.

When retrieval returns nothing relevant enough (no Chunk clears the
relevance threshold), no LLM call drafts, verifies, or proposes anything:
the response is the Insufficient-evidence path — empty Findings, a Known
limitation explaining the gap, and Actions suggesting how to narrow the
question.
"""

import json
import logging
import re
import time
from collections import Counter
from typing import Any, Literal, NamedTuple, Optional

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, field_validator, model_validator

from .availability import ENGLISH_ONLY_LIMITATION, PROTOTYPE_LIMITATION
from .corpus import source_short_names
from .llm import Llm
from .models import AnalyzeRequest, AnalyzeResponse, Answer, Citation, ClaimDecision, Chunk, Finding, GroundedSummary, ProvisionKind, ProvisionTarget, STRENGTH_ORDER, Strength, Trace, PROVISION_NUMBER_FIELDS, answer_citations, quote_snippet
from .progress import PhaseReport, ProgressSink
from .query_log import RequestObservation
from .retrieval import Retriever

logger = logging.getLogger(__name__)

# The workflow marker the Execution trace carries for every served Live answer.
LIVE_WORKFLOW_MARKER = "planner -> researcher -> verifier -> proposer -> summarizer"

# Prompt-side bound: each retrieved Chunk contributes at most this many
# characters of Evidence to a prompt, so even a full pool stays within a
# modest, predictable context regardless of how long stored provisions run.
_MAX_CHUNK_CHARS = 1500

# The Evidence pool derives from the plan (ADR-0003): every Research target
# claims this many seats, so seat demand scales with the Planner's
# decomposition — breadth arriving as more targets, depth as finer-grained
# ones — and sharper targets automatically widen the pool. Per-target
# retrieval depth stays a separate concern (retrieval.PER_TARGET_DEPTH):
# widening this pool never deepens crawling. Adjust only on scored evidence.
SEATS_PER_TARGET = 6

INSUFFICIENT_EVIDENCE_LIMITATION = (
    "Known limitation: the Corpus holds nothing relevant enough to answer this "
    "question, so no Findings were produced."
)

SUMMARIZER_FAILURE_LIMITATION = (
    "Known limitation: Provision relevance is unavailable for this answer — the Summarizer "
    "produced no usable statement; the Findings and Citations are unaffected."
)

SUMMARIZER_PARTIAL_LIMITATION = (
    "Known limitation: Provision relevance is incomplete for this answer — the Summarizer "
    "covered {covered} of {total} cited provision(s)."
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

# The Planner's declared budget: 1-3 Research targets. A provider response
# over this limit is corrected once by truncation — the Evidence pool derives
# from the accepted plan, never from an over-limit provider response.
MAX_RESEARCH_TARGETS = 3


# --- Workflow boundaries: every agent input/output crosses as a validated schema ---


class ResearchTarget(BaseModel):
    """One regulation-scoped line of inquiry the Planner derived from the
    Regulatory question; ``query`` carries the keyword search string used
    for retrieval.

    A bare string is tolerated as shorthand for ``{"query": <string>}`` —
    live solar-pro4 emits the keywords without the wrapper object despite
    the structural shape instruction.
    """

    query: str

    @model_validator(mode="before")
    @classmethod
    def accept_bare_string(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"query": data}
        return data


class Plan(BaseModel):
    """The Planner's decomposition of the Regulatory question."""

    targets: list[ResearchTarget] = Field(default_factory=list)


class DraftClaim(BaseModel):
    """A Candidate claim the Researcher drafted from retrieved Evidence."""

    statement: str
    evidence_refs: list[str] = Field(default_factory=list)


class DraftClaims(BaseModel):
    claims: list[DraftClaim] = Field(default_factory=list)


def _unwrap_strength_level(value: Any) -> Any:
    """Keep the bare level when the live model wraps its strength with a
    rationale — the one repair both the Verdict's and the Summarizer's
    strength fields apply before the application code judges the value."""
    if isinstance(value, dict) and "level" in value:
        return value["level"]
    return value


class Verdict(BaseModel):
    """The Verifier's decision about one drafted Claim."""

    statement: str
    supported: bool
    strength: Strength = Strength.moderate
    evidence_refs: list[str] = Field(default_factory=list)

    @field_validator("strength", mode="before")
    @classmethod
    def accept_strength_object(cls, value: Any) -> Any:
        return _unwrap_strength_level(value)


class Verdicts(BaseModel):
    verdicts: list[Verdict] = Field(default_factory=list)


class ActionProposal(BaseModel):
    """One candidate Action the Proposer distilled from kept Findings.

    ``kind`` declares the referral voice: ``contingency`` when the strongest
    anchor Finding is moderate, ``verify_against_facts`` when it is strong —
    the gate rejects a mismatch, never trusting the LLM. ``citation_refs``
    names the Citations of kept Findings that ground the Action; a proposal
    that resolves to no moderate- or strong-anchored kept Finding is
    rejected by application code, never the LLM.
    """

    action: str
    kind: Literal["contingency", "verify_against_facts"]
    citation_refs: list[str] = Field(default_factory=list)


class ActionProposals(BaseModel):
    proposals: list[ActionProposal] = Field(default_factory=list)


class ProvisionSummary(BaseModel):
    """One candidate Provision relevance the Summarizer emitted, referencing
    the cited provision by the label it was shown (P1, P2, ...).

    ``strength`` is the provision's rated Citation strength (ADR-0011) — a
    bare strong/moderate/weak string. The field stays deliberately loose:
    a malformed rating can never reject the relevance statement around it,
    and the grounding gate validates the value instead of defaulting it.
    """

    ref: str
    relevance: str
    strength: Any = None

    @field_validator("strength", mode="before")
    @classmethod
    def accept_strength_object(cls, value: Any) -> Any:
        return _unwrap_strength_level(value)


class Summaries(BaseModel):
    summaries: list[ProvisionSummary] = Field(default_factory=list)


class SummaryDecision(BaseModel):
    """One Summarizer decision in the detailed trace: kept, or rejected with
    why. Mirrors ``ClaimDecision``/``ActionDecision``: rejected relevance
    never vanishes silently. ``strength`` is the rating the provision
    carries (ADR-0011); on a kept decision whose rating is missing or
    invalid it is None and ``reason`` names the gap — the relevance ships,
    the strength does not, and the trace says why."""

    ref: str
    status: Literal["kept", "rejected"]
    reason: Optional[str] = None
    strength: Optional[Strength] = None


class ActionDecision(BaseModel):
    """One Proposer decision in the detailed trace: kept, or rejected with why.

    Mirrors ``ClaimDecision``: rejected proposals must not vanish silently.
    ``dropped_refs`` carries the citation labels that matched no kept Finding
    and were dropped from the proposal's grounding — kept proposals record
    them too, so a partial grounding failure never vanishes either.
    """

    action: str
    status: Literal["kept", "rejected"]
    reason: Optional[str] = None
    dropped_refs: list[str] = Field(default_factory=list)


class LabeledEvidence(BaseModel):
    """One retrieved Chunk with the stable label the agents reference it by."""

    label: str
    chunk: Chunk


class Grounding(BaseModel):
    """One label the Proposer reads, paired with what validates it: the kept
    Finding's Citation plus the Strength of the Finding owning it — the two
    values the grounding gate checks every proposal reference against."""

    citation: Citation
    anchor: Strength


class LiveState(BaseModel):
    """The LangGraph state: one field per boundary between the agents."""

    scenario_description: str = ""
    question: str = ""
    plan: list[ResearchTarget] = Field(default_factory=list)
    plan_correction: Optional[str] = None
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
    grounding_by_label: dict[str, Grounding] = Field(default_factory=dict)
    # The Proposer's no-findings edge (issue #24): with no kept Finding to
    # anchor an Action, the node itself emits the standing seek-counsel
    # hand-off alone — the composition serves this sheet verbatim, except on
    # the Insufficient-evidence path, whose narrowing Actions it owns.
    actions: list[str] = Field(default_factory=list)
    # The Summarizer's output (issue #47): one grounded relevance statement
    # per cited provision, the per-summary decisions for the detailed trace,
    # and the failure flag that degrades the Answer without failing the run.
    relevance: list[GroundedSummary] = Field(default_factory=list)
    summary_decisions: list[SummaryDecision] = Field(default_factory=list)
    summarizer_failed: bool = False


# --- Prompts: JSON-only instructions; the client repeats the schema contract ---


_PLANNER_SYSTEM = (
    "You are the Planner of a regulatory research assistant. Decompose the regulatory "
    "question into 1-3 short keyword Research targets that would locate the relevant "
    "provisions (articles, recitals, annexes) in a corpus of EU regulations: the AI Act, "
    "GDPR, and DORA. Name concepts and likely provision subjects, not full sentences. "
    "When the question asks what applies or what obligations exist, do not stop at "
    "regime-level classification: give the operational duties each regulation imposes in "
    "this scenario their own Research target — e.g. 'AI Act deployer obligations human "
    "oversight logging' or 'automated decision information duties' — so duty-bearing "
    "provisions surface alongside the classification ones."
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
    "provision bears on the statement, false when none does either way; strength — a bare "
    "string, exactly one of 'strong', 'moderate', or 'weak', never an object or rationale; "
    "use 'strong' when the provisions directly and explicitly establish the claim, 'moderate' "
    "when derived from provisions read together or contingent on facts the corpus cannot settle, "
    "'weak' when the provisions supply framing only (definitions, vocabulary); evidence_refs — "
    "the labels of the supporting provisions, empty for unsupported claims."
)

_PROPOSER_SYSTEM = (
    "You are the Proposer of a regulatory research assistant. You are given the Findings kept "
    "for an Answer, each badged with its Strength and carrying the Citations of the provisions "
    "that support it. Each Finding has a label like [F1]; under it, each Citation has its own "
    "label like [C1]. "
    f"Propose at most {MAX_ACTIONS} referral Actions for legal professionals: each names "
    "something only a professional can settle — a contingency the Findings cannot determine, "
    "or verification of a cited provision against the company's actual situation. Ground each "
    "Action using only the Citation labels (C1, C2, ...) — never the Finding labels (F1, F2, ...); "
    "the gate rejects a Finding label as invalid grounding. Moderate Findings anchor "
    "contingency referrals, strong Findings anchor verify-against-facts referrals, so set "
    "kind='contingency' when the strongest anchor is moderate and "
    "kind='verify_against_facts' when it is strong — the gate rejects a mismatch. Weak "
    "Findings may enrich an Action's wording but never support one alone. Never propose a "
    "compliance task, never presume a Regulation applies, never advise on your own authority."
)

_SUMMARIZER_SYSTEM = (
    "You are the Summarizer of a regulatory research assistant. You are given the provisions an "
    "Answer cites, each with a label like [P1] and the statements of the Findings that cite it. "
    "For each provision, write ONE answer-wide Provision relevance statement: why the provision "
    "matters to the overall Answer, aggregating across the Findings that cite it. Alongside it, "
    "rate the provision's Citation strength as a bare string, exactly one of 'strong', 'moderate', "
    "or 'weak', never an object or rationale: 'strong' when the provision directly imposes or "
    "decides the obligations the Answer turns on, 'moderate' when it is a supporting duty or factor "
    "the Answer relies on, 'weak' when it is definitional or framing. Ground each statement and its "
    "rating ONLY in the content of the Findings citing that provision — never on the provision's "
    "own text, never on other provisions, never on outside knowledge. Set ref to the provision's "
    "label (P1, P2, ...) — never a label that was not given to you. Never write more than one "
    "statement per provision."
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
        "source_short_name": source_short_names().get(chunk.source_id),
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
) -> tuple[str, dict[str, Grounding]]:
    """The kept Findings block the Proposer reads, plus the groundings the
    gate validates against.

    One enumeration serves both, so a label shown to the LLM always resolves
    the same way in validation. ``C`` labels are flat across Findings; each
    Grounding pairs the Citation with the Strength of the Finding owning it —
    the gate's anchor rule reads it.
    """
    groundings: dict[str, Grounding] = {}
    lines: list[str] = []
    index = 0
    for position, finding in enumerate(findings, 1):
        lines.append(f"[F{position} strength={finding.strength.value}] {finding.statement}")
        for citation in finding.citations:
            index += 1
            label = f"C{index}"
            groundings[label] = Grounding(citation=citation, anchor=finding.strength)
            name = citation.source_short_name or citation.source_id
            lines.append(f"  [{label}] {name} {citation.provision or ''}".rstrip())
    return "\n".join(lines), groundings


class SummarizerPrompt(NamedTuple):
    """The Summarizer's prompt block plus the ref → target map the gate
    validates against. One enumeration serves both, so the gate validates
    exactly the labels the LLM saw — the pair never travels apart."""

    block: str
    targets_by_ref: dict[str, ProvisionTarget]


def _labelled_provisions(
    findings: list[Finding],
) -> SummarizerPrompt:
    """The cited-provisions block the Summarizer reads, plus the ref → target
    map the gate validates against.

    One enumeration serves both, so a label shown to the LLM always resolves
    the same way in validation. Provisions group the kept Findings' Citations
    by structural target (first mention, P-labels flat), each listed with the
    statements of the Findings citing it — the only material the Summarizer
    may draw on. A Finding citing one provision through several Citations is
    listed once: the statements are the grounding material, and duplication
    in the prompt can only weight them, never inform.
    """
    targets_by_ref: dict[str, ProvisionTarget] = {}
    blocks: dict[str, list[str]] = {}
    listed_findings: dict[str, set[int]] = {}
    for position, finding in enumerate(findings):
        for citation in finding.citations:
            target = citation.provision_target
            ref = next(
                (ref for ref, known in targets_by_ref.items() if known == target),
                None,
            )
            if ref is None:
                ref = f"P{len(targets_by_ref) + 1}"
                targets_by_ref[ref] = target
                name = citation.source_short_name or citation.source_id
                blocks[ref] = [f"[{ref}] {name} {citation.provision or ''}".rstrip(), "  cited by:"]
                listed_findings[ref] = set()
            if position not in listed_findings[ref]:
                listed_findings[ref].add(position)
                blocks[ref].append(f"  - ({finding.strength.value}) {finding.statement}")
    lines = [line for block in blocks.values() for line in block]
    return SummarizerPrompt(block="\n".join(lines), targets_by_ref=targets_by_ref)


_NO_RESOLVABLE_CITATION_REASON = "its citations resolve to no kept Finding"
_INVALID_GROUNDING_REASON = (
    "invalid grounding: {finding_refs} name Findings, not their Citations — "
    "use the C-labels shown under each Finding"
)
_WEAK_ANCHOR_REASON = (
    "its citations resolve only to weak Findings; an Action needs a moderate or strong anchor"
)
_KIND_ANCHOR_MISMATCH_REASON = (
    "its referral kind ({kind}) does not match its strongest anchor ({anchor})"
)
_TOO_MANY_ACTIONS_REASON = f"the Answer serves at most {MAX_ACTIONS} Actions"

_FINDING_LABEL_PATTERN = re.compile(r"^F\d+$")


def _validate_proposals(
    proposals: ActionProposals,
    grounding_by_label: dict[str, Grounding],
) -> tuple[list[str], list[ActionDecision]]:
    """The grounding gate: only anchored referral Actions reach the Answer.

    A proposal is kept when at least one of its citation references resolves
    to a kept Finding's Citation AND that Citation anchors a moderate or
    strong Finding — weak Findings never anchor an Action alone (ADR-0004) —
    AND its declared referral kind matches its strongest anchor: strong
    anchors verify-against-facts referrals, moderate anchors contingency
    referrals. Unknown references are dropped from the grounding; proposals
    left with none are rejected and recorded in the detailed trace, so
    nothing proposed may vanish silently. A proposal whose unknown references
    are Finding labels (F1, F2) — not Citation labels (C1, C2) — is rejected
    with a reason that identifies the invalid grounding, so the LLM's
    confusion is visible in the trace rather than silently producing an empty
    Action set. At most ``MAX_ACTIONS`` validated Actions are served, the rest
    recorded as rejected.
    """
    served: list[str] = []
    decisions: list[ActionDecision] = []

    def reject(proposal: ActionProposal, reason: str, dropped: list[str]) -> ActionDecision:
        return ActionDecision(
            action=proposal.action, status="rejected", reason=reason, dropped_refs=dropped
        )

    for proposal in proposals.proposals:
        unknown = [ref for ref in proposal.citation_refs if ref not in grounding_by_label]
        resolved = [ref for ref in proposal.citation_refs if ref in grounding_by_label]
        finding_refs = [ref for ref in unknown if _FINDING_LABEL_PATTERN.match(ref)]
        if unknown:
            logger.warning(
                "proposer cited labels matching no kept Finding's Citation (%s); dropped from the Action's grounding",
                ", ".join(unknown),
            )
        if not resolved:
            if finding_refs:
                reason = _INVALID_GROUNDING_REASON.format(finding_refs=", ".join(finding_refs))
            else:
                reason = _NO_RESOLVABLE_CITATION_REASON
            decisions.append(reject(proposal, reason, unknown))
            continue
        anchors = [grounding_by_label[ref].anchor for ref in resolved]
        if not any(anchor in (Strength.moderate, Strength.strong) for anchor in anchors):
            decisions.append(reject(proposal, _WEAK_ANCHOR_REASON, unknown))
            continue
        strongest = Strength.strong if Strength.strong in anchors else Strength.moderate
        expected_kind = "verify_against_facts" if strongest is Strength.strong else "contingency"
        if proposal.kind != expected_kind:
            decisions.append(
                reject(proposal, _KIND_ANCHOR_MISMATCH_REASON.format(kind=proposal.kind, anchor=strongest.value), unknown)
            )
            continue
        if len(served) >= MAX_ACTIONS:
            decisions.append(reject(proposal, _TOO_MANY_ACTIONS_REASON, unknown))
            continue
        served.append(proposal.action)
        decisions.append(
            ActionDecision(action=proposal.action, status="kept", dropped_refs=unknown)
        )
    return served, decisions


# --- Summarizer grounding: labels over the cited provisions ---


_UNKNOWN_REF_REASON = "the label names no cited provision"
_DUPLICATE_REF_REASON = "this provision already carries its relevance statement"
_EMPTY_RELEVANCE_REASON = "the relevance statement is empty"
_MISSING_RATING_REASON = (
    "the Citation strength rating is missing; the provision keeps its "
    "relevance without a strength"
)
_INVALID_RATING_REASON = (
    "the Citation strength rating {value!r} is not one of strong, moderate, "
    "or weak; the provision keeps its relevance without a strength"
)


def _validated_rating(raw: Any) -> tuple[Optional[Strength], Optional[str]]:
    """The rating one emitted summary carries, validated against the three
    levels (ADR-0011): a bare strong/moderate/weak string — or a level the
    schema already unwrapped from a wrapped object — becomes its Strength;
    anything else is no rating. Returns the gap reason for the detailed
    trace instead of a default: relevance and strength degrade
    independently."""
    if raw is None:
        return None, _MISSING_RATING_REASON
    try:
        return Strength(raw), None
    except (ValueError, TypeError):
        return None, _INVALID_RATING_REASON.format(value=raw)


def _validate_summaries(
    summaries: Summaries,
    targets_by_ref: dict[str, ProvisionTarget],
) -> tuple[dict[ProvisionTarget, GroundedSummary], list[SummaryDecision]]:
    """The Summarizer's grounding gate: only cited provisions receive
    relevance, and a rating rides it only when it is one of the three
    levels (ADR-0011).

    A summary's reference must resolve to a cited provision through the same
    label map the prompt showed the LLM — a reference to anything else (a
    Finding label, an invented ref) is rejected and recorded in the detailed
    trace, so ungrounded relevance never reaches the Answer. One grounded
    statement per cited provision: a second statement for an
    already-summarised provision is rejected — the first stands, nothing
    merges or overwrites. A kept statement carries its rating as given; a
    missing or invalid rating keeps the relevance, carries no strength, and
    is recorded with its reason — no default value exists anywhere.
    """
    grounded: dict[ProvisionTarget, GroundedSummary] = {}
    decisions: list[SummaryDecision] = []
    for summary in summaries.summaries:
        target = targets_by_ref.get(summary.ref)
        if target is None:
            decisions.append(
                SummaryDecision(ref=summary.ref, status="rejected", reason=_UNKNOWN_REF_REASON)
            )
            continue
        if not summary.relevance.strip():
            decisions.append(
                SummaryDecision(ref=summary.ref, status="rejected", reason=_EMPTY_RELEVANCE_REASON)
            )
            continue
        if target in grounded:
            decisions.append(
                SummaryDecision(ref=summary.ref, status="rejected", reason=_DUPLICATE_REF_REASON)
            )
            continue
        strength, rating_reason = _validated_rating(summary.strength)
        grounded[target] = GroundedSummary(
            target=target, relevance=summary.relevance, strength=strength
        )
        decisions.append(
            SummaryDecision(ref=summary.ref, status="kept", reason=rating_reason, strength=strength)
        )
    return grounded, decisions


# --- The Summarizer's outcome: classified once, described by every surface ---


class SummarizerOutcome(NamedTuple):
    """The one classification of the Summarizer's pass, derived once from the
    state and read by every surface that describes it — the Known limitation,
    the Execution-trace sentence, and the detailed-trace step. ``covered`` of
    ``total`` cited provisions carry a relevance statement."""

    status: Literal["failed", "skipped", "partial", "complete"]
    covered: int
    total: int


def _classify_summarizer(state: LiveState, total: int) -> SummarizerOutcome:
    """Classify the Summarizer's pass: ``failed`` (it produced nothing
    usable), ``skipped`` (no cited provision needed a statement), ``partial``
    (some cited provisions are covered), or ``complete``. The three surfaces
    that describe the pass re-derive no conditions of their own."""
    if state.summarizer_failed:
        return SummarizerOutcome("failed", 0, total)
    if total == 0:
        return SummarizerOutcome("skipped", 0, 0)
    covered = len(state.relevance)
    return SummarizerOutcome(
        "partial" if covered < total else "complete", covered, total
    )


def _summarizer_limitation(outcome: SummarizerOutcome) -> Optional[str]:
    """The Known limitation the outcome owes the Answer, if any."""
    if outcome.status == "failed":
        return SUMMARIZER_FAILURE_LIMITATION
    if outcome.status == "partial":
        return SUMMARIZER_PARTIAL_LIMITATION.format(
            covered=outcome.covered, total=outcome.total
        )
    return None


def _summarizer_sentence(outcome: SummarizerOutcome) -> str:
    """The Execution-trace's sentence about the Summarizer's pass."""
    if outcome.status == "failed":
        return (
            "the Summarizer produced no usable Provision relevance statement, so the Answer "
            "ships without them (a Known limitation names it)"
        )
    if outcome.status == "partial":
        return (
            f"the Summarizer covered {outcome.covered} of {outcome.total} cited "
            "provision(s) with Provision relevance (a Known limitation names the gap)"
        )
    return (
        f"Summarizer wrote {outcome.covered} Provision relevance "
        f"statement(s) for the {outcome.total} cited provision(s)"
    )


def _summarizer_step_action(outcome: SummarizerOutcome) -> str:
    """The detailed trace's honest description of the Summarizer's pass."""
    if outcome.status == "failed":
        return (
            "the Summarizer produced no usable relevance statement: the Answer ships "
            "without Provision relevance, named by a Known limitation"
        )
    if outcome.status == "skipped":
        return (
            "no cited provision needed a relevance statement, and no LLM call was made"
        )
    if outcome.status == "partial":
        return (
            f"covered {outcome.covered} of {outcome.total} cited provision(s); "
            "the Answer ships with Provision relevance incomplete, named by a Known limitation"
        )
    return (
        "aggregate the Findings citing each provision into one grounded "
        "Provision relevance statement, drawn only on those Findings"
    )


# --- The graph: START → planner → researcher → verifier → proposer → END ---


def _build_graph(
    llm: Llm,
    retriever: Retriever,
    observation: RequestObservation | None,
    progress: ProgressSink | None,
):
    def planner(state: LiveState) -> dict:
        if progress:
            progress(PhaseReport(phase="planner", message="decomposing the Regulatory question into research targets"))
        started = time.perf_counter()
        user = f"Company/product scenario: {state.scenario_description or '(not described)'}\nRegulatory question: {state.question}"
        plan = llm.complete(system=_PLANNER_SYSTEM, user=user, schema=Plan)
        targets = plan.targets
        correction = None
        if len(targets) > MAX_RESEARCH_TARGETS:
            correction = f"Planner response truncated from {len(targets)} to {MAX_RESEARCH_TARGETS} Research targets"
            logger.warning("planner: %d targets exceeds budget of %d; truncating", len(targets), MAX_RESEARCH_TARGETS)
            targets = targets[:MAX_RESEARCH_TARGETS]
        logger.info("planner: %d target(s) in %.2fs", len(targets), time.perf_counter() - started)
        return {"plan": targets, "plan_correction": correction}

    def researcher(state: LiveState) -> dict:
        """Gather Evidence exclusively through the retrieval tool, then draft Claims."""
        if progress:
            progress(PhaseReport(phase="researcher", message="retrieving Evidence from the Corpus and drafting candidate Claims"))
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
        # remainder, so a greedy first target cannot starve the rest.
        evidence: list[LabeledEvidence] = []

        def take(chunk: Chunk) -> None:
            evidence.append(LabeledEvidence(label=f"E{len(evidence) + 1}", chunk=chunk))

        if per_target:
            # The pool derives from the plan (ADR-0003): SEATS_PER_TARGET
            # seats per Research target, and round one hands each target its
            # fair share of that pool — the pool divided by the planned
            # target count — before any target's broad results backfill the
            # remainder.
            pool_size = SEATS_PER_TARGET * len(state.plan)
            fair_share = max(1, pool_size // len(state.plan))
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
        if progress:
            progress(PhaseReport(phase="verifier", message="checking each drafted Claim against the retrieved Evidence and tagging its Strength"))
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
        if progress:
            progress(PhaseReport(phase="proposer", message="distilling the kept Findings into referral Actions grounded in their Citations"))
        findings, _, decisions = _decide_claims(state.drafted, state.verdicts, evidence_by_label)
        if not findings:
            # No kept Finding anchors anything: the node itself emits the
            # standing seek-counsel hand-off alone (ADR-0004), and no LLM
            # call is made over nothing. The Insufficient-evidence path
            # overrides this emission with its narrowing Actions.
            return {
                "proposals": ActionProposals(),
                "kept_findings": findings,
                "claim_decisions": decisions,
                "actions": [SEEK_COUNSEL_ACTION],
            }
        block, grounding_by_label = _labelled_findings(findings)
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
            "grounding_by_label": grounding_by_label,
        }

    def summarizer(state: LiveState) -> dict:
        """Turn the kept, cited Findings into Provision relevance (issue #47)
        and rate each cited provision's Citation strength (ADR-0011).

        The block and the ref map come from one computation over the kept
        Findings — the gate validates exactly the labels the LLM saw. The
        prompt carries nothing but that block: the citing Findings' statements
        are the relevance statements' and the ratings' only permitted
        grounding (CONTEXT.md), so the question and the Evidence text stay
        out of the window. A Summarizer failure never fails the run: the
        Answer ships without summaries plus a Known limitation.
        """
        if progress:
            progress(PhaseReport(phase="summarizer", message="aggregating the kept Findings into one grounded Provision relevance statement per cited provision"))
        prompt = _labelled_provisions(state.kept_findings)
        if not prompt.targets_by_ref:
            # No cited provision carries a Finding: nothing to summarize, and
            # no LLM call is made over nothing.
            return {}
        started = time.perf_counter()
        try:
            summaries = llm.complete(
                system=_SUMMARIZER_SYSTEM, user=prompt.block, schema=Summaries
            )
        except Exception as error:
            # Degrade, never fail (issue #47): any Summarizer failure — an
            # unreachable provider, a malformed reply, a bug — leaves the
            # Answer served without relevance, named by a Known limitation.
            logger.warning(
                "summarizer failed (%s); the Answer ships without Provision relevance", error
            )
            return {"summarizer_failed": True}
        grounded, decisions = _validate_summaries(summaries, prompt.targets_by_ref)
        if not grounded:
            # Nothing usable came back — every statement was rejected, or none
            # was made: the same degradation as an outright failure, never a
            # silent absence.
            return {"summarizer_failed": True, "summary_decisions": decisions}
        logger.info(
            "summarizer: %d relevance statement(s) over %d provision(s) in %.2fs",
            len(grounded),
            len(prompt.targets_by_ref),
            time.perf_counter() - started,
        )
        return {
            "relevance": list(grounded.values()),
            "summary_decisions": decisions,
        }

    builder = StateGraph(LiveState)
    builder.add_node("planner", planner)
    builder.add_node("researcher", researcher)
    builder.add_node("verifier", verifier)
    builder.add_node("proposer", proposer)
    builder.add_node("summarizer", summarizer)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "verifier")
    builder.add_edge("verifier", "proposer")
    builder.add_edge("proposer", "summarizer")
    builder.add_edge("summarizer", END)
    return builder.compile()


def run_live_analysis(
    request: AnalyzeRequest,
    *,
    llm: Llm,
    retriever: Retriever,
    observation: RequestObservation | None = None,
    progress: ProgressSink | None = None,
) -> AnalyzeResponse:
    """Answer an arbitrary Scenario through the Live workflow.

    The composition root injects the provider protocols; this function
    stays the one seam the API layer calls. ``observation`` receives the
    per-request observability counts — the size of the Evidence pool this
    pass actually reasoned over. ``progress``, when given, receives each
    phase transition the moment its workflow agent starts: the five
    agent names in order (Planner → Researcher → Verifier → Proposer →
    Summarizer) with an informative message each. The composition root
    builds it with
    ``progress.progress_sink(request_id)`` — bound to the shared registry
    by default — and None leaves the run unreported.
    """
    scenario_description = request.scenario.description or request.scenario.title or ""
    result = _build_graph(llm, retriever, observation, progress).invoke(
        LiveState(scenario_description=scenario_description, question=request.question)
    )
    state = result if isinstance(result, LiveState) else LiveState.model_validate(result)

    # The Verifier's output was decided once, by the Proposer: kept Findings,
    # per-claim decisions, and the citation labels the grounding gate reads
    # all travel through the state from that single computation.
    findings = sorted(state.kept_findings, key=lambda f: STRENGTH_ORDER.get(f.strength, 1))
    decisions = state.claim_decisions
    discarded = [decision.claim for decision in decisions if decision.status == "rejected"]
    # The Summarizer's gate-validated output, keyed by provision target for
    # the Answer's one-entry-per-provision citation list (ADR-0011): each
    # GroundedSummary carries the relevance and the rated strength together.
    summaries_by_target = {grounded.target: grounded for grounded in state.relevance}
    all_citations = answer_citations(findings, summaries_by_target)
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
    # The Summarizer's pass is classified once; the Known limitation, the
    # trace sentence, and the step action all read that classification.
    summarizer_outcome = _classify_summarizer(state, total=len(all_citations))
    summarizer_limitation = _summarizer_limitation(summarizer_outcome)
    if summarizer_limitation is not None:
        known_limitations.append(summarizer_limitation)

    proposal_decisions: list[ActionDecision] = []
    if insufficient:
        # The Insufficient-evidence path is unchanged: narrowing suggestions,
        # and the Proposer made no LLM call over nothing.
        actions = list(NARROW_THE_QUESTION_ACTIONS)
    elif state.actions:
        # The Proposer itself emitted the standing hand-off alone over no
        # kept Findings — the sheet is served verbatim, never empty.
        actions = list(state.actions)
    else:
        actions, proposal_decisions = _validate_proposals(
            state.proposals, state.grounding_by_label
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
        kept_proposals = sum(1 for d in proposal_decisions if d.status == "kept")
        rejected_proposals = sum(1 for d in proposal_decisions if d.status == "rejected")
        summary = (
            f"Planner identified research targets ({plan_summary}); Researcher retrieved "
            f"{len(retrieved_chunks)} Chunk(s) via the retrieval tool and drafted "
            f"{len(state.drafted.claims)} claim(s); Verifier kept {len(findings)} Finding(s) "
            f"and recorded {len(discarded)} Unsupported claim(s) as rejected; Proposer "
            f"distilled {kept_proposals} referral Action(s) from the kept Findings "
            f"and recorded {rejected_proposals} ungrounded proposal(s) as rejected; "
            f"{_summarizer_sentence(summarizer_outcome)}."
        )

    if insufficient:
        # The trace stays honest about the skipped pass: nothing was distilled.
        proposer_step_action = (
            "no Evidence was retrieved: the Insufficient-evidence path was served "
            "and the Proposer made no LLM call"
        )
    elif state.actions:
        proposer_step_action = (
            "no kept Finding remained to anchor an Action: the standing "
            "seek-counsel hand-off was served alone, and no LLM call was made"
        )
    else:
        proposer_step_action = (
            "distill the kept Findings into referral Actions, each grounded "
            "in a kept Finding's Citations"
        )

    summarizer_step_action = _summarizer_step_action(summarizer_outcome)

    planner_step: dict[str, Any] = {
        "step": "planner",
        "action": "decompose the Regulatory question into research targets",
        "research_targets": queries,
    }
    if state.plan_correction:
        planner_step["correction"] = state.plan_correction

    detailed_trace: list[dict[str, Any]] = [
        planner_step,
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
            "action": proposer_step_action,
            "action_decisions": [decision.model_dump() for decision in proposal_decisions],
        },
        {
            "step": "summarizer",
            "action": summarizer_step_action,
            "summary_decisions": [decision.model_dump() for decision in state.summary_decisions],
        },
    ]

    return AnalyzeResponse(
        answer=Answer(findings=findings, actions=actions, citations=all_citations),
        trace=Trace(workflow=LIVE_WORKFLOW_MARKER, summary=summary, unsupported_claims_discarded=discarded),
        detailed_trace=detailed_trace,
        known_limitations=known_limitations,
    )
