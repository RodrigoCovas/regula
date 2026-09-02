from enum import Enum
from typing import Dict, Iterable, List, Literal, NamedTuple, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Scenario(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None


class Mode(str, Enum):
    demo = "demo"
    live = "live"


class AnalyzeRequest(BaseModel):
    """One analysis run (ADR-0008): mode is a per-run choice carried on the
    request itself. A request that omits ``mode`` gets the server default
    (REGULA_MODE, itself defaulting to Demo)."""

    scenario: Scenario
    question: str
    mode: Optional[Mode] = None


class Strength(str, Enum):
    strong = "strong"
    moderate = "moderate"
    weak = "weak"


class ProvisionKind(str, Enum):
    article = "article"
    recital = "recital"
    annex = "annex"


# The one map from provision kind to the exactly-one-target Citation field it
# populates — shared by every validator, citation derivation, and lookup that
# must agree on the mapping.
PROVISION_NUMBER_FIELDS: dict[ProvisionKind, str] = {
    ProvisionKind.article: "article_number",
    ProvisionKind.recital: "recital_number",
    ProvisionKind.annex: "annex_number",
}


# The one map from provision kind to its human-readable provision noun —
# shared by every chunk heading, citation derivation, inventory render, and
# label that must agree on how a provision is named.
PROVISION_NOUNS: dict[ProvisionKind, str] = {
    ProvisionKind.article: "Article",
    ProvisionKind.recital: "Recital",
    ProvisionKind.annex: "Annex",
}


class ProvisionTarget(NamedTuple):
    """The one concrete provision a Citation points at: its source document,
    the provision kind, and the document's structural number — what ground
    truth and production are compared on."""

    source_id: str
    kind: ProvisionKind
    number: int


class Citation(BaseModel):
    source_id: str
    source_short_name: Optional[str] = None
    article_number: Optional[int] = None
    recital_number: Optional[int] = None
    annex_number: Optional[int] = None
    section: Optional[str] = None
    provision: Optional[str] = None
    quote: Optional[str] = None
    # Answer-wide fields (issue #47): the Citations section renders them from
    # the Answer's list only — a per-Finding Citation carries neither.
    # ``strength`` is the provision's rated Citation strength (ADR-0011),
    # never derived from the citing Findings.
    relevance: Optional[str] = None
    strength: Optional[Strength] = None

    @model_validator(mode="after")
    def exactly_one_target(self):
        targets = [getattr(self, field) is not None for field in PROVISION_NUMBER_FIELDS.values()]
        if sum(targets) != 1:
            raise ValueError(
                "A Citation must target exactly one of article_number, recital_number, or annex_number"
            )
        return self

    @property
    def provision_target(self) -> ProvisionTarget:
        """This Citation's structural target, from its validated exactly-one
        metadata — never its free-text provision label."""
        for kind, field_name in PROVISION_NUMBER_FIELDS.items():
            number = getattr(self, field_name)
            if number is not None:
                return ProvisionTarget(self.source_id, kind, number)
        raise ValueError("Citation carries no provision number")  # unreachable: validator-enforced


class Finding(BaseModel):
    statement: str
    strength: Strength = Strength.moderate
    citations: List[Citation] = Field(default_factory=list)


# How early a Strength sorts: strong outranks moderate outranks weak. The
# map the max-rule reads — the strongest Strength wins, on the eval's
# expected side — and the one the answer's strength-first Finding order
# sorts by.
STRENGTH_ORDER: Dict[Strength, int] = {
    Strength.strong: 0,
    Strength.moderate: 1,
    Strength.weak: 2,
}


def max_rule_strengths(
    pairs: Iterable[Tuple[ProvisionTarget, Strength]]
) -> Dict[ProvisionTarget, Strength]:
    """Citation strength (CONTEXT.md) under the max-rule: per provision
    target, the strongest Strength among the rated Citations citing it. Since
    ADR-0011 the max-rule serves the eval's expected side only — coverage
    weights and the expected half of strength agreement; the produced side
    carries the Summarizer's ratings as given."""
    strengths: Dict[ProvisionTarget, Strength] = {}
    for target, strength in pairs:
        current = strengths.get(target)
        if current is None or STRENGTH_ORDER[strength] < STRENGTH_ORDER[current]:
            strengths[target] = strength
    return strengths


class GroundedSummary(BaseModel):
    """One cited provision's Provision relevance, grounded in the Findings
    citing it, with the provision's rated Citation strength — the
    gate-validated Summarizer output the Answer carries (issue #47,
    ADR-0011). ``strength`` is carried as rated and stays None when the
    rating was missing or invalid — never defaulted from the citing
    Findings."""

    target: ProvisionTarget
    relevance: str
    strength: Optional[Strength] = None


def answer_citations(
    findings: List[Finding],
    summaries: Optional[Dict[ProvisionTarget, GroundedSummary]] = None,
) -> List[Citation]:
    """The Answer's flat Citation list (issue #47): one entry per cited
    provision target, in first-mention order, badged with the rated Citation
    strength the Summarizer carried (ADR-0011) and — when produced — the
    provision's grounded Provision relevance. Provision relevance and
    Citation strength are answer-wide, so they attach here and never to the
    per-Finding Citations the Findings section renders. Only provisions the
    summaries carry get their answer-wide fields: a provision whose rating is
    missing or invalid keeps its relevance but carries no strength — nothing
    is derived from the citing Findings' Strengths."""
    known = summaries or {}
    entries: Dict[ProvisionTarget, Citation] = {}
    for finding in findings:
        for citation in finding.citations:
            target = citation.provision_target
            if target not in entries:
                summary = known.get(target)
                entries[target] = citation.model_copy(update={
                    "strength": summary.strength if summary else None,
                    "relevance": summary.relevance if summary else None,
                })
    return list(entries.values())


# The default embedding model (nomic-embed-text, served locally by Ollama)
# outputs this many dimensions; the pgvector column and the embedder agree on
# it, and any EMBEDDING_MODEL override must produce the same width.
EMBEDDING_DIMENSION = 768


# The identity of a stored Chunk: source, provision kind, the exactly-one
# provision number, and the chunk index. Two Chunks with one identity are
# the same stored Chunk, whatever read path surfaced them.
ChunkIdentity = tuple[str, str, Optional[int], int]


class Chunk(BaseModel):
    """A retrievable passage of one Regulation, targeting a single provision.

    The provision metadata is what Citations are later derived from, so it is
    validated with the same exactly-one-target invariant as Citation.
    """

    source_id: str
    kind: ProvisionKind
    text: str
    title: Optional[str] = None
    chunk_index: int = 0
    num_chunks: int = 1
    article_number: Optional[int] = None
    recital_number: Optional[int] = None
    annex_number: Optional[int] = None

    @property
    def provision_number(self) -> Optional[int]:
        """The exactly-one provision number, whichever kind this Chunk targets."""
        return self.article_number or self.recital_number or self.annex_number

    @property
    def identity(self) -> ChunkIdentity:
        """The stored Chunk's identity (``ChunkIdentity``): two Chunks with
        one identity are the same stored Chunk, whatever read path surfaced
        them."""
        return (self.source_id, self.kind.value, self.provision_number, self.chunk_index)

    @model_validator(mode="after")
    def exactly_one_target(self):
        targets = {kind: getattr(self, field) for kind, field in PROVISION_NUMBER_FIELDS.items()}
        if sum(value is not None for value in targets.values()) != 1:
            raise ValueError(
                "A Chunk must target exactly one of article_number, recital_number, or annex_number"
            )
        if targets[self.kind] is None:
            raise ValueError("A Chunk's kind must agree with the populated provision number")
        return self


class ScoredChunk(BaseModel):
    """One search result: a stored Chunk plus its cosine distance to the query.

    Wrapping (rather than flattening) the Chunk keeps one owner for its
    provision metadata and enforces the exactly-one-target invariant at
    construction — an unscoreable or malformed row cannot become Evidence.
    """

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    distance: float


class Answer(BaseModel):
    findings: List[Finding]
    actions: List[str]
    citations: List[Citation]


class Trace(BaseModel):
    workflow: str
    summary: str
    unsupported_claims_discarded: List[str] = Field(default_factory=list)


class ClaimDecision(BaseModel):
    """One Verifier decision in the detailed trace: kept, or rejected with why."""

    claim: str
    status: Literal["kept", "rejected"]
    reason: Optional[str] = None


class AnalyzeResponse(BaseModel):
    answer: Answer
    trace: Trace
    detailed_trace: Optional[List[dict]] = Field(default_factory=list)
    known_limitations: List[str] = Field(default_factory=list)


class Readiness(BaseModel):
    """Live-mode Readiness (CONTEXT.md) as three independent booleans.

    GET /readiness reports them so tooling and the frontend can check the
    prerequisites — a configured provider key, the embedding model available,
    a non-empty ingested Corpus — without probing internals. Each is probed
    separately, so a checklist can name exactly which one is missing.
    """

    api_key_set: bool
    embedding_model_present: bool
    corpus_ingested: bool


def quote_snippet(text: str, limit: int = 300) -> str:
    """The citation-quote form of a provision text: first ``limit`` chars, elided."""
    return text[:limit] + ("..." if len(text) > limit else "")
