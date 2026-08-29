from enum import Enum
from typing import List, Literal, NamedTuple, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Scenario(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None


class AnalyzeRequest(BaseModel):
    scenario: Scenario
    question: str


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


# The default embedding model (nomic-embed-text, served locally by Ollama)
# outputs this many dimensions; the pgvector column and the embedder agree on
# it, and any EMBEDDING_MODEL override must produce the same width.
EMBEDDING_DIMENSION = 768


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


def quote_snippet(text: str, limit: int = 300) -> str:
    """The citation-quote form of a provision text: first ``limit`` chars, elided."""
    return text[:limit] + ("..." if len(text) > limit else "")
