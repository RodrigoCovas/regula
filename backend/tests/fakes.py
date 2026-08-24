"""Deterministic fakes for the Live pipeline seams (spec #9, ticket #17).

Injected at the composition root via FastAPI dependency overrides so the
HTTP seam runs fully offline — no network, no Ollama, no pgvector.
"""

from src.llm import Llm
from src.live_workflow import DraftClaim, DraftClaims, Plan, ResearchTarget, Verdict, Verdicts
from src.models import Chunk, ProvisionKind, Strength
from src.retrieval import Retriever


def make_chunk(
    source_id: str = "ai-act",
    kind: ProvisionKind = ProvisionKind.article,
    number: int = 6,
    text: str = "",
    title: str | None = None,
) -> Chunk:
    """A valid single-provision Chunk with the exactly-one-target metadata."""
    return Chunk(
        source_id=source_id,
        kind=kind,
        text=text or f"Provision text of {source_id} {kind.value} {number}.",
        title=title,
        article_number=number if kind == ProvisionKind.article else None,
        recital_number=number if kind == ProvisionKind.recital else None,
        annex_number=number if kind == ProvisionKind.annex else None,
    )


HIGH_RISK_CHUNK = make_chunk(text="Creditworthiness evaluation AI systems are high-risk.", title="Classification rules")
AUTOMATED_DECISION_CHUNK = make_chunk(source_id="gdpr", number=22, text="Automated individual decision-making.", title="Automated decision-making")
DEFINITIONS_CHUNK = make_chunk(number=3, text="Definitions of AI system, provider and deployer.", title="Definitions")


class FakeRetriever:
    """Returns the same canned Chunks for every query; records the queries.

    ``per_query`` overrides the default result set for specific queries, so a
    test can give each research target its own retrieval results.
    """

    def __init__(self, chunks: list[Chunk] | None = None, per_query: dict[str, list[Chunk]] | None = None):
        self.chunks = chunks if chunks is not None else [
            HIGH_RISK_CHUNK,
            AUTOMATED_DECISION_CHUNK,
            DEFINITIONS_CHUNK,
        ]
        self.per_query = per_query or {}
        self.queries: list[str] = []

    def retrieve(self, query: str) -> list[Chunk]:
        self.queries.append(query)
        if query in self.per_query:
            return list(self.per_query[query])
        return list(self.chunks)


class ScriptedLlm:
    """Replays canned structured outputs keyed by boundary schema."""

    def __init__(
        self,
        plan: Plan | None = None,
        claims: DraftClaims | None = None,
        verdicts: Verdicts | None = None,
    ):
        self.plan = plan or Plan(targets=[ResearchTarget(query="creditworthiness evaluation")])
        self.claims = claims or DraftClaims(claims=[])
        self.verdicts = verdicts or Verdicts(verdicts=[])
        self.calls: list[tuple[str, str, type]] = []

    def complete(self, system: str, user: str, schema: type):
        self.calls.append((system, user, schema))
        canned = {Plan: self.plan, DraftClaims: self.claims, Verdicts: self.verdicts}[schema]  # type: ignore[index]
        return canned.model_copy(deep=True)


def grounded_verdict(statement: str, strength: Strength, refs: list[str]) -> Verdict:
    return Verdict(statement=statement, supported=True, strength=strength, evidence_refs=refs)


def make_offline_llm() -> ScriptedLlm:
    """One strong grounded claim, one weak framing claim, one unsupported claim."""
    return ScriptedLlm(
        plan=Plan(targets=[
            ResearchTarget(query="creditworthiness"),
            ResearchTarget(query="automated decisions"),
        ]),
        claims=DraftClaims(
            claims=[
                DraftClaim(
                    statement="Creditworthiness evaluation is a high-risk use case.",
                    evidence_refs=["E1"],
                ),
                DraftClaim(
                    statement="Loan scoring data counts as special-category data.",
                    evidence_refs=[],
                ),
                DraftClaim(
                    statement="The system qualifies as an AI system under the definitions.",
                    evidence_refs=["E3"],
                ),
            ]
        ),
        verdicts=Verdicts(
            verdicts=[
                Verdict(statement="Creditworthiness evaluation is a high-risk use case.", supported=True, strength=Strength.strong, evidence_refs=["E1"]),
                Verdict(statement="Loan scoring data counts as special-category data.", supported=False),
                Verdict(statement="The system qualifies as an AI system under the definitions.", supported=True, strength=Strength.weak, evidence_refs=["E3"]),
            ]
        ),
    )


assert isinstance(FakeRetriever([]), Retriever)
assert isinstance(ScriptedLlm(), Llm)
