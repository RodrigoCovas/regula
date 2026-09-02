"""Deterministic fakes for the Live pipeline seams (spec #9, tickets #17–#18).

Injected at the composition root via FastAPI dependency overrides so the
HTTP seam runs fully offline — no network, no Ollama, no pgvector. The
embedder, search-store, and scored-hit fakes also drive a real
``HybridRetriever`` over canned search results (#18, #64).
"""

from src.eval_judge import JudgeVerdict, PairVerdict, RelevancePair, pair_fidelity
from src.llm import Llm
from src.live_workflow import (    ActionProposal,
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
from src.models import Chunk, ProvisionKind, ScoredChunk, Strength
from src.retrieval import Retriever


class FakeClock:
    """A deterministic monotonic clock for time-dependent modules under test.

    Start at an arbitrary value and advance it by hand: anything that reads
    ``time.monotonic``-style time through an injected ``now`` callable sees
    exactly what the test dictates.
    """

    def __init__(self, start: float = 0.0) -> None:
        self.value = start

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeEmbedder:
    """Deterministic vectors keyed by text length; records every batch."""

    def __init__(self):
        self.batches: list[list[str]] = []

    def embed(self, texts):
        self.batches.append(list(texts))
        return [[float(len(text)), 1.0] for text in texts]


class FakeSearchStore:
    """Records search calls; replays canned hits regardless of any bound.

    That blind replay is what makes it usable as an adversarial store too:
    handed junk-only hits, it returns them whatever max_distance the caller
    pushes down — exactly the situation the seam-side relevance floor (#18)
    must survive. The lexical read path (spec #64) replays the same way:
    the hybrid retriever's per-leg caps are verified by assertion, never by
    the fake enforcing them. With no ``lexical_hits`` given, the lexical leg
    matches nothing — tests opt in to lexical Evidence.
    """

    def __init__(self, hits, lexical_hits=None):
        self.hits = hits
        self.lexical_hits = list(lexical_hits or [])
        self.calls: list[dict] = []
        self.lexical_calls: list[dict] = []

    def search_chunks(self, query_embedding, limit, max_distance):
        self.calls.append(
            {
                "query_embedding": query_embedding,
                "limit": limit,
                "max_distance": max_distance,
            }
        )
        return self.hits

    def search_chunks_lexically(self, query, limit):
        self.lexical_calls.append({"query": query, "limit": limit})
        return self.lexical_hits


def chunk_hit(
    source_id="ai-act",
    kind="article",
    number=6,
    text="Article 6 body",
    title=None,
    index=0,
    num_chunks=1,
    distance=0.1,
) -> ScoredChunk:
    """One store hit: a valid single-provision Chunk plus its cosine distance."""
    return ScoredChunk(
        chunk=Chunk(
            source_id=source_id,
            kind=ProvisionKind(kind),
            title=title,
            chunk_index=index,
            num_chunks=num_chunks,
            article_number=number if kind == "article" else None,
            recital_number=number if kind == "recital" else None,
            annex_number=number if kind == "annex" else None,
            text=text,
        ),
        distance=distance,
    )


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
    """Replays canned structured outputs keyed by boundary schema.

    ``usage`` mirrors the real client's exposure: when given, every
    completed call leaves one provider usage dictionary behind, so
    per-request aggregation has something to sum.
    """

    def __init__(
        self,
        plan: Plan | None = None,
        claims: DraftClaims | None = None,
        verdicts: Verdicts | None = None,
        proposals: ActionProposals | None = None,
        summaries: Summaries | None = None,
        usage: dict | None = None,
    ):
        self.plan = plan or Plan(targets=[ResearchTarget(query="creditworthiness evaluation")])
        self.claims = claims or DraftClaims(claims=[])
        self.verdicts = verdicts or Verdicts(verdicts=[])
        self.proposals = proposals or ActionProposals(proposals=[])
        self.summaries = summaries or Summaries(summaries=[])
        self._canned_usage = usage
        self.calls: list[tuple[str, str, type]] = []
        self.usage: list[dict] = []

    def complete(self, system: str, user: str, schema: type):
        self.calls.append((system, user, schema))
        if self._canned_usage is not None:
            self.usage.append(dict(self._canned_usage))
        canned = {
            Plan: self.plan,
            DraftClaims: self.claims,
            Verdicts: self.verdicts,
            ActionProposals: self.proposals,
            Summaries: self.summaries,
        }[schema]  # type: ignore[index]
        return canned.model_copy(deep=True)


class ScriptedJudgeLlm:
    """The judge's provider seam, kept apart from the pipeline fake: the
    fidelity judge is a second LLM (ADR-0010) with its own schema, so it gets
    its own scripted completion — one canned ``JudgeVerdict`` reply, calls
    recorded for the batched-call and rubric-prompt assertions."""

    def __init__(self, reply: JudgeVerdict):
        self.reply = reply
        self.calls: list[tuple[str, str, type]] = []

    def complete(self, system: str, user: str, schema: type):
        self.calls.append((system, user, schema))
        return self.reply.model_copy(deep=True)


def grounded_verdict(statement: str, strength: Strength, refs: list[str]) -> Verdict:
    return Verdict(statement=statement, supported=True, strength=strength, evidence_refs=refs)


class ScriptedJudge:
    """Scripts the summary-fidelity judge seam (parent spec, seam 2): canned
    pair verdicts scored through the real rubric derivation, every compared
    pair list recorded for the batched-call assertions."""

    def __init__(self, verdicts: dict[str, PairVerdict]):
        self._verdicts = verdicts
        self.calls: list[list[RelevancePair]] = []

    def compare(self, pairs: list[RelevancePair]) -> dict[str, float]:
        self.calls.append(list(pairs))
        return {pair.ref: pair_fidelity(self._verdicts[pair.ref]) for pair in pairs}


# The four rubric edges (issue #50) as canned P1 verdicts, shared by every
# suite that scripts the judge. A pair with another reference needs its own
# verdict — the ref inside travels with the verdict, so these serve P1 only.
FULL_AGREEMENT_VERDICT = PairVerdict(ref="P1", same_role=True, same_direction=True, contradiction=False)
ROLE_MISMATCH_VERDICT = PairVerdict(ref="P1", same_role=False, same_direction=True, contradiction=False)
POLARITY_FLIP_VERDICT = PairVerdict(ref="P1", same_role=True, same_direction=False, contradiction=False)
CONTRADICTION_VERDICT = PairVerdict(ref="P1", same_role=True, same_direction=True, contradiction=True)


def make_offline_llm(usage: dict | None = None) -> ScriptedLlm:
    """One strong grounded claim, one weak framing claim, one unsupported claim.

    The canned proposal is grounded on C1 — the first kept Finding's first
    Citation (the strong high-risk claim in the default scenario) — so the
    happy path serves one validated referral Action plus the hand-off.
    """
    return ScriptedLlm(
        usage=usage,
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
        proposals=ActionProposals(
            proposals=[
                ActionProposal(
                    action="Have a qualified professional verify that the company's credit evaluation duties match the high-risk provisions cited.",
                    kind="verify_against_facts",
                    citation_refs=["C1"],
                ),
            ]
        ),
        # The kept Findings cite two provisions (P1: the strong high-risk
        # claim's Citation, P2: the weak definitions one); each relevance
        # statement draws only on its citing Findings' content, and each
        # provision carries its rated Citation strength (ADR-0011).
        summaries=Summaries(
            summaries=[
                ProvisionSummary(
                    ref="P1",
                    relevance=(
                        "Annex III point 5(b) of the AI Act names creditworthiness evaluation "
                        "of natural persons as a high-risk use, which is what puts the "
                        "company's loan-scoring system under the full high-risk obligations."
                    ),
                    strength=Strength.strong,
                ),
                ProvisionSummary(
                    ref="P2",
                    relevance="The definitions Article supplies the vocabulary the other Findings rely on — AI system, provider, deployer, profiling — without establishing an obligation on its own.",
                    strength=Strength.weak,
                ),
            ]
        ),
    )


assert isinstance(FakeRetriever([]), Retriever)
assert isinstance(ScriptedLlm(), Llm)
assert isinstance(ScriptedJudgeLlm(JudgeVerdict(verdicts=[])), Llm)
