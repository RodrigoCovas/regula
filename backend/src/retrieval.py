"""Retrieval service — the Evidence-gathering primitive (spec #9, ticket #15).

Query text in, relevant Chunks out. ``HybridRetriever`` composes its provider
protocols — it embeds the query locally via an ``Embedder``, searches the
pgvector store through the vector read path, and reads the store's lexical
full-text path — fusing the two legs by Reciprocal Rank Fusion so the
``Retriever`` seam stays query-in, Chunks-out (spec #64, ADR-0012). Tests
and the Live workflow both depend on that single seam at the composition root.

Locked behaviour:

- Two legs per retrieval, each a bounded pass with its own cap pushed down
  into the store's LIMIT clause: the vector leg takes top-``VECTOR_LEG_DEPTH``
  (12), the lexical leg top-``LEXICAL_LEG_DEPTH`` (8). One retrieval can
  return up to the two caps' sum in unique Chunks. The Evidence pool served
  to the agents is a separate concern derived from the plan (live_workflow's
  ``SEATS_PER_TARGET``) — one constant playing both roles is how #23
  happened.
- The relevance floor binds the vector leg only. A vector hit must clear the
  cosine similarity floor, enforced twice — pushed down into the store's
  search and re-checked here — so the guarantee holds at this seam regardless
  of how a Store implementation behaves. Lexical matches carry no similarity
  score, so the floor never gate-keeps them: lexical-only evidence may fill
  the pool, and the junk-only guarantee the floor once provided alone is
  deliberately weakened — the tightened Researcher and Verifier are the
  second line of defense (ADR-0012).
- Both legs exclude Recital Chunks (spec #68, ticket #69): Recitals are
  framing, not operative provisions, so the exclusion rides both legs'
  pushed-down parameters into the store's SQL and is re-checked here like
  the relevance floor — a Recital Chunk never consumes vector leg depth,
  lexical leg depth, or Evidence pool seats. Recitals stay ingested; Demo
  mode serves fixed content and retrieves nothing, so the canonical ground
  truth that cites a GDPR Recital is grandfathered.
- The ranked legs fuse by Reciprocal Rank Fusion (k = ``RRF_K``): each Chunk
  scores 1/(k + rank) per leg that found it, summed — rank-based, so cosine
  distance and ts_rank never need calibrating against each other. A Chunk
  both legs found appears once, ranked by its summed contribution.
- Every returned Chunk is metadata-complete: it came wrapped in a
  ``ScoredChunk`` (vector leg) or was validated at store construction
  (lexical leg), ready for deterministic Citation derivation.

The threshold default was tuned against the real ingested Corpus with the
locked embedding model; cosine similarity of genuinely relevant provisions
sits far above junk matches.
"""

from typing import Protocol, runtime_checkable, Sequence

from .embedder import Embedder
from .models import Chunk, ChunkIdentity, ProvisionKind, ScoredChunk

# The vector leg's per-target depth (ADR-0012): how deep the vector search
# reaches — the store's LIMIT for the vector query, whatever the plan looks
# like. Raised from 8 alongside the widened Evidence pool: the pool's seats
# per target must stay reachable. The total pool served to the agents is
# derived separately from the plan (live_workflow.SEATS_PER_TARGET): raising
# that pool must not silently deepen crawling.
VECTOR_LEG_DEPTH = 12

# The lexical leg's per-target cap (ADR-0012): the store's LIMIT for the
# full-text query. Lexically distinctive provisions surface here even when
# the embedder cannot see them.
LEXICAL_LEG_DEPTH = 8

# Reciprocal Rank Fusion constant (ADR-0012): the standard k = 60 dampens
# top ranks so the two legs' orders fuse by rank alone — no calibration
# between cosine distance and ts_rank.
RRF_K = 60

# Cosine similarity floor for a Chunk to count as relevant, tuned against
# the ingested Corpus with the locked embedding model: junk-only questions
# top out around 0.54 while targeted relevant provisions score from ~0.55 up
# (strong hits land at 0.69+). pgvector speaks distance
# (similarity = 1 - distance), so this converts once at the boundary.
DEFAULT_MIN_SIMILARITY = 0.55

# The provision kinds Live retrieval never returns (spec #68, ticket #69):
# Recitals are framing, not operative provisions, so they are excluded on
# both legs — pushed down into the store's SQL and re-checked at this seam —
# and the legs' depth and the Evidence pool's seats go to operative
# provisions. Recitals stay ingested; only the read path filters.
EXCLUDED_KINDS: tuple[ProvisionKind, ...] = (ProvisionKind.recital,)


@runtime_checkable
class Retriever(Protocol):
    """The Evidence source of the Live workflow."""

    def retrieve(self, query: str) -> list[Chunk]: ...


@runtime_checkable
class SearchStore(Protocol):
    """The read path of the pgvector store."""

    def search_chunks(
        self,
        query_embedding: Sequence[float],
        limit: int,
        max_distance: float,
        excluded_kinds: Sequence[ProvisionKind],
    ) -> Sequence[ScoredChunk]: ...


@runtime_checkable
class LexicalSearchStore(Protocol):
    """The lexical read path of the store (spec #60, ADR-0012).

    A sibling of ``SearchStore``, deliberately a separate seam: the vector
    doubles and callers stay valid, and the hybrid retriever (#64) composes
    both legs from one store without either protocol knowing the other.
    """

    def search_chunks_lexically(
        self, query: str, limit: int, excluded_kinds: Sequence[ProvisionKind]
    ) -> Sequence[Chunk]: ...


@runtime_checkable
class HybridSearchStore(SearchStore, LexicalSearchStore, Protocol):
    """Both read paths the hybrid retriever composes: one store, two legs."""


class VectorRetriever:
    """The vector leg: embeds the query, vector-searches the store, returns
    the Chunks that clear the relevance floor, best similarity first."""

    def __init__(
        self,
        store: SearchStore,
        embedder: Embedder,
        depth: int = VECTOR_LEG_DEPTH,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
    ):
        self._store = store
        self._embedder = embedder
        self._depth = depth
        self._min_similarity = min_similarity
        # The relevance floor as cosine distance, derived once: the same bound
        # is pushed down into the store's search and enforced again at this
        # seam, so both enforcements always agree.
        self._max_distance = 1 - self._min_similarity

    def retrieve(self, query: str) -> list[Chunk]:
        [query_embedding] = self._embedder.embed([query])
        hits = self._store.search_chunks(
            query_embedding=query_embedding,
            limit=self._depth,
            max_distance=self._max_distance,
            excluded_kinds=EXCLUDED_KINDS,
        )
        # The floor and the Recital exclusion are enforced here, not only in
        # the store's SQL: whatever a Store implementation returns, no Chunk
        # beyond the relevance floor — and no excluded provision kind — can
        # cross this seam and become Evidence.
        return [
            hit.chunk
            for hit in hits
            if hit.distance <= self._max_distance and hit.chunk.kind not in EXCLUDED_KINDS
        ]


def _rrf_fuse(legs: list[list[Chunk]], k: int) -> list[Chunk]:
    """Fuse ranked Chunk lists by Reciprocal Rank Fusion.

    A Chunk scores 1/(k + rank) per leg that found it — rank starting at 1 —
    summed across legs; a Chunk both legs found appears once, ranked by its
    summed contribution. Score ties keep first-appearance order (the vector
    leg's results precede the lexical leg's), so the fusion is deterministic.
    """
    scores: dict[ChunkIdentity, float] = {}
    first_seen: dict[ChunkIdentity, Chunk] = {}
    for leg in legs:
        for rank, chunk in enumerate(leg, start=1):
            key = chunk.identity
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(key, chunk)
    # A stable descending sort: equal scores keep dict insertion order — the
    # first-appearance order the ties must resolve by.
    return [
        first_seen[key]
        for key in sorted(scores, key=lambda key: scores[key], reverse=True)
    ]


class HybridRetriever:
    """Fuses the vector and lexical legs of one Research target's retrieval.

    The vector leg (a ``VectorRetriever`` over the same store) keeps its
    similarity floor and takes top-``VECTOR_LEG_DEPTH``; the lexical leg
    takes top-``LEXICAL_LEG_DEPTH`` through the store's full-text read. The
    ranked lists fuse by Reciprocal Rank Fusion — rank-based, so the legs'
    incomparable scores never meet, only their orders do (ADR-0012).
    """

    def __init__(
        self,
        store: HybridSearchStore,
        embedder: Embedder,
        vector_depth: int = VECTOR_LEG_DEPTH,
        lexical_depth: int = LEXICAL_LEG_DEPTH,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
    ):
        self._store = store
        self._vector_leg = VectorRetriever(
            store=store,
            embedder=embedder,
            depth=vector_depth,
            min_similarity=min_similarity,
        )
        self._lexical_depth = lexical_depth

    def retrieve(self, query: str) -> list[Chunk]:
        vector_results = self._vector_leg.retrieve(query)
        lexical_hits = self._store.search_chunks_lexically(
            query, self._lexical_depth, excluded_kinds=EXCLUDED_KINDS
        )
        # The exclusion is re-checked on the lexical leg too: a store that
        # ignores the pushed-down kinds cannot seat a Recital through fusion.
        lexical_results = [
            chunk for chunk in lexical_hits if chunk.kind not in EXCLUDED_KINDS
        ]
        return _rrf_fuse([vector_results, lexical_results], k=RRF_K)
