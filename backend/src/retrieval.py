"""Retrieval service — the Evidence-gathering primitive (spec #9, ticket #15).

Query text in, top-k most relevant Chunks out. ``VectorRetriever`` composes
its provider protocols — it embeds the query locally via an ``Embedder``
and searches the pgvector store through a search-capable ``Store`` — so tests
and the Live workflow both depend on the single ``Retriever`` seam at the
composition root.

Locked behaviour:

- The result set is capped at the single-pass budget (~8): the budget is
  requested from the store, whose LIMIT clause enforces it; retrieval is one
  bounded pass, never a crawl.
- A relevance threshold excludes junk-only result sets: when nothing in the
  Corpus matches well enough, the caller gets no Chunks rather than
  irrelevant Evidence. That empty result is what the Insufficient-evidence
  path builds on. The floor is enforced twice — pushed down into the
  store's search and re-checked here — so the guarantee holds at this seam
  regardless of how a Store implementation behaves.
- Every returned Chunk arrives wrapped in a ``ScoredChunk``, so its
  provision metadata (source_id plus exactly one provision number kind) was
  validated at construction, ready for deterministic Citation derivation.

The threshold default was tuned against the real ingested Corpus with the
locked embedding model; cosine similarity of genuinely relevant provisions
sits far above junk matches.
"""

from typing import Protocol, runtime_checkable, Sequence

from .embedder import Embedder
from .models import Chunk, ScoredChunk

# Locked single-pass budget (~8): the Researcher never sees more Evidence
# than this from one retrieval pass.
MAX_RETRIEVED_CHUNKS = 8

# Cosine similarity floor for a Chunk to count as relevant, tuned against
# the ingested Corpus with the locked embedding model: junk-only questions
# top out around 0.54 while targeted relevant provisions score from ~0.55 up
# (strong hits land at 0.69+). pgvector speaks distance
# (similarity = 1 - distance), so this converts once at the boundary.
DEFAULT_MIN_SIMILARITY = 0.55


@runtime_checkable
class Retriever(Protocol):
    """The Evidence source of the Live workflow."""

    def retrieve(self, query: str) -> list[Chunk]: ...


class SearchStore(Protocol):
    """The read path of the pgvector store."""

    def search_chunks(
        self,
        query_embedding: Sequence[float],
        limit: int,
        max_distance: float,
    ) -> Sequence[ScoredChunk]: ...


class VectorRetriever:
    """Embeds the query, vector-searches the store, returns relevant Chunks."""

    def __init__(
        self,
        store: SearchStore,
        embedder: Embedder,
        max_chunks: int = MAX_RETRIEVED_CHUNKS,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
    ):
        self._store = store
        self._embedder = embedder
        self._max_chunks = max_chunks
        self._min_similarity = min_similarity
        # The relevance floor as cosine distance, derived once: the same bound
        # is pushed down into the store's search and enforced again at this
        # seam, so both enforcements always agree.
        self._max_distance = 1 - self._min_similarity

    def retrieve(self, query: str) -> list[Chunk]:
        [query_embedding] = self._embedder.embed([query])
        hits = self._store.search_chunks(
            query_embedding=query_embedding,
            limit=self._max_chunks,
            max_distance=self._max_distance,
        )
        # The floor is enforced here, not only in the store's SQL: whatever a
        # Store implementation returns, no Chunk beyond the relevance floor
        # can cross this seam and become Evidence.
        return [hit.chunk for hit in hits if hit.distance <= self._max_distance]
