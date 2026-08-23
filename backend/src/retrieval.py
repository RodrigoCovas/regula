"""Retrieval service — the Evidence-gathering primitive (spec #9, ticket #15).

Query text in, top-k most relevant Chunks out. ``VectorRetriever`` composes
the two provider protocols — it embeds the query locally via an ``Embedder``
and searches the pgvector store through a search-capable ``Store`` — so tests
and the Live workflow both depend on the single ``Retriever`` seam at the
composition root.

Locked behaviour:

- The result set is capped at the single-pass budget (~8 Chunks); retrieval
  is one bounded pass, never a crawl.
- A relevance threshold excludes junk-only result sets: when nothing in the
  Corpus matches well enough, the caller gets no Chunks rather than
  irrelevant Evidence. That empty result is what the Insufficient-evidence
  path builds on.
- Every returned Chunk carries its provision metadata (source_id plus exactly
  one provision number kind), validated by the Chunk model itself, ready for
  deterministic Citation derivation.

The threshold default was tuned against the real ingested Corpus with the
locked embedding model; cosine similarity of genuinely relevant provisions
sits far above junk matches.
"""

from typing import Any, Protocol, Sequence

from .embedder import Embedder
from .models import Chunk, ProvisionKind

# Locked single-pass budget (~8): the Researcher never sees more Evidence
# than this from one retrieval pass.
MAX_RETRIEVED_CHUNKS = 8

# Cosine similarity floor for a Chunk to count as relevant, tuned against
# the ingested Corpus with the locked embedding model: junk-only questions
# top out around 0.54 while targeted relevant provisions score from ~0.55 up
# (strong hits land at 0.69+). pgvector speaks distance
# (similarity = 1 - distance), so this converts once at the boundary.
DEFAULT_MIN_SIMILARITY = 0.55


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
    ) -> Sequence[dict[str, Any]]: ...


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

    def retrieve(self, query: str) -> list[Chunk]:
        [query_embedding] = self._embedder.embed([query])
        rows = self._store.search_chunks(
            query_embedding=query_embedding,
            limit=self._max_chunks,
            max_distance=1 - self._min_similarity,
        )
        chunks = [_row_to_chunk(row) for row in rows]
        return chunks[: self._max_chunks]


def _row_to_chunk(row: dict[str, Any]) -> Chunk:
    return Chunk(
        source_id=row["source_id"],
        kind=ProvisionKind(row["kind"]),
        title=row.get("title"),
        chunk_index=row["chunk_index"],
        num_chunks=row["num_chunks"],
        article_number=row["article_number"],
        recital_number=row["recital_number"],
        annex_number=row["annex_number"],
        text=row["text"],
    )
