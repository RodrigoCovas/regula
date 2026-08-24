"""Unit tests for the retrieval service (spec #9, ticket #15).

The service composes the two provider protocols — an Embedder and a
search-capable Store — into the Retriever seam the Live workflow will call.
Tests inject deterministic fakes: no Ollama, no PostgreSQL, no network.
Real-PostgreSQL behaviour is covered by test_ingest_integration.py, and
semantic quality over the real corpus is verified operator-side against the
ingested stack.

The exactly-one-target invariant is not re-asserted here: results are
ScoredChunk-wrapped Chunks, so it is enforced by construction.
"""

import pytest

from src.models import Chunk, ProvisionKind, ScoredChunk
from src.retrieval import MAX_RETRIEVED_CHUNKS, VectorRetriever

from fakes import FakeEmbedder


class FakeSearchStore:
    """Records search calls; replays canned hits."""

    def __init__(self, hits):
        self.hits = hits
        self.calls: list[dict] = []

    def search_chunks(self, query_embedding, limit, max_distance):
        self.calls.append(
            {
                "query_embedding": query_embedding,
                "limit": limit,
                "max_distance": max_distance,
            }
        )
        return self.hits


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
    """One store hit with its provision metadata validated at construction."""
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


def make_retriever(hits, min_similarity=0.5):
    embedder = FakeEmbedder()
    store = FakeSearchStore(hits)
    retriever = VectorRetriever(store=store, embedder=embedder, min_similarity=min_similarity)
    return retriever, embedder, store


# --- Query flow: embed the query, search with its vector -----------------------


def test_retrieve_embeds_the_query_and_searches_the_store_with_its_vector():
    retriever, embedder, store = make_retriever(hits=[])

    retriever.retrieve("Is my credit scoring system high risk?")

    assert embedder.batches == [["Is my credit scoring system high risk?"]]
    (call,) = store.calls
    assert call["query_embedding"] == [38.0, 1.0]


# --- Results: hits unwrap to their metadata-complete Chunks --------------------


def test_retrieve_returns_each_hits_chunk_with_metadata_intact():
    hit = chunk_hit(
        source_id="gdpr",
        kind="recital",
        number=71,
        text="Recital 71 body",
        distance=0.25,
    )
    retriever, _, _ = make_retriever(hits=[hit])

    chunks = retriever.retrieve("automated decisions")

    (chunk,) = chunks
    assert chunk.source_id == "gdpr"
    assert chunk.kind is ProvisionKind.recital
    assert chunk.recital_number == 71
    assert chunk.article_number is None and chunk.annex_number is None
    assert chunk.text == "Recital 71 body"


# --- Budget: the single-pass budget is what gets requested ---------------------


def test_retrieve_requests_the_locked_single_pass_budget_from_the_store():
    retriever, _, store = make_retriever(hits=[])

    retriever.retrieve("oversight duties")

    assert store.calls[0]["limit"] == MAX_RETRIEVED_CHUNKS == 8


# --- Threshold: junk-only result sets come back empty --------------------------


def test_retrieve_converts_min_similarity_into_pgvector_max_distance():
    retriever, _, store = make_retriever(hits=[], min_similarity=0.45)

    retriever.retrieve("profiling")

    assert store.calls[0]["max_distance"] == pytest.approx(0.55)


def test_retrieve_yields_no_chunks_when_nothing_clears_the_threshold():
    retriever, _, _ = make_retriever(hits=[])

    assert retriever.retrieve("quantum gravity") == []


# --- Threshold enforcement: the floor holds at the seam, not only in SQL -------


def test_retrieve_drops_every_hit_beyond_the_relevance_floor():
    """A Store implementation that ignores max_distance cannot smuggle junk
    through the seam: the retriever enforces the relevance floor on every hit
    itself, so a junk-only result set can never reach drafting."""
    retriever, _, _ = make_retriever(hits=[chunk_hit(distance=0.9), chunk_hit(distance=0.99)])

    assert retriever.retrieve("anything at all") == []


def test_retrieve_keeps_thin_but_usable_hits_and_drops_junk():
    """The threshold separates 'nothing relevant' from 'thin but usable':
    a marginal hit just inside the floor survives, junk beyond it does not."""
    thin_but_usable = chunk_hit(source_id="gdpr", kind="recital", number=71, distance=0.45)
    junk = chunk_hit(source_id="spam", number=1, distance=0.51)
    retriever, _, _ = make_retriever(hits=[thin_but_usable, junk], min_similarity=0.5)

    chunks = retriever.retrieve("automated decisions")

    assert [c.source_id for c in chunks] == ["gdpr"]


def test_retrieve_keeps_a_hit_exactly_at_the_floor_like_the_stores_sql_does():
    """pgvector filters with ``distance <= max_distance``; the seam-side floor
    agrees on the boundary so both enforcements admit the same hits."""
    at_the_floor = chunk_hit(distance=0.5)
    retriever, _, _ = make_retriever(hits=[at_the_floor], min_similarity=0.5)

    chunks = retriever.retrieve("oversight duties")

    assert [c.source_id for c in chunks] == ["ai-act"]
