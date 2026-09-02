"""Unit tests for the retrieval service (spec #9, tickets #15 and #18).

The service composes the two provider protocols — an Embedder and a
search-capable Store — into the Retriever seam the Live workflow will call.
Tests inject deterministic fakes: no Ollama, no PostgreSQL, no network.
Real-PostgreSQL behaviour is covered by test_ingest_integration.py, and
semantic quality over the real corpus is verified operator-side against the
ingested stack.

The exactly-one-target invariant is not re-asserted here: results are
ScoredChunk-wrapped Chunks, so it is enforced by construction.

Ticket #18 added the threshold-enforcement section: the relevance floor is
re-checked at this seam, so junk-only search results can never reach drafting.
Spec #60 added the read-seams section: the composition-root store must
satisfy both the vector and the lexical read seam.
"""

import pytest

from src.db import LazyStore
from src.models import ProvisionKind
from src.retrieval import PER_TARGET_DEPTH, LexicalSearchStore, SearchStore, VectorRetriever

from fakes import FakeEmbedder, FakeSearchStore, chunk_hit


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


# --- Depth: the per-target depth is what gets requested ------------------------


def test_retrieve_requests_the_locked_per_target_depth_from_the_store():
    """One retrieval reaches ``PER_TARGET_DEPTH`` deep — the store's LIMIT,
    however many research targets the plan carries. The Evidence pool served
    to the agents is sized separately from the plan (live_workflow's
    SEATS_PER_TARGET), so widening that pool never deepens crawling (#23)."""
    retriever, _, store = make_retriever(hits=[])

    retriever.retrieve("oversight duties")

    assert store.calls[0]["limit"] == PER_TARGET_DEPTH == 8


# --- Threshold: junk-only search results come back empty -----------------------


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
    itself, so junk can never enter the Evidence pool and reach drafting."""
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


def test_retrieve_admits_a_hit_exactly_at_the_floor():
    """The seam-side floor uses the same inclusive boundary as the store's
    SQL (``distance <= max_distance``, db.py), so both enforcements admit
    identical hits. The SQL side itself is exercised against real Postgres
    by test_ingest_integration.py."""
    at_the_floor = chunk_hit(distance=0.5)
    retriever, _, _ = make_retriever(hits=[at_the_floor], min_similarity=0.5)

    chunks = retriever.retrieve("oversight duties")

    assert [c.source_id for c in chunks] == ["ai-act"]


# --- Read seams: the store's two read paths stay separate protocols ------------


def test_composition_root_store_satisfies_both_read_seams():
    """The lexical capability enters as its own read seam (spec #60): the
    request-scoped store serves vector and lexical reads over one deferred
    connection, while ``SearchStore`` and its doubles stay untouched."""
    store = LazyStore("postgresql://unused")

    assert isinstance(store, SearchStore)
    assert isinstance(store, LexicalSearchStore)
