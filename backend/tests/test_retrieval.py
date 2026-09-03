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
satisfy both the vector and the lexical read seam. The ``LazyStore`` check
opens no connection — deferral is the point — so the no-PostgreSQL promise
holds. Issue #64 added the hybrid section: both legs fuse by Reciprocal Rank
Fusion inside the retriever, and the seam stays query-in, Chunks-out.
"""

import pytest

from src.db import LazyStore
from src.models import ProvisionKind
from src.retrieval import (
    LEXICAL_LEG_DEPTH,
    RRF_K,
    VECTOR_LEG_DEPTH,
    HybridRetriever,
    HybridSearchStore,
    LexicalSearchStore,
    Retriever,
    SearchStore,
    VectorRetriever,
)

from fakes import FakeEmbedder, FakeSearchStore, chunk_hit, make_chunk


def make_retriever(hits, min_similarity=0.5):
    embedder = FakeEmbedder()
    store = FakeSearchStore(hits)
    retriever = VectorRetriever(store=store, embedder=embedder, min_similarity=min_similarity)
    return retriever, embedder, store


def make_hybrid(vector_hits, lexical_hits, min_similarity=0.5):
    store = FakeSearchStore(hits=vector_hits, lexical_hits=lexical_hits)
    retriever = HybridRetriever(store=store, embedder=FakeEmbedder(), min_similarity=min_similarity)
    return retriever, store


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
        kind="annex",
        number=3,
        text="Annex III body",
        distance=0.25,
    )
    retriever, _, _ = make_retriever(hits=[hit])

    chunks = retriever.retrieve("automated decisions")

    (chunk,) = chunks
    assert chunk.source_id == "gdpr"
    assert chunk.kind is ProvisionKind.annex
    assert chunk.annex_number == 3
    assert chunk.article_number is None and chunk.recital_number is None
    assert chunk.text == "Annex III body"


# --- Depth: the vector leg's per-target depth is what gets requested ----------


def test_vector_leg_requests_the_locked_per_target_depth_from_the_store():
    """One vector leg reaches ``VECTOR_LEG_DEPTH`` deep — the store's LIMIT,
    however many research targets the plan carries. The Evidence pool served
    to the agents is sized separately from the plan (live_workflow's
    SEATS_PER_TARGET), so widening that pool never deepens crawling (#23)."""
    retriever, _, store = make_retriever(hits=[])

    retriever.retrieve("oversight duties")

    assert store.calls[0]["limit"] == VECTOR_LEG_DEPTH == 12


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
    thin_but_usable = chunk_hit(source_id="gdpr", kind="annex", number=4, distance=0.45)
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
    assert isinstance(store, HybridSearchStore)


# --- Hybrid retrieval: both legs fused by RRF (issue #64, ADR-0012) ------------


def test_hybrid_retriever_satisfies_the_retriever_seam():
    """The workflow's Evidence source stays query-in, Chunks-out: fusion
    lives entirely inside the retrieval service."""
    retriever, _ = make_hybrid(vector_hits=[], lexical_hits=[])

    assert isinstance(retriever, Retriever)


def test_hybrid_fusion_orders_chunks_by_reciprocal_rank_contribution():
    """RRF scores a Chunk 1/(k + rank) per leg that found it, summed: a Chunk
    both legs found outranks either leg's own top hit, and single-leg Chunks
    order by their rank. Ranks start at 1, and the fusion constant is the
    standard k = 60 (ADR-0012). Within the 12/8 caps the fused order is
    k-stable for every k >= 8 — the ordering alone cannot distinguish k = 60
    from nearby values, so the constant itself is pinned here."""
    assert RRF_K == 60

    retriever, _ = make_hybrid(
        vector_hits=[
            chunk_hit(source_id="vector-top", number=1),
            chunk_hit(source_id="both-legs", number=2),
            chunk_hit(source_id="vector-last", number=3),
        ],
        lexical_hits=[
            make_chunk(source_id="both-legs", number=2),
            make_chunk(source_id="lexical-only", number=4),
        ],
    )

    chunks = retriever.retrieve("oversight duties")

    # both-legs: 1/61 + 1/62; vector-top: 1/61; lexical-only: 1/62;
    # vector-last: 1/63.
    assert [c.source_id for c in chunks] == ["both-legs", "vector-top", "lexical-only", "vector-last"]


def test_hybrid_fusion_earns_its_sum_at_the_legs_deepest_ranks():
    """The extreme within-caps case: a Chunk found by both legs at their
    deepest ranks (vector 12, lexical 8) outranks a single-leg Chunk at
    rank 1 — the summed contribution beats the top single-leg hit at k = 60.
    The ordering inverts only below k ≈ 7.8, which is what bounds the
    k-stability the constant pin carries."""
    shared = chunk_hit(source_id="shared", number=30)
    vector_hits = [chunk_hit(source_id=f"vec-{i}", number=i + 1) for i in range(11)] + [shared]
    lexical_hits = [make_chunk(source_id=f"lex-{i}", number=i + 1) for i in range(7)] + [shared.chunk]
    retriever, _ = make_hybrid(vector_hits=vector_hits, lexical_hits=lexical_hits)

    chunks = retriever.retrieve("oversight duties")

    assert chunks[0].source_id == "shared", "the summed both-legs contribution outranks every single-leg hit"
    assert len(chunks) == 19, "all 12 vector + 8 lexical hits minus the one dedup"


def test_hybrid_fusion_breaks_score_ties_by_first_appearance():
    """Two single-leg Chunks at the same rank tie under RRF; the chunk seen
    first (the vector leg's, ranked before the lexical leg) keeps precedence —
    the fusion is deterministic."""
    retriever, _ = make_hybrid(
        vector_hits=[chunk_hit(source_id="from-vector", number=1)],
        lexical_hits=[make_chunk(source_id="from-lexical", number=2)],
    )

    chunks = retriever.retrieve("oversight duties")

    assert [c.source_id for c in chunks] == ["from-vector", "from-lexical"]


def test_hybrid_dedups_a_chunk_both_legs_found():
    """A Chunk both legs found appears exactly once — its RRF contributions
    sum into one fused entry, never a duplicate across legs."""
    retriever, _ = make_hybrid(
        vector_hits=[
            chunk_hit(source_id="shared", number=7),
            chunk_hit(source_id="vector-only", number=1),
        ],
        lexical_hits=[make_chunk(source_id="shared", number=7)],
    )

    chunks = retriever.retrieve("oversight duties")

    assert [c.source_id for c in chunks] == ["shared", "vector-only"]


def test_hybrid_vector_floor_binds_the_vector_leg_only():
    """Every vector hit sits beyond the relevance floor, yet the lexical
    leg's exact-term matches survive: the floor never gate-keeps the lexical
    leg, so lexical-only evidence may fill the pool (ADR-0012)."""
    retriever, _ = make_hybrid(
        vector_hits=[chunk_hit(source_id="junk", number=1, distance=0.9)],
        lexical_hits=[make_chunk(source_id="gdpr", number=30)],
    )

    chunks = retriever.retrieve("records of processing activities")

    assert [c.source_id for c in chunks] == ["gdpr"]


def test_hybrid_requests_each_legs_own_cap_from_the_store():
    """Two legs, two caps, each pushed down into its own store read (ADR-0012):
    the vector leg asks for top-12, the lexical leg for top-8."""
    retriever, store = make_hybrid(vector_hits=[], lexical_hits=[])

    retriever.retrieve("oversight duties")

    assert store.calls[0]["limit"] == VECTOR_LEG_DEPTH == 12
    assert store.lexical_calls[0]["limit"] == LEXICAL_LEG_DEPTH == 8
    assert store.lexical_calls[0]["query"] == "oversight duties"


# --- Recital exclusion: operative provisions only (ticket #69) -----------------


def test_vector_leg_pushes_the_recital_exclusion_down_to_the_store():
    """The Recital exclusion rides the vector leg's pushed-down parameters
    (ticket #69): the store's SQL can filter before its LIMIT is spent, so a
    Recital Chunk never consumes vector leg depth — operative provisions take
    those seats instead."""
    retriever, _, store = make_retriever(hits=[])

    retriever.retrieve("oversight duties")

    assert store.calls[0]["excluded_kinds"] == (ProvisionKind.recital,)


def test_lexical_leg_pushes_the_recital_exclusion_down_to_the_store():
    """The same pushed-down exclusion on the lexical leg: neither read path
    can seat a Recital Chunk ahead of an operative provision (ticket #69)."""
    retriever, store = make_hybrid(vector_hits=[], lexical_hits=[])

    retriever.retrieve("oversight duties")

    assert store.lexical_calls[0]["excluded_kinds"] == (ProvisionKind.recital,)


def test_vector_leg_drops_recital_hits_a_store_returns_anyway():
    """A Store implementation that ignores the pushed-down exclusion cannot
    smuggle a Recital through the seam: like the relevance floor, the
    exclusion is re-checked here, so a Recital Chunk never crosses into
    Evidence (ticket #69)."""
    retriever, _, _ = make_retriever(
        hits=[
            chunk_hit(source_id="gdpr", kind="recital", number=71, distance=0.1),
            chunk_hit(source_id="ai-act", number=6, distance=0.2),
        ]
    )

    chunks = retriever.retrieve("automated decisions")

    assert [c.source_id for c in chunks] == ["ai-act"]
    assert all(c.kind is not ProvisionKind.recital for c in chunks)


def test_hybrid_drops_recital_hits_returned_by_either_leg():
    """Both legs are re-checked at the seam (ticket #69): Recitals a store
    returns anyway are dropped before fusion, and the operative provisions
    around them keep their fused order."""
    retriever, _ = make_hybrid(
        vector_hits=[
            chunk_hit(source_id="gdpr", kind="recital", number=71, distance=0.1),
            chunk_hit(source_id="ai-act", number=6),
        ],
        lexical_hits=[
            make_chunk(source_id="gdpr", kind=ProvisionKind.recital, number=70),
            make_chunk(source_id="gdpr", number=30),
        ],
    )

    chunks = retriever.retrieve("oversight duties")

    assert [c.source_id for c in chunks] == ["ai-act", "gdpr"]
    assert all(c.kind is not ProvisionKind.recital for c in chunks)


def test_hybrid_returns_empty_when_both_legs_come_back_empty():
    """Neither leg matched anything: the retriever returns empty — the shape
    the Insufficient-evidence path builds on."""
    retriever, _ = make_hybrid(vector_hits=[], lexical_hits=[])

    assert retriever.retrieve("quantum gravity") == []


def test_hybrid_returns_empty_when_vector_hits_are_junk_and_lexical_finds_nothing():
    """Vector hits beyond the floor plus zero lexical matches is both legs
    empty: no Chunk crosses the seam, however much the store returned."""
    retriever, _ = make_hybrid(
        vector_hits=[chunk_hit(source_id="junk", number=1, distance=0.9)],
        lexical_hits=[],
    )

    assert retriever.retrieve("quantum gravity") == []
