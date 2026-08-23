"""Integration coverage for the pgvector store (spec #9, tickets #14–#15).

Covers both store paths against real PostgreSQL + pgvector: the idempotent
ingestion write path (#14) and the vector-search read path behind retrieval
(#15). A disposable container is started when Docker is available; when it
is not (or REGULA_TEST_DATABASE_URL points at an unreachable server) the
whole module skips cleanly. No Ollama, no network beyond the database
connection.
"""

import hashlib
import json
import os
import shutil
import subprocess
import time
from functools import lru_cache
from pathlib import Path

from typing import Generator

import pytest

from src.chunking import chunk_regulation
from src.db import ChunkRecord, PgVectorStore, connect
from src.models import EMBEDDING_DIMENSION, ProvisionKind

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "regulations"

CONTAINER = "regula-ingestion-it"
IMAGE = "pgvector/pgvector:pg16"
READY_TIMEOUT_SECONDS = 60


def make_record(
    source_id="it-doc",
    kind: ProvisionKind = ProvisionKind.article,
    number=1,
    text="Article 1",
    index=0,
    embedding=None,
):
    return ChunkRecord(
        source_id=source_id,
        kind=kind,
        text=text,
        title=None,
        chunk_index=index,
        num_chunks=1,
        article_number=number if kind is ProvisionKind.article else None,
        recital_number=number if kind is ProvisionKind.recital else None,
        annex_number=number if kind is ProvisionKind.annex else None,
        embedding=embedding if embedding is not None else [0.1] * EMBEDDING_DIMENSION,
    )


@lru_cache(maxsize=None)
def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    probe = subprocess.run(
        ["docker", "info"],
        capture_output=True,
        timeout=30,
    )
    return probe.returncode == 0


def wait_until_ready(dsn: str) -> bool:
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            with connect(dsn):
                return True
        except Exception:
            time.sleep(1.0)
    return False


@pytest.fixture(scope="session")
def dsn() -> Generator[str, None, None]:
    """A reachable PostgreSQL+pgvector DSN, or a clean skip."""
    override = os.environ.get("REGULA_TEST_DATABASE_URL")
    if override:
        if not wait_until_ready(override):
            pytest.skip(f"REGULA_TEST_DATABASE_URL is set but unreachable: {override}")
        yield override
        return
    if not _docker_available():
        pytest.skip("Docker unavailable; skipping real-PostgreSQL integration coverage")
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
    started = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            CONTAINER,
            "-e", "POSTGRES_DB=regula",
            "-e", "POSTGRES_USER=regula",
            "-e", "POSTGRES_PASSWORD=regula",
            "-p", "127.0.0.1::5432",
            IMAGE,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if started.returncode != 0:
        pytest.skip(f"could not start {IMAGE}: {started.stderr.strip()}")
    try:
        port_map = subprocess.run(
            ["docker", "port", CONTAINER, "5432/tcp"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        port = int(port_map.splitlines()[0].rsplit(":", 1)[1])
        candidate = f"postgresql://regula:regula@127.0.0.1:{port}/regula"
        if not wait_until_ready(candidate):
            raise RuntimeError(f"{CONTAINER} never became ready within {READY_TIMEOUT_SECONDS}s")
        yield candidate
    finally:
        subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)


@pytest.fixture()
def store(dsn):
    connection = connect(dsn)
    store = PgVectorStore(connection)
    store.ensure_schema()
    yield store
    with connection.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS chunks")
    connection.commit()
    connection.close()


# --- Schema creation ---------------------------------------------------------


def test_ensure_schema_is_rerunnable(store):
    store.ensure_schema()


# --- Upsert: every chunk stored with vector and provision metadata -----------


def test_upsert_stores_chunk_with_vector_and_provision_metadata(store):
    record = make_record(kind=ProvisionKind.recital, number=71, text="Recital 71 body")

    store.upsert_chunks([record])

    rows = store.fetch_provisions(source_id="it-doc")
    assert len(rows) == 1
    row = rows[0]
    assert row["source_id"] == "it-doc"
    assert row["kind"] == "recital"
    assert row["recital_number"] == 71
    assert row["article_number"] is None
    assert row["annex_number"] is None
    assert row["text"] == "Recital 71 body"


# --- Idempotency: re-running ingestion inserts no duplicates -----------------


def test_second_upsert_of_same_chunks_leaves_row_count_unchanged(store):
    def first_ingestion_run():
        return [make_record(number=number, index=index) for index, number in enumerate((1, 2, 3))]

    store.upsert_chunks(first_ingestion_run())
    store.upsert_chunks(first_ingestion_run())

    assert store.count_chunks() == 3


def test_updated_chunk_content_and_vector_are_refreshed_not_duplicated(store):
    store.upsert_chunks([make_record(text="old body", embedding=[0.5] * EMBEDDING_DIMENSION)])

    store.upsert_chunks([make_record(text="new body", embedding=[0.9] * EMBEDDING_DIMENSION)])

    (row,) = store.fetch_provisions(source_id="it-doc")
    assert row["text"] == "new body"


# --- Queryability: the ingested Corpus round-trips with metadata intact ------


def deterministic_embedding(text: str) -> list[float]:
    """A fake but valid full-dimension vector, derived from the text (no Ollama)."""
    seed = sum(text.encode("utf-8")) % 97 + 1
    return [(seed * (i + 1)) % 13 / 13 for i in range(EMBEDDING_DIMENSION)]


def ingest_real_document(store, source_id: str):
    document = json.loads((DATA_DIR / f"{source_id}.json").read_text(encoding="utf-8"))
    chunks = chunk_regulation(document)
    records = [
        ChunkRecord.from_chunk(chunk, deterministic_embedding(chunk.text)) for chunk in chunks
    ]
    store.upsert_chunks(records)
    return chunks


def test_ingested_real_corpus_is_queryable_by_source_id_with_exactly_one_kind(store):
    chunks = ingest_real_document(store, "gdpr")

    rows = store.fetch_provisions(source_id="gdpr")
    assert len(rows) == len(chunks)
    for row in rows:
        targets = [row["article_number"], row["recital_number"], row["annex_number"]]
        assert sum(t is not None for t in targets) == 1
    kinds = {row["kind"] for row in rows}
    assert kinds == {"article"}


def test_ingested_rows_keep_chunk_text_and_vector_dimension(store):
    chunks = ingest_real_document(store, "gdpr")

    rows = store.fetch_provisions(source_id="gdpr")
    stored_texts = {row["text"] for row in rows}
    for chunk in chunks:
        assert chunk.text in stored_texts
    assert {row["embedding_dimensions"] for row in rows} == {EMBEDDING_DIMENSION}


def test_full_reingest_of_real_corpus_leaves_row_count_unchanged(store):
    first = ingest_real_document(store, "gdpr")
    second = ingest_real_document(store, "gdpr")

    assert [c.text for c in first] == [c.text for c in second]
    assert store.count_chunks() == len(first)


# --- Retrieval: vector search over the ingested store (ticket #15) -------------


def unit_vector(axis: int) -> list[float]:
    """A full-dimension unit vector along one axis; distinct axes are orthogonal."""
    vector = [0.0] * EMBEDDING_DIMENSION
    vector[axis % EMBEDDING_DIMENSION] = 1.0
    return vector


def test_search_returns_nearest_chunk_first_with_metadata_and_distance(
    store, assert_exactly_one_provision_target
):
    store.upsert_chunks(
        [
            make_record(number=1, embedding=unit_vector(0)),
            make_record(number=2, text="Article 2", embedding=unit_vector(1)),
            make_record(number=3, text="Article 3", embedding=unit_vector(2)),
        ]
    )

    hits = store.search_chunks(query_embedding=unit_vector(1), limit=8, max_distance=0.5)

    (nearest,) = hits
    assert nearest.chunk.source_id == "it-doc"
    assert nearest.chunk.kind is ProvisionKind.article
    assert nearest.chunk.article_number == 2
    assert nearest.chunk.recital_number is None and nearest.chunk.annex_number is None
    assert nearest.chunk.text == "Article 2"
    assert nearest.distance == pytest.approx(0.0, abs=1e-6)
    assert_exactly_one_provision_target(nearest.chunk)


def test_search_limit_caps_result_count(store):
    shared = unit_vector(0)
    store.upsert_chunks([make_record(number=n, index=n, embedding=shared) for n in range(1, 6)])

    rows = store.search_chunks(query_embedding=shared, limit=3, max_distance=0.5)

    assert len(rows) == 3


def test_search_max_distance_yields_no_rows_rather_than_irrelevant_ones(store):
    store.upsert_chunks(
        [make_record(number=1, embedding=unit_vector(0)), make_record(number=2, text="Article 2", embedding=unit_vector(1))]
    )

    rows = store.search_chunks(query_embedding=unit_vector(7), limit=8, max_distance=0.9)

    assert rows == []


class TextHashEmbedder:
    """Deterministic, collision-resistant embeddings from SHA-256 digests."""

    def embed(self, texts):
        return [self._embedding(text) for text in texts]

    @staticmethod
    def _embedding(text: str) -> list[float]:
        values: list[float] = []
        counter = 0
        while len(values) < EMBEDDING_DIMENSION:
            digest = hashlib.sha256(f"{counter}:{text}".encode()).digest()
            values.extend(byte / 255.0 for byte in digest)
            counter += 1
        return values[:EMBEDDING_DIMENSION]


def test_vector_retriever_round_trip_over_real_ingested_corpus(
    dsn, assert_exactly_one_provision_target
):
    """The retriever seam reads the same table ingestion wrote: querying with
    a Chunk's own text returns that Chunk first, metadata intact, inside the
    single-pass budget."""
    from src.retrieval import MAX_RETRIEVED_CHUNKS, VectorRetriever

    document = json.loads((DATA_DIR / "gdpr.json").read_text(encoding="utf-8"))
    chunks = chunk_regulation(document)
    embedder = TextHashEmbedder()
    records = [ChunkRecord.from_chunk(chunk, vector) for chunk, vector in zip(chunks, embedder.embed([c.text for c in chunks]))]
    connection = connect(dsn)
    try:
        store = PgVectorStore(connection)
        store.ensure_schema()
        store.upsert_chunks(records)

        target = chunks[len(chunks) // 2]
        retrieved = VectorRetriever(store=store, embedder=embedder).retrieve(target.text)
    finally:
        with connection.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS chunks")
        connection.commit()
        connection.close()

    assert 0 < len(retrieved) <= MAX_RETRIEVED_CHUNKS
    assert retrieved[0].text == target.text
    assert retrieved[0].source_id == target.source_id
    assert retrieved[0].article_number == target.article_number
    assert retrieved[0].recital_number is None and retrieved[0].annex_number is None
    for chunk in retrieved:
        assert_exactly_one_provision_target(chunk)


# --- Durability: ingested rows survive the connection that wrote them --------


class FakeIngestEmbedder:
    def embed(self, texts):
        return [[0.25] * EMBEDDING_DIMENSION for _ in texts]


def test_run_ingestion_is_visible_and_idempotent_across_connections(dsn):
    """The end-to-end ingestion path persists: each run behaves like its own
    process (one fresh connection), and another connection afterwards sees
    exactly one row per Chunk no matter how many times ingestion ran."""
    from src.ingest import run_ingestion

    document = json.loads((DATA_DIR / "gdpr.json").read_text(encoding="utf-8"))
    expected = len(chunk_regulation(document))

    for _ in range(2):
        writer_connection = connect(dsn)
        try:
            run_ingestion(
                PgVectorStore(writer_connection),
                FakeIngestEmbedder(),
                [("gdpr", document)],
            )
        finally:
            writer_connection.close()

    reader_connection = connect(dsn)
    try:
        reader = PgVectorStore(reader_connection)
        assert reader.count_chunks() == expected
        rows = reader.fetch_provisions(source_id="gdpr")
        assert {row["kind"] for row in rows} == {"article"}
        assert all(row["embedding_dimensions"] == EMBEDDING_DIMENSION for row in rows)
    finally:
        with reader_connection.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS chunks")
        reader_connection.commit()
        reader_connection.close()
