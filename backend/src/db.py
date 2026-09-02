"""PostgreSQL + pgvector persistence for Chunks (spec #9, tickets #14–#15, #60).

The store owns the schema, one idempotent write path, and the two read paths
the retrieval service calls — vector search and lexical full-text search
(spec #60, ADR-0012) — and commits after every write so ingested rows
survive the process that wrote them. Raw SQL over psycopg2 keeps the surface
small — no ORM, no migrations framework: ``ensure_schema`` creates everything
IF NOT EXISTS.
"""

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import psycopg2
import psycopg2.extras

from .models import EMBEDDING_DIMENSION, Chunk, ProvisionKind, ScoredChunk

_SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    source_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    title TEXT,
    chunk_index INT NOT NULL DEFAULT 0,
    num_chunks INT NOT NULL DEFAULT 1,
    article_number INT,
    recital_number INT,
    annex_number INT,
    content TEXT NOT NULL,
    embedding vector({EMBEDDING_DIMENSION}) NOT NULL,
    CONSTRAINT chunks_exactly_one_target
        CHECK (num_nonnulls(article_number, recital_number, annex_number) = 1),
    CONSTRAINT chunks_kind_matches_target CHECK (
        (kind = 'article' AND article_number IS NOT NULL)
        OR (kind = 'recital' AND recital_number IS NOT NULL)
        OR (kind = 'annex' AND annex_number IS NOT NULL)
    )
);

-- Identity of a Chunk under the deterministic transform; the target of the
-- idempotent upsert. COALESCE collapses the exactly-one provision number.
CREATE UNIQUE INDEX IF NOT EXISTS chunks_identity_key
    ON chunks (source_id, kind,
               COALESCE(article_number, recital_number, annex_number),
               chunk_index);

-- The lexical read path (ADR-0012): a generated tsvector over the Chunk
-- content in the english configuration, backfilled for pre-existing rows by
-- the one-time table rewrite this ALTER performs on first start after
-- upgrade — no re-ingestion, no change to the write path. The GIN index
-- keeps the term match off a sequential scan; ts_rank then scores only the
-- matched rows.
ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX IF NOT EXISTS chunks_content_tsv_idx
    ON chunks USING GIN (content_tsv);
"""

_UPSERT = """
INSERT INTO chunks (source_id, kind, title, chunk_index, num_chunks,
                    article_number, recital_number, annex_number, content, embedding)
VALUES (%(source_id)s, %(kind)s, %(title)s, %(chunk_index)s, %(num_chunks)s,
        %(article_number)s, %(recital_number)s, %(annex_number)s, %(content)s,
        %(embedding)s::vector)
ON CONFLICT (source_id, kind,
             COALESCE(article_number, recital_number, annex_number),
             chunk_index)
DO UPDATE SET title = EXCLUDED.title,
              num_chunks = EXCLUDED.num_chunks,
              content = EXCLUDED.content,
              embedding = EXCLUDED.embedding
"""

# The generated column's declared expression, read back after the DDL:
# ADD COLUMN IF NOT EXISTS matches by name, so a content_tsv generated with
# any other configuration would silently rank with the wrong lexicon.
_CONTENT_TSV_EXPRESSION = """
SELECT generation_expression FROM information_schema.columns
WHERE table_name = 'chunks' AND column_name = 'content_tsv'
"""


def _to_pgvector(values: Sequence[float]) -> str:
    """A vector literal in pgvector's text format."""
    return "[" + ",".join(repr(float(x)) for x in values) + "]"


# The Chunk projection shared by every read path; its column aliases pair
# with _row_to_chunk, the one place those names become Chunk fields.
_CHUNK_COLUMNS = """source_id, kind, title, chunk_index, num_chunks,
                    article_number, recital_number, annex_number,
                    content AS text"""


def _row_to_chunk(row: dict[str, Any]) -> Chunk:
    """The one place SQL column names become Chunk fields."""
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


def _row_to_scored_chunk(row: dict[str, Any]) -> ScoredChunk:
    return ScoredChunk(chunk=_row_to_chunk(row), distance=float(row["distance"]))


@dataclass(frozen=True)
class ChunkRecord:
    """One persistable Chunk with its embedding."""

    source_id: str
    kind: ProvisionKind
    text: str
    title: Optional[str]
    chunk_index: int
    num_chunks: int
    article_number: Optional[int]
    recital_number: Optional[int]
    annex_number: Optional[int]
    embedding: Sequence[float]

    @classmethod
    def from_chunk(cls, chunk: Chunk, embedding: Sequence[float]) -> "ChunkRecord":
        return cls(
            source_id=chunk.source_id,
            kind=ProvisionKind(chunk.kind),
            text=chunk.text,
            title=chunk.title,
            chunk_index=chunk.chunk_index,
            num_chunks=chunk.num_chunks,
            article_number=chunk.article_number,
            recital_number=chunk.recital_number,
            annex_number=chunk.annex_number,
            embedding=embedding,
        )


def connect(dsn: str):
    return psycopg2.connect(dsn)


class LazyStore:
    """A ``SearchStore`` and ``LexicalSearchStore`` over one deferred connection.

    Live mode's dependency is resolved for every request, but Demo mode may
    never touch PostgreSQL at all — so the connection is deferred until a
    search actually happens and closed when the request scope ends.
    """

    def __init__(self, dsn: str):
        self._dsn = dsn
        self._connection = None

    def _ensure_connection(self):
        if self._connection is None:
            self._connection = connect(self._dsn)
        return self._connection

    def search_chunks(
        self,
        query_embedding: Sequence[float],
        limit: int,
        max_distance: float,
    ) -> list[ScoredChunk]:
        return PgVectorStore(self._ensure_connection()).search_chunks(
            query_embedding=query_embedding,
            limit=limit,
            max_distance=max_distance,
        )

    def search_chunks_lexically(self, query: str, limit: int) -> list[Chunk]:
        return PgVectorStore(self._ensure_connection()).search_chunks_lexically(
            query=query,
            limit=limit,
        )

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None


class PgVectorStore:
    """The pgvector-backed Chunk store."""

    def __init__(self, connection):
        self._connection = connection

    def ensure_schema(self) -> None:
        """Create everything IF NOT EXISTS, then verify the generated lexical
        column's expression: name-level idempotency must not silently keep a
        foreign configuration."""
        with self._connection.cursor() as cursor:
            cursor.execute(_SCHEMA)
            cursor.execute(_CONTENT_TSV_EXPRESSION)
            row = cursor.fetchone()
        declared = row[0] if row else None
        if declared is None or "to_tsvector('english" not in declared:
            self._connection.rollback()
            raise RuntimeError(
                "chunks.content_tsv does not carry the english tsvector "
                f"generation the lexical read path requires (found: {declared!r}). "
                "Fix with `ALTER TABLE chunks DROP COLUMN content_tsv`, then "
                "re-run the schema step to rebuild it from chunks.content."
            )
        self._connection.commit()

    def upsert_chunks(self, records: Sequence[ChunkRecord]) -> int:
        parameters = [
            {
                "source_id": record.source_id,
                "kind": record.kind.value,
                "title": record.title,
                "chunk_index": record.chunk_index,
                "num_chunks": record.num_chunks,
                "article_number": record.article_number,
                "recital_number": record.recital_number,
                "annex_number": record.annex_number,
                "content": record.text,
                "embedding": _to_pgvector(record.embedding),
            }
            for record in records
        ]
        with self._connection.cursor() as cursor:
            psycopg2.extras.execute_batch(cursor, _UPSERT, parameters)
        self._connection.commit()
        return len(records)

    def fetch_provisions(self, source_id: str) -> list[dict[str, Any]]:
        """Stored provision metadata for one source, ordered for stable assertions."""
        with self._connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(
                f"""
                SELECT {_CHUNK_COLUMNS},
                       vector_dims(embedding) AS embedding_dimensions
                FROM chunks
                WHERE source_id = %s
                ORDER BY kind, COALESCE(article_number, recital_number, annex_number), chunk_index
                """,
                (source_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def search_chunks(
        self,
        query_embedding: Sequence[float],
        limit: int,
        max_distance: float,
    ) -> list[ScoredChunk]:
        """Nearest Chunks to a query vector, nearest first, within max_distance.

        Cosine distance (pgvector ``<=>``) is the ranking metric; results
        beyond ``max_distance`` are excluded entirely rather than returned as
        junk. Each result wraps its fully validated Chunk with the distance
        that scored it.
        """
        with self._connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(
                f"""
                SELECT {_CHUNK_COLUMNS},
                       embedding <=> %(query)s::vector AS distance
                FROM chunks
                WHERE embedding <=> %(query)s::vector <= %(max_distance)s
                ORDER BY distance ASC, id ASC
                LIMIT %(limit)s
                """,
                {
                    "query": _to_pgvector(query_embedding),
                    "max_distance": max_distance,
                    "limit": limit,
                },
            )
            return [_row_to_scored_chunk(row) for row in cursor.fetchall()]

    def search_chunks_lexically(self, query: str, limit: int) -> list[Chunk]:
        """Chunks whose content matches the query's terms, best rank first.

        Postgres full-text search over the generated tsvector is the ranking
        metric — ts_rank with Postgres's default weights and normalization,
        highest first — the lexical leg of hybrid retrieval (ADR-0012). The
        query parses with ``websearch_to_tsquery`` so search operators stay
        meaningful; each result is a fully validated Chunk, and the LIMIT
        clause bounds one query like the vector path does.
        """
        with self._connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(
                f"""
                WITH parsed AS (SELECT websearch_to_tsquery('english', %(query)s) AS terms)
                SELECT {_CHUNK_COLUMNS}
                FROM chunks, parsed
                WHERE content_tsv @@ parsed.terms
                ORDER BY ts_rank(content_tsv, parsed.terms) DESC, id ASC
                LIMIT %(limit)s
                """,
                {"query": query, "limit": limit},
            )
            return [_row_to_chunk(row) for row in cursor.fetchall()]

    def count_chunks(self) -> int:
        with self._connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM chunks")
            return int(cursor.fetchone()[0])
