"""PostgreSQL + pgvector persistence for Chunks (spec #9, ticket #14).

The store owns the schema and one idempotent write path, and commits after
every operation so ingested rows survive the process that wrote them.
Ingestion is the only writer this iteration; retrieval (ticket #15) reads
the same table. Raw SQL over psycopg2 keeps the surface small — no ORM, no
migrations framework: ``ensure_schema`` creates everything IF NOT EXISTS.
"""

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import psycopg2
import psycopg2.extras

from .models import EMBEDDING_DIMENSION, Chunk, ProvisionKind

# nomic-embed-text (the locked embedding model) outputs 768-dimensional vectors.
EMBEDDING_DIMENSION = 768

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


class PgVectorStore:
    """The pgvector-backed Chunk store."""

    def __init__(self, connection):
        self._connection = connection

    def ensure_schema(self) -> None:
        with self._connection.cursor() as cursor:
            cursor.execute(_SCHEMA)
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
                "embedding": "[" + ",".join(repr(float(x)) for x in record.embedding) + "]",
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
                """
                SELECT source_id, kind, title, chunk_index, num_chunks,
                       article_number, recital_number, annex_number, content AS text,
                       vector_dims(embedding) AS embedding_dimensions
                FROM chunks
                WHERE source_id = %s
                ORDER BY kind, COALESCE(article_number, recital_number, annex_number), chunk_index
                """,
                (source_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def count_chunks(self) -> int:
        with self._connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM chunks")
            return int(cursor.fetchone()[0])
