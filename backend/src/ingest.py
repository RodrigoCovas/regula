"""Corpus ingestion CLI (spec #9, ticket #14).

One documented command ingests the Corpus: it reads the three lexplorer JSON
documents, applies the locked hybrid chunking (#11), embeds each Chunk locally
with the Ollama embedding model (nomic-embed-text v1.5), and upserts everything
into PostgreSQL + pgvector with its provision metadata. Re-runs refresh existing
rows and insert no duplicates (idempotency is enforced by the store's identity
key).

The backend never ingests on startup: this explicit command is the documented
setup path, so a slow model download never hides inside app boot.

Run from the repository root:

    python -m backend.src.ingest

or inside the stack:

    docker compose exec backend python -m backend.src.ingest
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

from .chunking import chunk_regulation
from .config import DEFAULT_DATABASE_URL, DEFAULT_OLLAMA_URL
from .db import ChunkRecord, PgVectorStore, connect
from .embedder import DEFAULT_MODEL, Embedder, EmbeddingError, OllamaEmbedder

DEFAULT_DATA_DIR = Path("data/regulations")


class IngestError(RuntimeError):
    """Ingestion cannot proceed; the message names what to do about it."""


class Store(Protocol):
    def ensure_schema(self) -> None: ...
    def upsert_chunks(self, records: Sequence[ChunkRecord]) -> int: ...


@dataclass(frozen=True)
class IngestReport:
    chunks_ingested: int = 0
    sources: list[str] = field(default_factory=list)


def load_corpus_documents(data_dir: Path) -> list[tuple[str, dict]]:
    """Every lexplorer JSON document in ``data_dir``, ordered by file name."""
    if not data_dir.is_dir():
        raise IngestError(
            f"Corpus directory {data_dir} does not exist. Pass --data-dir or run "
            "from the repository root."
        )
    documents = []
    for path in sorted(data_dir.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise IngestError(f"Could not read corpus file {path}: {error}") from error
        documents.append((str(document["metadata"]["id"]), document))
    if not documents:
        raise IngestError(f"No corpus JSON documents found in {data_dir}.")
    return documents


def run_ingestion(
    store: Store,
    embedder: Embedder,
    documents: Sequence[tuple[str, dict]],
) -> IngestReport:
    """Transform, embed, and upsert every Chunk of every document.

    The embedder owns batching (its ``batch_size`` bounds one request);
    each batch is upserted as soon as it is embedded.
    """
    store.ensure_schema()
    sources = [source_id for source_id, _ in documents]
    ingested = 0
    for _, document in documents:
        chunks = chunk_regulation(document)
        vectors = embedder.embed([chunk.text for chunk in chunks])
        records = [
            ChunkRecord.from_chunk(chunk, vector) for chunk, vector in zip(chunks, vectors)
        ]
        store.upsert_chunks(records)
        ingested += len(records)
    return IngestReport(chunks_ingested=ingested, sources=sources)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m backend.src.ingest",
        description="Embed and upsert the Corpus into PostgreSQL/pgvector (idempotent).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"directory holding the lexplorer JSON documents (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL),
        help="PostgreSQL connection URL (default: DATABASE_URL env or local compose stack)",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_API_URL", DEFAULT_OLLAMA_URL),
        help="Ollama base URL (default: OLLAMA_API_URL env or local compose stack)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"embedding model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="texts per embedding request")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        documents = load_corpus_documents(args.data_dir)
        embedder: Embedder = OllamaEmbedder(
            base_url=args.ollama_url, model=args.model, batch_size=args.batch_size
        )
        store = PgVectorStore(connect(args.database_url))
        report = run_ingestion(store, embedder, documents)
    except (IngestError, EmbeddingError) as error:
        print(f"Ingestion failed: {error}", file=sys.stderr)
        return 1
    print(
        f"Ingestion complete: {report.chunks_ingested} chunks stored "
        f"(inserted or updated) from {len(report.sources)} sources "
        f"({', '.join(report.sources)}). Re-running is safe: existing rows are "
        "updated, no duplicates are inserted."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
