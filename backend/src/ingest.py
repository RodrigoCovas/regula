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
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

from pydantic import ValidationError

from .chunking import chunk_regulation
from .config import Settings, source_local_env
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


def load_corpus_documents(data_dir: Path, log=print) -> list[tuple[str, dict]]:
    """Every lexplorer JSON document in ``data_dir``, ordered by file name.

    A file is a corpus document when it carries a ``metadata`` object; JSON
    files without one (e.g. the ground-truth ``citations.json`` that shares
    the directory) are skipped with a logged note, never ingested or
    crashed on.
    """
    if not data_dir.is_dir():
        raise IngestError(
            f"Corpus directory {data_dir} does not exist. Pass --data-dir or run "
            "from the repository root."
        )
    documents = []
    skipped = []
    for path in sorted(data_dir.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise IngestError(f"Could not read corpus file {path}: {error}") from error
        if not isinstance(document, dict) or "metadata" not in document:
            skipped.append(path.name)
            continue
        source_id = str(document["metadata"]["id"])
        documents.append((source_id, document))
    if skipped:
        log(f"Skipped non-corpus JSON documents: {', '.join(skipped)}")
    if not documents:
        raise IngestError(f"No corpus JSON documents found in {data_dir}.")
    source_ids = [s for s, _ in documents]
    log(f"Loaded {len(documents)} corpus documents: {', '.join(source_ids)}")
    return documents


def run_ingestion(
    store: Store,
    embedder: Embedder,
    documents: Sequence[tuple[str, dict]],
    log=print,
) -> IngestReport:
    """Transform, embed, and upsert every Chunk of every document.

    The embedder owns batching (its ``batch_size`` bounds one request);
    each batch is upserted as soon as it is embedded.
    """
    store.ensure_schema()
    sources = [source_id for source_id, _ in documents]
    ingested = 0
    for source_id, document in documents:
        chunks = chunk_regulation(document)
        log(f"Chunking {source_id}: {len(chunks)} chunks")
        log(f"Embedding {len(chunks)} chunks...")
        vectors = embedder.embed([chunk.text for chunk in chunks])
        records = [
            ChunkRecord.from_chunk(chunk, vector) for chunk, vector in zip(chunks, vectors)
        ]
        log(f"Storing {len(records)} chunks...")
        store.upsert_chunks(records)
        ingested += len(records)
    return IngestReport(chunks_ingested=ingested, sources=sources)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    # One config seam: the same Settings the backend serves from supplies the
    # defaults (env vars and the root .env, in that precedence). Plain
    # Settings, not load_settings — ingestion is mode-agnostic and must not
    # trip Live-mode boot validation.
    try:
        settings = Settings()
    except ValidationError as error:
        raise IngestError(f"Invalid configuration: {error}") from error
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
        default=settings.database_url,
        help="PostgreSQL connection URL (default: DATABASE_URL or the local compose stack)",
    )
    parser.add_argument(
        "--ollama-url",
        default=settings.ollama_api_url,
        help="Ollama base URL (default: OLLAMA_API_URL or the local compose stack)",
    )
    parser.add_argument(
        "--model",
        default=settings.embedding_model,
        help=(
            f"embedding model (default: EMBEDDING_MODEL or {DEFAULT_MODEL}); "
            "must match the model the backend queries with"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64, help="texts per embedding request")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    # .env feeds the env-var defaults below (EMBEDDING_MODEL included);
    # the real environment wins either way.
    source_local_env()
    args = parse_args(argv)
    try:
        documents = load_corpus_documents(args.data_dir)
        print(f"Connecting to database...")
        store = PgVectorStore(connect(args.database_url))
        embedder: Embedder = OllamaEmbedder(
            base_url=args.ollama_url, model=args.model, batch_size=args.batch_size
        )
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
