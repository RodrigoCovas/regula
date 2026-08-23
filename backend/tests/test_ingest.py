"""Unit tests for the ingestion pipeline glue (spec #9, ticket #14).

The pipeline composes the pure transform (#11) with the two provider
protocols — a Store and an Embedder. Tests inject deterministic fakes:
no PostgreSQL, no Ollama, no network. Real-PostgreSQL behaviour is covered
separately by test_ingest_integration.py.
"""

import json
from pathlib import Path

import pytest

import src.ingest as ingest_module
from src.ingest import IngestError, load_corpus_documents, run_ingestion


class FakeEmbedder:
    """Deterministic vectors keyed by text; records every batch it sees."""

    def __init__(self):
        self.batches: list[list[str]] = []

    def embed(self, texts):
        self.batches.append(list(texts))
        return [[float(len(text)), 1.0] for text in texts]


class FakeStore:
    """Captures upserted records; mimics idempotent row accounting."""

    def __init__(self):
        self.records: list = []
        self.ensure_schema_calls = 0

    def ensure_schema(self):
        self.ensure_schema_calls += 1

    def upsert_chunks(self, records):
        self.records.extend(records)
        return len(records)


def tiny_document(source_id="tiny", articles=1):
    return {
        "metadata": {"id": source_id, "shortName": "Tiny"},
        "chapters": [
            {
                "id": f"{source_id}-c1",
                "sections": [
                    {
                        "id": f"{source_id}-s1",
                        "articles": [
                            {
                                "id": f"{source_id}-a{n}",
                                "number": n,
                                "title": f"Title {n}",
                                "paragraphs": [
                                    {"number": 1, "text": f"Article {n} says words.", "points": []}
                                ],
                            }
                            for n in range(1, articles + 1)
                        ],
                    }
                ],
            }
        ],
        "recitals": [],
        "annexes": [],
    }


# --- Document loading ---------------------------------------------------------


def test_load_corpus_documents_reads_every_json_file_sorted(tmp_path: Path):
    (tmp_path / "dora.json").write_text(json.dumps(tiny_document("dora")), encoding="utf-8")
    (tmp_path / "ai-act.json").write_text(json.dumps(tiny_document("ai-act")), encoding="utf-8")
    (tmp_path / "PROVENANCE.md").write_text("not a corpus file", encoding="utf-8")

    documents = load_corpus_documents(tmp_path)

    assert [source_id for source_id, _ in documents] == ["ai-act", "dora"]


def test_load_corpus_documents_with_no_corpus_fails_loudly(tmp_path: Path):
    with pytest.raises(IngestError) as excinfo:
        load_corpus_documents(tmp_path)
    assert str(tmp_path) in str(excinfo.value)


# --- Pipeline over the provider protocols -------------------------------------


def test_run_ingestion_ensures_schema_then_persists_every_chunk_metadata_intact():
    store = FakeStore()
    documents = [("tiny", tiny_document(articles=2))]

    report = run_ingestion(store, FakeEmbedder(), documents)

    assert store.ensure_schema_calls == 1
    assert [(r.source_id, r.kind) for r in store.records] == [
        ("tiny", "article"),
        ("tiny", "article"),
    ]
    first, second = store.records
    assert first.article_number == 1 and second.article_number == 2
    assert first.recital_number is None and first.annex_number is None
    assert first.title == "Title 1"
    assert [r.chunk_index for r in store.records] == [0, 0]
    assert report.chunks_ingested == 2
    assert report.sources == ["tiny"]


def test_run_ingestion_pairs_each_chunk_with_the_embedding_of_its_own_text():
    store = FakeStore()
    documents = [("tiny", tiny_document(articles=1))]

    run_ingestion(store, FakeEmbedder(), documents)

    (record,) = store.records
    assert record.embedding[0] == float(len(record.text))


def test_run_ingestion_sends_each_document_to_the_embedder_in_one_call():
    store = FakeStore()
    embedder = FakeEmbedder()
    documents = [("tiny", tiny_document(articles=5))]

    run_ingestion(store, embedder, documents)

    assert len(embedder.batches) == 1
    assert len(embedder.batches[0]) == 5


# --- CLI contract: failures exit nonzero with a message, never a traceback ----


def test_main_exits_nonzero_when_corpus_cannot_be_loaded(monkeypatch, capsys):
    monkeypatch.setattr(
        ingest_module,
        "load_corpus_documents",
        lambda data_dir: (_ for _ in ()).throw(IngestError("no corpus here")),
    )

    assert ingest_module.main([]) == 1
    assert "no corpus here" in capsys.readouterr().err
