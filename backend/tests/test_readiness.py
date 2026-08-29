"""Readiness endpoint tests (issue #45): Live-mode Readiness as three booleans.

GET /readiness reports {api_key_set, embedding_model_present, corpus_ingested}
derived from real state — the configured settings, the Ollama tags probe, and
the vector store. Tests fake each prerequisite at the established seams: env
vars booted through the real lifespan for settings, monkeypatched probes for
the store and the embedding-model check. The liveness endpoint's unchanged
shape is pinned beside them, as the third acceptance criterion pairs them.
"""

import sys
from pathlib import Path

# Ensure backend.src is importable when running tests from repo root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import psycopg2
import pytest
from fastapi.testclient import TestClient

from src import availability, readiness
from src.main import app

READINESS_KEYS = {"api_key_set", "embedding_model_present", "corpus_ingested"}


@pytest.fixture(autouse=True)
def _hermetic_embedding_probe(monkeypatch):
    """The readiness check never reaches for a real Ollama unless a test
    replaces the probe explicitly — mirroring _hermetic_store_probe."""
    monkeypatch.setattr(
        readiness, "embedding_model_available", lambda base_url, model: True
    )
    yield


def boot(monkeypatch, raise_server_exceptions=True, **env):
    """Start the app through its real lifecycle with the given environment."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


def patch_embedding_probe(monkeypatch, fake):
    """Install ``fake`` as the embedding-model probe the readiness check calls."""
    monkeypatch.setattr(readiness, "embedding_model_available", fake)


def patch_stored_chunk_count(monkeypatch, fake):
    """Install ``fake`` as the vector-store probe for the next probes."""
    monkeypatch.setattr(availability, "stored_chunk_count", fake)


def test_ready_state_reports_all_three_prerequisites_present(monkeypatch):
    """Key configured, embedding model available, Corpus ingested: the ready
    case reads true across the board — and nothing else ships in the payload."""
    patch_stored_chunk_count(monkeypatch, lambda _database_url: 42)
    with boot(monkeypatch, OPENROUTER_API_KEY="sk-or-test") as client:
        resp = client.get("/readiness")
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == READINESS_KEYS
    assert data == {
        "api_key_set": True,
        "embedding_model_present": True,
        "corpus_ingested": True,
    }


def test_missing_provider_key_reports_api_key_unset_alone(monkeypatch):
    """Without OPENROUTER_API_KEY, only the key boolean flips: the other
    prerequisites are probed independently, so the checklist can name the
    one gap instead of withholding everything."""
    patch_stored_chunk_count(monkeypatch, lambda _database_url: 42)
    with boot(monkeypatch) as client:
        data = client.get("/readiness").json()
    assert data["api_key_set"] is False
    assert data["embedding_model_present"] is True
    assert data["corpus_ingested"] is True


def test_missing_embedding_model_reports_it_individually(monkeypatch):
    """An unavailable embedding model (Ollama down or model not pulled) shows
    up as exactly that boolean — never as a server error or a blanket false."""
    patch_stored_chunk_count(monkeypatch, lambda _database_url: 42)
    patch_embedding_probe(monkeypatch, lambda base_url, model: False)
    with boot(monkeypatch, OPENROUTER_API_KEY="sk-or-test") as client:
        data = client.get("/readiness").json()
    assert data["embedding_model_present"] is False
    assert data["api_key_set"] is True
    assert data["corpus_ingested"] is True


def test_empty_store_reports_corpus_not_ingested_individually(monkeypatch):
    """An ingested Corpus is its own boolean: an empty store fails it while
    key and embedding model still read true."""
    patch_stored_chunk_count(monkeypatch, lambda _database_url: 0)
    with boot(monkeypatch, OPENROUTER_API_KEY="sk-or-test") as client:
        data = client.get("/readiness").json()
    assert data["corpus_ingested"] is False
    assert data["api_key_set"] is True
    assert data["embedding_model_present"] is True


def test_unreachable_store_is_not_ready_but_never_a_server_error(monkeypatch):
    """A connection outage cannot confirm a non-empty store, so the corpus
    boolean reads false — the conservative answer — while the endpoint still
    serves 200 for tooling to poll."""
    def unreachable(_database_url):
        raise ConnectionError("connection refused")

    patch_stored_chunk_count(monkeypatch, unreachable)
    with boot(monkeypatch, OPENROUTER_API_KEY="sk-or-test") as client:
        resp = client.get("/readiness")
    assert resp.status_code == 200
    assert resp.json()["corpus_ingested"] is False


def test_store_whose_schema_never_existed_yet_reads_not_ingested(monkeypatch):
    """A fresh stack's Postgres holds no chunks table until the first ingest
    creates it, so a missing-table error is the un-ingested state as the
    poller sees it — not-ingested (whose fix, ingest, creates the schema),
    never a server error. The per-request gates keep the stricter taxonomy."""
    def missing_table(_database_url):
        raise psycopg2.ProgrammingError('relation "chunks" does not exist')

    patch_stored_chunk_count(monkeypatch, missing_table)
    with boot(monkeypatch, OPENROUTER_API_KEY="sk-or-test") as client:
        resp = client.get("/readiness")
    assert resp.status_code == 200
    assert resp.json()["corpus_ingested"] is False


def test_readiness_reflects_a_key_configured_after_boot(monkeypatch):
    """The endpoint reads the live settings each call: a key set in the
    environment for a restart flips api_key_set on the next boot."""
    patch_stored_chunk_count(monkeypatch, lambda _database_url: 42)
    with boot(monkeypatch) as client:
        assert client.get("/readiness").json()["api_key_set"] is False
    with boot(monkeypatch, OPENROUTER_API_KEY="sk-or-test") as client:
        assert client.get("/readiness").json()["api_key_set"] is True


def test_liveness_endpoint_keeps_its_shape_and_semantics(monkeypatch):
    """The liveness endpoint is unchanged: process-up only, never a Readiness
    verdict — the two stay sibling but distinct."""
    with boot(monkeypatch) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
