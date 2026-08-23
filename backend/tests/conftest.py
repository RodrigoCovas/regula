import sys
from pathlib import Path

import pytest

# Ensure backend.src is importable when running tests from repo root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from src.models import Chunk


@pytest.fixture()
def assert_exactly_one_provision_target():
    """Every retrieved Chunk targets exactly one provision kind — the
    invariant deterministic Citation derivation relies on."""

    def _assert(chunk: Chunk) -> None:
        targets = (chunk.article_number, chunk.recital_number, chunk.annex_number)
        assert sum(value is not None for value in targets) == 1

    return _assert


@pytest.fixture(autouse=True)
def _restore_main_settings():
    """Boot tests may re-load src.main.settings from env; restore it afterwards."""
    import src.main as main

    saved = main.settings
    yield
    main.settings = saved


@pytest.fixture(autouse=True)
def _hermetic_store_probe(monkeypatch):
    """Keep every boot hermetic: the startup emptiness probe never reaches for
    a real PostgreSQL unless a test replaces it explicitly."""
    import src.availability as availability

    monkeypatch.setattr(availability, "stored_chunk_count", lambda _database_url: 0)
    yield


@pytest.fixture(autouse=True)
def _clear_live_dependency_overrides():
    """The FastAPI app object is module-global; dependency overrides installed
    for one Live-mode test must never leak into another."""
    yield
    import src.main as main

    main.app.dependency_overrides.clear()


def install_offline_pipeline(llm, retriever):
    """Wire deterministic fakes into the Live pipeline at the composition root."""
    from src.main import app, get_llm, get_retriever

    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_retriever] = lambda: retriever


def boot_live_offline(monkeypatch, llm, retriever, chunk_count=42):
    """The full offline Live-mode preamble: ingested store probe + fakes at
    the composition root. Returns a TestClient to enter."""
    monkeypatch.setenv("REGULA_MODE", "live")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    import src.availability as availability

    monkeypatch.setattr(availability, "stored_chunk_count", lambda _database_url: chunk_count)
    install_offline_pipeline(llm, retriever)
    from fastapi.testclient import TestClient

    import src.main as main

    return TestClient(main.app)

