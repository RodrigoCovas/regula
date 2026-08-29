import sys
import time
from pathlib import Path
from tempfile import mkdtemp

import pytest

# Ensure backend.src is importable when running tests from repo root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

# Neutralise backend/.env.local for the whole session — and it MUST happen
# here, at conftest import time: test modules import src.main, whose
# module-level load_settings() then runs before any fixture could intervene
# and would bake the host's real .env.local (the LLM API key lives there)
# into os.environ for every test. Tests that exercise the loading point it
# at their own tmp file via monkeypatch instead (see test_config.py).
import src.config as _config

_config._ENV_LOCAL_PATH = Path(mkdtemp()) / ".env.local"

from src.models import Chunk


@pytest.fixture()
def query_log_path(tmp_path) -> Path:
    """Where the hermetic query log lands for one test.

    Nested one directory deep so every test also proves the writer creates
    missing parent directories.
    """
    return tmp_path / "queries" / "queries.jsonl"


@pytest.fixture(autouse=True)
def _hermetic_query_log(monkeypatch, query_log_path):
    """Query logs stay out of the working tree: tests write them into
    tmp_path unless they say otherwise.

    Covers both boots that re-load settings through the lifespan (env var)
    and module-level clients that reuse the import-time settings object.
    """
    monkeypatch.setenv("QUERY_LOG_PATH", str(query_log_path))
    import src.main as main

    monkeypatch.setattr(main.settings, "query_log_path", str(query_log_path), raising=False)
    yield


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


def install_fake_pipeline(llm, retriever):
    """Wire deterministic fakes into the Live pipeline at the composition root."""
    from src.main import app, get_llm, get_retriever

    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_retriever] = lambda: retriever


def poll_progress(client, request_id, ready, timeout=5.0):
    """Poll GET /api/progress/{request_id} until ``ready(data)`` holds, then
    return the last data seen — the shared shape behind every "the background
    thread reached its terminal state" assertion."""
    end = time.monotonic() + timeout
    data: dict = {}
    while time.monotonic() < end:
        data = client.get(f"/api/progress/{request_id}").json()
        if ready(data):
            break
        time.sleep(0.05)
    return data


def boot_live_with_fakes(monkeypatch, llm, retriever, chunk_count=42, raise_server_exceptions=True):
    """The hermetic Live-mode preamble: ingested store probe + fakes at
    the composition root. Returns a TestClient to enter."""
    monkeypatch.setenv("REGULA_MODE", "live")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    import src.availability as availability

    monkeypatch.setattr(availability, "stored_chunk_count", lambda _database_url: chunk_count)
    install_fake_pipeline(llm, retriever)
    from fastapi.testclient import TestClient

    import src.main as main

    return TestClient(main.app, raise_server_exceptions=raise_server_exceptions)

