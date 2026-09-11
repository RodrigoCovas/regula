import sys
import time
from pathlib import Path
from tempfile import mkdtemp

import pytest

# Ensure backend.src is importable when running tests from repo root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

# Neutralise the root .env for the whole session — and it MUST happen
# here, at conftest import time: test modules import src.main, whose
# module-level load_settings() then runs before any fixture could intervene
# and would bake the host's real .env (the LLM API key lives there)
# into os.environ for every test. Tests that exercise the loading point it
# at their own tmp file via monkeypatch instead (see test_config.py).
import src.config as _config

_config._ENV_PATH = Path(mkdtemp()) / ".env"

from src.models import Chunk

# The canonical demo Scenario's request payload, shared by every test that
# posts the canonical demo id: routing pins on the id alone (ADR-0005), the
# description rides along as the real demo button would send it.
CANONICAL_SCENARIO_ID = "bank-cloud-outage"
CANONICAL_SCENARIO_DESCRIPTION = (
    "A bank relies on an external cloud provider to host critical systems used for online banking. "
    "A major technical failure at the cloud provider makes the bank's online banking services "
    "unavailable to customers for several hours."
)


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
def _hermetic_embedding_probe(monkeypatch):
    """The readiness check's embedding probe never reaches for a real Ollama
    unless a test replaces it explicitly — the store probe's sibling."""
    import src.readiness as readiness

    monkeypatch.setattr(
        readiness, "embedding_model_available", lambda base_url, model: True
    )
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


def boot_with_env(monkeypatch, raise_server_exceptions=True, **env):
    """Start the app through its real lifecycle with the given environment.

    The preamble every lifespan-boot test shares: scrub the two state-bearing
    variables first so a test only sees the environment it asked for."""
    monkeypatch.delenv("REGULA_MODE", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    from fastapi.testclient import TestClient

    import src.main as main

    return TestClient(main.app, raise_server_exceptions=raise_server_exceptions)


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
    import src.availability as availability

    monkeypatch.setattr(availability, "stored_chunk_count", lambda _database_url: chunk_count)
    install_fake_pipeline(llm, retriever)
    return boot_with_env(
        monkeypatch,
        raise_server_exceptions=raise_server_exceptions,
        REGULA_MODE="live",
        OPENROUTER_API_KEY="sk-or-test",
    )

