"""Configuration seam: REGULA_MODE is the server default mode; provider settings are env-driven.

Per ADR-0008, mode is a per-run choice carried on each analysis request and
REGULA_MODE survives only as the server-side default when a request omits it
— so the backend boots without a Live-capable configuration: Live mode
without OPENROUTER_API_KEY still loads, and the missing key surfaces as a
per-request Readiness gap. Per ADR-0009, the provider configuration (LLM
model, base URL, API key, embedding model) is environment-driven with
defaults that reproduce the original pin (Solar Pro 4 over OpenRouter, nomic
embedder).
"""

import os

import pytest
from pathlib import Path

from src.config import ConfigurationError, load_settings
from src.models import Mode


def test_default_mode_is_demo(monkeypatch):
    monkeypatch.delenv("REGULA_MODE", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    settings = load_settings()
    assert settings.regula_mode == Mode.demo


def test_live_mode_selected_via_env_var(monkeypatch):
    monkeypatch.setenv("REGULA_MODE", "live")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    settings = load_settings()
    assert settings.regula_mode == Mode.live
    assert settings.openrouter_api_key is not None


def test_live_mode_without_api_key_still_loads_key_gap_surfaces_per_request(monkeypatch):
    """ADR-0008: the backend boots without a Live-capable configuration — a
    missing provider key is a per-request Readiness gap (the Not-available
    response), not a boot refusal."""
    monkeypatch.setenv("REGULA_MODE", "live")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    settings = load_settings()
    assert settings.regula_mode == Mode.live
    assert settings.openrouter_api_key is None


def test_empty_api_key_env_value_reads_as_not_configured(monkeypatch):
    """An empty-string key reads as no key (the truthiness contract every
    consumer — Readiness, the Live gates, the client builder — relies on).

    Docker Compose forwards the provider key by always setting the variable
    in the container, so an env file that omits it delivers OPENROUTER_API_KEY=''
    — which must read as unset, not as a configured-but-blank key."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    settings = load_settings()
    assert not settings.openrouter_api_key


def test_invalid_mode_value_raises_configuration_error_not_silent_demo(monkeypatch):
    monkeypatch.setenv("REGULA_MODE", "banana")
    with pytest.raises(ConfigurationError) as excinfo:
        load_settings()
    assert "REGULA_MODE" in str(excinfo.value)


def test_database_url_defaults_to_the_local_stack_dsn(monkeypatch):
    """The un-ingested guard probes pgvector at the same default DSN the ingest CLI writes to."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    settings = load_settings()
    assert settings.database_url == "postgresql://regula:regula@localhost:5432/regula"


def test_database_url_selected_via_env_var(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://regula:regula@postgres:5432/regula")
    settings = load_settings()
    assert settings.database_url == "postgresql://regula:regula@postgres:5432/regula"


def test_ollama_api_url_defaults_to_the_local_stack_url(monkeypatch):
    """Live query embeddings go to the same default Ollama the ingest CLI uses."""
    monkeypatch.delenv("OLLAMA_API_URL", raising=False)
    settings = load_settings()
    assert settings.ollama_api_url == "http://localhost:11434"


def test_ollama_api_url_selected_via_env_var(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_URL", "http://ollama:11434")
    settings = load_settings()
    assert settings.ollama_api_url == "http://ollama:11434"


def test_query_log_defaults_to_the_documented_jsonl_convention(monkeypatch):
    """Per-request observability lands at the documented logs/queries.jsonl
    unless overridden (ticket #19)."""
    monkeypatch.delenv("QUERY_LOG_PATH", raising=False)
    settings = load_settings()
    assert settings.query_log_path == "logs/queries.jsonl"


def test_query_log_path_selected_via_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("QUERY_LOG_PATH", str(tmp_path / "queries.jsonl"))
    settings = load_settings()
    assert settings.query_log_path == str(tmp_path / "queries.jsonl")


# --- Provider configuration is environment-driven (ADR-0009, issue #43) -------


def test_llm_model_defaults_to_solar_pro4(monkeypatch):
    """Omitting LLM_MODEL reproduces today's behaviour: Solar Pro 4."""
    monkeypatch.delenv("LLM_MODEL", raising=False)
    settings = load_settings()
    assert settings.llm_model == "upstage/solar-pro4"


def test_llm_model_selected_via_env_var(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct")
    settings = load_settings()
    assert settings.llm_model == "meta-llama/llama-3.3-70b-instruct"


def test_llm_base_url_defaults_to_the_openrouter_version_root(monkeypatch):
    """The OpenAI-compatible convention: the base ends at the version root —
    no chat-completions path baked into the setting."""
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    settings = load_settings()
    assert settings.llm_base_url == "https://openrouter.ai/api/v1"
    assert not settings.llm_base_url.rstrip("/").endswith("/chat/completions")


def test_llm_base_url_convention_reproduces_todays_endpoint_when_joined(monkeypatch):
    """The client appends /chat/completions to the base, so the default
    settings join back to exactly the URL the pinned client posted to."""
    from src.llm import CHAT_COMPLETIONS_PATH

    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    settings = load_settings()
    assert (
        settings.llm_base_url.rstrip("/") + CHAT_COMPLETIONS_PATH
        == "https://openrouter.ai/api/v1/chat/completions"
    )


def test_llm_base_url_selected_via_env_var(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    settings = load_settings()
    assert settings.llm_base_url == "https://api.groq.com/openai/v1"


def test_embedding_model_defaults_to_nomic(monkeypatch):
    """Omitting EMBEDDING_MODEL reproduces today's behaviour: the nomic
    embedder (the same default the ingest CLI has always used)."""
    from src.embedder import DEFAULT_MODEL

    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    settings = load_settings()
    assert settings.embedding_model == DEFAULT_MODEL


def test_embedding_model_selected_via_env_var(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "nomic-embed-text")
    settings = load_settings()
    assert settings.embedding_model == "nomic-embed-text"


# --- Compose forwards the provider configuration (issue #49) --------------------


def test_compose_defaults_mirror_the_backend_builtins():
    """docker-compose.yml substitutes the provider variables with `:-`
    defaults for a first-time cloner who sets none of them — those literals
    must equal the backend's built-in defaults so the two can never drift.
    DATABASE_URL and OLLAMA_API_URL are deliberately absent: inside the stack
    they stay pinned to the compose service names."""
    from src.embedder import DEFAULT_MODEL as EMBEDDING_DEFAULT
    from src.llm import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL

    compose = (Path(__file__).resolve().parents[2] / "docker-compose.yml").read_text()
    assert f"OPENROUTER_API_KEY: ${{OPENROUTER_API_KEY:-}}" in compose
    assert f"LLM_MODEL: ${{LLM_MODEL:-{DEFAULT_LLM_MODEL}}}" in compose
    assert f"LLM_BASE_URL: ${{LLM_BASE_URL:-{DEFAULT_LLM_BASE_URL}}}" in compose
    assert f"EMBEDDING_MODEL: ${{EMBEDDING_MODEL:-{EMBEDDING_DEFAULT}}}" in compose
    assert "REGULA_MODE: ${REGULA_MODE:-demo}" in compose


# --- .env.local loading (the documented configuration home) -------------------


def test_load_settings_reads_backend_env_local_when_present(monkeypatch, tmp_path):
    """backend/.env.local is the documented configuration home (.env.example,
    README): values there reach load_settings without exporting anything."""
    import src.config

    env_local = tmp_path / ".env.local"
    env_local.write_text("LLM_MODEL=from-env-local\n")
    monkeypatch.setattr(src.config, "_ENV_LOCAL_PATH", env_local)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    settings = load_settings()

    assert settings.llm_model == "from-env-local"
    # load_dotenv planted the file's values straight into os.environ — a
    # side effect monkeypatch cannot undo, because the delenv above recorded
    # the variable's absent state before the plant happened. Remove the
    # planted value explicitly so the leak cannot reach later tests (a
    # suite-order-dependent model leak once surfaced exactly here).
    os.environ.pop("LLM_MODEL", None)


def test_the_real_environment_wins_over_env_local(monkeypatch, tmp_path):
    """source_local_env never overrides an exported variable: the process
    environment stays the stronger contract."""
    import src.config

    env_local = tmp_path / ".env.local"
    env_local.write_text("LLM_MODEL=from-env-local\n")
    monkeypatch.setattr(src.config, "_ENV_LOCAL_PATH", env_local)
    monkeypatch.setenv("LLM_MODEL", "from-export")

    settings = load_settings()

    assert settings.llm_model == "from-export"


def test_the_session_never_points_at_the_real_env_local():
    """conftest neutralises _ENV_LOCAL_PATH at collection time — before any
    test module imports src.main and its import-time load_settings() runs —
    so the host's real backend/.env.local can never leak into the session."""
    import src.config

    assert not src.config._ENV_LOCAL_PATH.exists()
