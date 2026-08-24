"""Configuration: one env var selects Demo mode (default) or Live mode.

Misconfiguration must fail fast at startup, never surface as a mid-request
server error. Live mode requires OPENROUTER_API_KEY and refuses to start
without it; nothing ever silently degrades. DATABASE_URL points the un-ingested
guard at the pgvector store — the store is never required to boot.
"""

from enum import Enum
from typing import Optional

from pydantic import SecretStr, ValidationError
from pydantic_settings import BaseSettings

from .query_log import DEFAULT_QUERY_LOG_PATH

# The local stack DSN: where the ingest CLI writes Chunks and where the
# backend's guard reads them. One constant so the two ends can never drift.
DEFAULT_DATABASE_URL = "postgresql://regula:regula@localhost:5432/regula"

# Where query embeddings come from in Live mode — one constant shared with
# the ingest CLI so both ends of the vector path agree.
DEFAULT_OLLAMA_URL = "http://localhost:11434"


class Mode(str, Enum):
    demo = "demo"
    live = "live"


class ConfigurationError(RuntimeError):
    """Invalid configuration; startup must refuse to serve rather than degrade."""


class Settings(BaseSettings):
    regula_mode: Mode = Mode.demo
    openrouter_api_key: Optional[SecretStr] = None
    # Where the pgvector store lives — the same DSN the ingest CLI writes to.
    # The store is never required at boot: an unreachable or empty database is
    # a warning, never a startup failure (Demo mode needs no database at all).
    database_url: str = DEFAULT_DATABASE_URL
    # Where query embeddings come from in Live mode — same default as the
    # ingest CLI's OLLAMA_API_URL so both ends of the vector path agree.
    ollama_api_url: str = DEFAULT_OLLAMA_URL
    # Where per-request observability records land — the documented JSONL
    # convention (logs/queries.jsonl), overridable for tests.
    query_log_path: str = DEFAULT_QUERY_LOG_PATH


def load_settings() -> Settings:
    try:
        settings = Settings()
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid configuration (check REGULA_MODE): {exc}") from exc
    if settings.regula_mode == Mode.live and not settings.openrouter_api_key:
        raise ConfigurationError(
            "REGULA_MODE=live requires OPENROUTER_API_KEY to be set. "
            "Set the key or start with REGULA_MODE=demo (the default)."
        )
    return settings
