"""Configuration: the environment provides the server default mode and the
provider (ADR-0009).

Misconfiguration must fail fast at startup, never surface as a mid-request
server error. Mode itself is a per-run choice (ADR-0008): every analysis
request carries its mode explicitly, and REGULA_MODE survives only as the
server-side default for requests that omit it — so the backend boots without
a Live-capable configuration, and a missing provider key is a per-request
Readiness gap (the Not-available response), not a boot refusal. DATABASE_URL
points the un-ingested guard at the pgvector store — the store is never
required to boot. The LLM provider configuration (model, base URL, key,
embedding model) is environment-driven: deployments differ only by
configuration.
"""

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import SecretStr, ValidationError
from pydantic_settings import BaseSettings

from .embedder import DEFAULT_MODEL as DEFAULT_EMBEDDING_MODEL
from .llm import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER, OpenRouterClient
from .models import Mode
from .query_log import DEFAULT_QUERY_LOG_PATH

# The local stack DSN: where the ingest CLI writes Chunks and where the
# backend's guard reads them. One constant so the two ends can never drift.
DEFAULT_DATABASE_URL = "postgresql://regula:regula@localhost:5432/regula"

# Where query embeddings come from in Live mode — one constant shared with
# the ingest CLI so both ends of the vector path agree.
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# The documented configuration home (backend/.env.example, README): the
# gitignored file the LLM API key lives in by convention. A module constant so
# the hermetic test fixture can point it elsewhere.
_ENV_LOCAL_PATH = Path(__file__).resolve().parents[1] / ".env.local"


class ConfigurationError(RuntimeError):
    """Invalid configuration; startup must refuse to serve rather than degrade."""


class Settings(BaseSettings):
    # The server-side default mode for requests that omit mode (ADR-0008) —
    # not a boot-time pin. Demo is the default so the keyless path serves
    # with zero configuration.
    regula_mode: Mode = Mode.demo
    openrouter_api_key: Optional[SecretStr] = None
    # The chat model every completion asks for (ADR-0009): env-overridable,
    # defaulting to the Solar Pro 4 pin so omitting the variables reproduces
    # the original behaviour exactly.
    llm_model: str = DEFAULT_LLM_MODEL
    # OpenAI-compatible base URL, ending at the version root; the client
    # appends the chat-completions path. Default: OpenRouter (the tested path).
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    # The OpenRouter provider routing preference (ADR-0009): the provider slug
    # OpenRouter tries first for every completion, keeping its usual fallbacks.
    # Empty sends no routing preference at all — for a base URL other than
    # OpenRouter, which does not know the field.
    llm_provider: str = DEFAULT_LLM_PROVIDER
    # The embedding model used at ingest and query time — one setting shared
    # with the ingest CLI so both ends of the vector path agree. Changing it
    # requires re-ingest (documented in .env.example).
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
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
    source_local_env()
    try:
        settings = Settings()
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid configuration (check REGULA_MODE): {exc}") from exc
    return settings


def chat_client(settings: Settings) -> OpenRouterClient:
    """The chat-completions client wired to the configured provider (ADR-0009).

    The one recipe both consumers share — the workflow's composition root and
    the eval's fidelity judge — so model, base URL, provider routing
    preference, and the unwrapped key can never disagree between the pipeline
    that answers and the judge that scores it.
    """
    return OpenRouterClient(
        api_key=(
            settings.openrouter_api_key.get_secret_value() if settings.openrouter_api_key else ""
        ),
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        provider=settings.llm_provider or None,
    )


def source_local_env() -> None:
    """Place backend/.env.local into the environment when present.

    Every configuration entry point goes through here: the backend boots by
    calling ``load_settings``, and the ingest CLI calls it before parsing
    arguments. Values only ever enter the process environment — they are
    never read back, echoed, or logged. The real environment wins: nothing
    here overrides an exported variable.
    """
    if _ENV_LOCAL_PATH.exists():
        load_dotenv(_ENV_LOCAL_PATH, override=False)
