"""Readiness: the state that lets a run execute in Live mode (CONTEXT.md).

Three independent booleans — a configured provider key, the embedding model
available on Ollama, and a non-empty ingested Corpus — probed from real state
on every call, so tooling and the frontend can check Live-mode Readiness
without probing internals. A probe that cannot answer reports not-ready for
its own component only: Readiness false is the conservative answer. A store
that cannot confirm it holds Chunks — unreachable, or a schema that has
never been created — reads as not-ingested, for which running the documented
ingest is the actual fix; the per-request availability gates keep the
outage/broken-schema distinction this endpoint deliberately does not carry,
and remain the backstop for races between a check and a run.
"""

from . import availability
from .config import Settings
from .embedder import embedding_model_available
from .models import Readiness


def live_readiness(settings: Settings) -> Readiness:
    """Probe the three Live-mode prerequisites right now.

    The store probe reuses the availability module's short-lived connection,
    so an ingestion that lands after boot is visible on the next call. Any
    store failure reads as not-ingested: a Readiness check reports, it never
    errors — pollers need no special handling for a stack still coming up.
    """
    try:
        corpus_ingested = not availability.vector_store_is_empty(settings.database_url)
    except Exception:
        corpus_ingested = False
    return Readiness(
        api_key_set=bool(settings.openrouter_api_key),
        embedding_model_present=embedding_model_available(
            settings.ollama_api_url, settings.embedding_model
        ),
        corpus_ingested=corpus_ingested,
    )
