"""Local embeddings via Ollama (spec #9, ticket #14).

``Embedder`` is one of the provider protocols at the composition root:
tests inject deterministic fakes, production wires ``OllamaEmbedder``.
Failures are loud and actionable — an unreachable Ollama names the exact
``ollama pull`` command that fixes it, mirroring the Not-available philosophy
elsewhere in Regula.
"""

import requests
from typing import Callable, Protocol, Sequence

from .models import EMBEDDING_DIMENSION

# The default embedding model is nomic-embed-text (v1.5); the default points at
# its Hugging Face GGUF build because that pulls where registry.ollama.ai is
# unreachable. EMBEDDING_MODEL overrides it (ADR-0009) — any model whose output
# is EMBEDDING_DIMENSION-wide works; the embedder checks the dimension on every
# reply, so a mismatching model fails loudly instead of silently poisoning the
# store.
DEFAULT_MODEL = "hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:F16"

_EMBED_TIMEOUT_SECONDS = 120

# A model-listing probe must fail fast: Readiness polling cannot hang on the
# embed path's generous batch timeout.
_TAGS_TIMEOUT_SECONDS = 5


class EmbeddingError(RuntimeError):
    """Embedding failed in a way the operator must fix before ingesting."""


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def _checked_json(response: requests.Response) -> dict:
    """The shared tail of every Ollama transport: 4xx/5xx raise, then parse."""
    response.raise_for_status()
    return response.json()


def _requests_transport(url: str, payload: dict) -> dict:
    return _checked_json(requests.post(url, json=payload, timeout=_EMBED_TIMEOUT_SECONDS))


def _requests_get_json(url: str) -> dict:
    return _checked_json(requests.get(url, timeout=_TAGS_TIMEOUT_SECONDS))


class OllamaEmbedder:
    """Batches texts against Ollama's /api/embed endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str = DEFAULT_MODEL,
        batch_size: int = 64,
        expected_dimension: int = EMBEDDING_DIMENSION,
        transport: Callable[[str, dict], dict] | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._batch_size = batch_size
        self._expected_dimension = expected_dimension
        self._transport = transport or _requests_transport

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _embed_batch(self, batch: Sequence[str]) -> list[list[float]]:
        url = f"{self._base_url}/api/embed"
        payload = {"model": self._model, "input": list(batch)}
        try:
            body = self._transport(url, payload)
        except requests.ConnectionError as error:
            raise EmbeddingError(
                f"Could not reach Ollama at {self._base_url} to embed {len(batch)} text(s) "
                f"with model '{self._model}': {error}. "
                f"Start the stack: `docker compose up -d ollama`."
            ) from error
        except requests.HTTPError as error:
            detail = ""
            if error.response is not None:
                detail = f" Ollama said: {error.response.text.strip()}"
                if error.response.status_code == 404 and "not found" in detail.lower():
                    detail += (
                        " Pull the embedding model first: "
                        f"`docker compose exec ollama ollama pull {self._model}`."
                    )
            raise EmbeddingError(
                f"Ollama rejected the embedding request for model '{self._model}' "
                f"({len(batch)} text(s)).{detail}"
            ) from error
        try:
            embeddings = body["embeddings"]
        except (KeyError, TypeError) as error:
            raise EmbeddingError(
                f"Ollama returned an unexpected response from {url}: {body!r}"
            ) from error
        for vector in embeddings:
            if len(vector) != self._expected_dimension:
                raise EmbeddingError(
                    f"Model '{self._model}' produced a {len(vector)}-dimension vector but the "
                    f"store expects {self._expected_dimension}. The embedding model must match "
                    "the one used at ingestion."
                )
        return [list(vector) for vector in embeddings]


def _model_matches(configured: str, listed: str) -> bool:
    """Whether a model Ollama lists satisfies a configured model name.

    Mirrors Ollama's own resolution: a tag-less name resolves to ``:latest``
    (what a plain ``ollama pull nomic-embed-text`` produces), so the probe
    must accept it too — otherwise Readiness would report false for a model
    embed requests would find.
    """
    if listed == configured:
        return True
    return ":" not in configured and listed == f"{configured}:latest"


def embedding_model_available(
    base_url: str,
    model: str = DEFAULT_MODEL,
    transport: Callable[[str], dict] | None = None,
) -> bool:
    """Whether Ollama is reachable and has ``model`` among its installed models.

    Probes Ollama's /api/tags listing — the same source ``ollama list`` reads
    — so a missing model is visible before any embedding request fails. The
    boolean is deliberately coarse: an unreachable Ollama, a rejected request,
    or an unreadable body all mean the model cannot be confirmed present, and
    one conservative False covers them (the embedder's own errors name which
    fix applies when a real embedding request hits the same gap).
    """
    fetch = transport or _requests_get_json
    try:
        body = fetch(f"{base_url.rstrip('/')}/api/tags")
        entries = body.get("models") or []
        return any(_model_matches(model, entry.get("name") or "") for entry in entries)
    except Exception:
        return False
