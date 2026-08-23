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

# The locked embedding model is nomic-embed-text (v1.5); the default points at
# its Hugging Face GGUF build because that pulls where registry.ollama.ai is
# unreachable. Any Ollama tag with 768-dim output works via --model/--model flag.
DEFAULT_MODEL = "hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:F16"

_EMBED_TIMEOUT_SECONDS = 120


class EmbeddingError(RuntimeError):
    """Embedding failed in a way the operator must fix before ingesting."""


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def _requests_transport(url: str, payload: dict) -> dict:
    response = requests.post(url, json=payload, timeout=_EMBED_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


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
