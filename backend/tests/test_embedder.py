"""Unit tests for the Ollama embedder (spec #9, ticket #14).

The embedder is a provider protocol at the composition root, so tests inject
a fake transport — no network, no Ollama in the unit suite.
"""

import pytest
import requests

from src.embedder import (
    DEFAULT_MODEL,
    EMBEDDING_DIMENSION,
    EmbeddingError,
    OllamaEmbedder,
    embedding_model_available,
)


class FakeTransport:
    """Records requests; replays canned responses keyed by call order."""

    def __init__(self, responses=None, error=None):
        self.calls: list[tuple[str, dict]] = []
        self.responses = responses or []
        self.error = error

    def __call__(self, url: str, payload: dict) -> dict:
        if self.error is not None:
            raise self.error
        self.calls.append((url, payload))
        return self.responses[len(self.calls) - 1]


def vectors(*lengths):
    return {"embeddings": [[0.1] * length for length in lengths]}


def test_embed_posts_model_and_texts_to_ollama_embed_endpoint():
    transport = FakeTransport(responses=[vectors(EMBEDDING_DIMENSION)])
    embedder = OllamaEmbedder(base_url="http://ollama:11434", transport=transport)

    result = embedder.embed(["Article 1 body"])

    assert result == [[0.1] * EMBEDDING_DIMENSION]
    (url, payload), = transport.calls
    assert url == "http://ollama:11434/api/embed"
    assert payload == {"model": DEFAULT_MODEL, "input": ["Article 1 body"]}


def test_embed_batches_long_inputs_into_several_requests():
    transport = FakeTransport(
        responses=[vectors(768, 768), vectors(768)],
    )
    embedder = OllamaEmbedder(base_url="http://x", batch_size=2, transport=transport)

    result = embedder.embed(["a", "b", "c"])

    assert len(result) == 3
    assert [len(v) for v in result] == [768, 768, 768]
    assert [payload["input"] for _, payload in transport.calls] == [["a", "b"], ["c"]]


def test_embed_is_empty_safe():
    transport = FakeTransport()
    embedder = OllamaEmbedder(base_url="http://x", transport=transport)

    assert embedder.embed([]) == []
    assert transport.calls == []


def test_unreachable_ollama_names_the_fix_in_the_error():
    transport = FakeTransport(error=requests.ConnectionError("connection refused"))
    embedder = OllamaEmbedder(base_url="http://localhost:11434", transport=transport)

    with pytest.raises(EmbeddingError) as excinfo:
        embedder.embed(["text"])
    message = str(excinfo.value)
    assert DEFAULT_MODEL in message
    assert "docker compose up -d ollama" in message


class FakeResponse:
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text


def test_missing_model_error_carries_ollamas_message_and_pull_hint():
    transport = FakeTransport(
        error=requests.HTTPError(
            "404",
            response=FakeResponse(404, f'{{"error":"model \\"{DEFAULT_MODEL}\\" not found"}}'),
        )
    )
    embedder = OllamaEmbedder(base_url="http://x", transport=transport)

    with pytest.raises(EmbeddingError) as excinfo:
        embedder.embed(["text"])
    message = str(excinfo.value)
    assert "not found" in message
    assert f"ollama pull {DEFAULT_MODEL}" in message


def test_wrong_dimension_vector_fails_loudly():
    transport = FakeTransport(responses=[vectors(3)])
    embedder = OllamaEmbedder(base_url="http://x", transport=transport)

    with pytest.raises(EmbeddingError) as excinfo:
        embedder.embed(["text"])
    assert "dimension" in str(excinfo.value).lower()


def test_model_can_be_overridden_for_tests():
    transport = FakeTransport(responses=[vectors(5)])
    embedder = OllamaEmbedder(
        base_url="http://x", model="fake-embed", expected_dimension=5, transport=transport
    )

    result = embedder.embed(["t"])

    assert result == [[0.1] * 5]
    assert transport.calls[0][1]["model"] == "fake-embed"


# --- Embedding-model availability probe: the Readiness boolean's source (issue #45) ---


class FakeTagsTransport:
    """Records GET urls; replays one canned /api/tags body."""

    def __init__(self, body=None, error=None):
        self.calls: list[str] = []
        self.body = body if body is not None else {"models": []}
        self.error = error

    def __call__(self, url: str) -> dict:
        if self.error is not None:
            raise self.error
        self.calls.append(url)
        return self.body


def tags(*names):
    return {"models": [{"name": name, "model": name} for name in names]}


def test_probe_hits_the_tags_endpoint_and_finds_the_configured_model():
    transport = FakeTagsTransport(body=tags("some-chat-model", DEFAULT_MODEL))

    assert embedding_model_available("http://ollama:11434", transport=transport) is True
    assert transport.calls == ["http://ollama:11434/api/tags"]


def test_probe_overrides_the_default_model():
    transport = FakeTagsTransport(body=tags("my-embed"))

    assert embedding_model_available("http://x", model="my-embed", transport=transport) is True


def test_model_absent_from_the_listing_reports_not_present():
    transport = FakeTagsTransport(body=tags("some-chat-model"))

    assert embedding_model_available("http://x", transport=transport) is False


@pytest.mark.parametrize(
    "error",
    [
        requests.ConnectionError("connection refused"),
        requests.HTTPError("500"),
        ValueError("not JSON"),
    ],
    ids=["unreachable", "rejected", "malformed-json"],
)
def test_probe_that_cannot_answer_reports_not_present(error):
    """Unreachable Ollama, a rejected request, or an unreadable body all mean
    the model cannot be confirmed present — one conservative False."""
    transport = FakeTagsTransport(error=error)

    assert embedding_model_available("http://localhost:11434", transport=transport) is False


def test_probe_accepts_a_bare_configured_name_resolved_to_latest():
    """Ollama resolves a tag-less name to :latest (that is what a plain
    `ollama pull nomic-embed-text` produces), so the probe must too —
    otherwise Readiness reports false for a model embed requests would find."""
    transport = FakeTagsTransport(body=tags("nomic-embed-text:latest"))

    assert (
        embedding_model_available("http://x", model="nomic-embed-text", transport=transport)
        is True
    )


def test_tagged_configured_name_does_not_match_a_bare_listing():
    """The resolution only goes one way: an explicitly tagged model is not
    satisfied by a bare listing of the same name."""
    transport = FakeTagsTransport(body=tags("nomic-embed-text"))

    assert (
        embedding_model_available(
            "http://x", model="nomic-embed-text:latest", transport=transport
        )
        is False
    )


@pytest.mark.parametrize(
    "body",
    [
        [],
        {"models": None},
        {"models": ["not-a-dict"]},
        {"unexpected": True},
    ],
    ids=["list-body", "null-models", "non-dict-entries", "missing-models"],
)
def test_malformed_tags_body_reports_not_present(body):
    transport = FakeTagsTransport(body=body)

    assert embedding_model_available("http://x", transport=transport) is False
