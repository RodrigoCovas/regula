"""Unit tests for the Ollama embedder (spec #9, ticket #14).

The embedder is a provider protocol at the composition root, so tests inject
a fake transport — no network, no Ollama in the unit suite.
"""

import pytest
import requests

from src.embedder import DEFAULT_MODEL, EMBEDDING_DIMENSION, EmbeddingError, OllamaEmbedder


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
