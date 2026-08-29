"""Unit tests for the OpenRouter LLM client (spec #9, ticket #17).

The LLM is a provider protocol at the composition root, so tests inject
a fake transport — no network, no OpenRouter in the unit suite.
"""

import sys
from pathlib import Path

# Ensure backend.src is importable when running tests from repo root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import pytest
from pydantic import BaseModel

from src.llm import LlmError, OpenRouterClient


class Plan(BaseModel):
    targets: list[str]


class FakeTransport:
    """Records requests; replays canned responses keyed by call order."""

    def __init__(self, responses=None, error=None):
        self.calls: list[tuple[str, dict, dict]] = []
        self.responses = responses or []
        self.error = error

    def __call__(self, url: str, headers: dict, payload: dict) -> dict:
        if self.error is not None:
            raise self.error
        self.calls.append((url, headers, payload))
        return self.responses[len(self.calls) - 1]


def chat_response(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def make_client(transport) -> OpenRouterClient:
    return OpenRouterClient(api_key="sk-or-test", transport=transport)


def test_complete_returns_validated_schema_parsed_from_json_content():
    transport = FakeTransport(responses=[chat_response('{"targets": ["a", "b"]}')])
    client = make_client(transport)

    plan = client.complete(system="plan", user="question", schema=Plan)

    assert plan == Plan(targets=["a", "b"])


def test_complete_posts_the_default_model_and_auth_header_to_openrouter():
    """With no overrides the client reproduces the historical pin: Solar Pro 4
    posted to OpenRouter's version root with the chat-completions path joined."""
    transport = FakeTransport(responses=[chat_response('{"targets": []}')])
    client = make_client(transport)

    client.complete(system="s", user="u", schema=Plan)

    url, headers, payload = transport.calls[0]
    from src.llm import CHAT_COMPLETIONS_PATH, DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL

    assert url == DEFAULT_LLM_BASE_URL + CHAT_COMPLETIONS_PATH
    assert headers["Authorization"] == "Bearer sk-or-test"
    assert payload["model"] == DEFAULT_LLM_MODEL
    # The locked single-pass reasoning budget bounds every completion.
    from src.llm import MAX_COMPLETION_TOKENS

    assert payload["max_tokens"] == MAX_COMPLETION_TOKENS
    assert [m["content"] for m in payload["messages"]] == [
        "s\n\nRespond with ONLY a single JSON object of exactly this shape — "
        'no prose, no code fences: {"targets": [<string>]}. Close every '
        "string with its double quote on the same line you open it; never "
        "put raw newlines inside JSON strings.",
        "u",
    ]


def test_configured_model_and_base_url_reach_the_joined_chat_completions_url():
    """ADR-0009: the client appends /chat/completions to the configured base
    (which ends at the version root) and asks for the configured model —
    plain Bearer auth, standard OpenAI shape, any OpenAI-compatible provider."""
    transport = FakeTransport(responses=[chat_response('{"targets": []}')])
    client = OpenRouterClient(
        api_key="sk-other-provider",
        model="meta-llama/llama-3.3-70b-instruct",
        base_url="https://api.groq.com/openai/v1/",
        transport=transport,
    )

    client.complete(system="s", user="u", schema=Plan)

    url, headers, payload = transport.calls[0]
    assert url == "https://api.groq.com/openai/v1/chat/completions"
    assert headers["Authorization"] == "Bearer sk-other-provider"
    assert payload["model"] == "meta-llama/llama-3.3-70b-instruct"
    assert payload["messages"][0]["role"] == "system"


def test_nested_schemas_render_structurally_in_the_instruction():
    """The shape template shows array elements as objects, not bare values —
    a live nemotron reply returned bare strings when only 'array' was named."""
    from pydantic import BaseModel

    class Target(BaseModel):
        query: str

    class NestedPlan(BaseModel):
        targets: list[Target]

    transport = FakeTransport(responses=[chat_response('{"targets": [{"query": "x"}]}')])
    client = make_client(transport)
    client.complete(system="s", user="u", schema=NestedPlan)

    _, _, payload = transport.calls[0]
    assert '{"targets": [{"query": <string>}]}' in payload["messages"][0]["content"]


def test_verdict_strength_wrapper_from_live_model_is_normalized():
    """The live model may return strength with its rationale as an object."""
    from src.live_workflow import Verdicts
    from src.models import Strength

    content = (
        '{"verdicts": [{"statement": "Claim", "supported": true, '
        '"strength": {"level": "strong", "rationale": "Direct provision."}, '
        '"evidence_refs": ["E1"]}]}'
    )
    transport = FakeTransport(responses=[chat_response(content)])
    client = make_client(transport)

    verdicts = client.complete(system="verify", user="claims", schema=Verdicts)

    assert verdicts.verdicts[0].strength is Strength.strong


def test_enum_refs_render_as_scalar_choices_in_the_instruction():
    from src.llm import _schema_example
    from src.live_workflow import Verdicts

    shape = _schema_example(Verdicts)

    assert '"strength": {}' not in shape
    assert all(value in shape for value in ("strong", "moderate", "weak"))


def test_markdown_fences_and_prose_around_the_json_are_tolerated():
    transport = FakeTransport(
        responses=[chat_response('Here you go:\n```json\n{"targets": ["x"]}\n```')]
    )
    client = make_client(transport)

    assert client.complete(system="s", user="u", schema=Plan) == Plan(targets=["x"])


def test_length_truncation_is_reported_as_budget_exhaustion_not_a_parse_failure():
    """A reply cut off by the completion cap (finish_reason='length') has no
    closing brace, so it masquerades as unparseable JSON — the error must
    name the budget instead of sending you debugging the parser. Live nemotron
    reasoning tokens consumed ~1.5K of the old 2048 cap before any visible
    JSON (logs/queries.jsonl 2026-08-26T09:29:36)."""
    body = chat_response('{"verdicts": [{"statement": "The EU AI')
    body["choices"][0]["finish_reason"] = "length"
    transport = FakeTransport(responses=[body])
    client = make_client(transport)

    with pytest.raises(LlmError) as excinfo:
        client.complete(system="s", user="u", schema=Plan)
    assert "completion budget" in str(excinfo.value)
    assert "finish_reason=length" in str(excinfo.value)


def test_stop_completions_are_not_flagged_as_truncated():
    body = chat_response('{"targets": ["x"]}')
    body["choices"][0]["finish_reason"] = "stop"
    transport = FakeTransport(responses=[body])
    client = make_client(transport)

    assert client.complete(system="s", user="u", schema=Plan) == Plan(targets=["x"])


def test_http_error_surfaces_status_and_body_snippet_as_llm_error():
    import requests

    response = requests.Response()
    response.status_code = 429
    response._content = b'{"error": {"message": "Rate limit exceeded"}}'
    transport = FakeTransport(error=requests.HTTPError(response=response))
    client = make_client(transport)

    with pytest.raises(LlmError) as excinfo:
        client.complete(system="s", user="u", schema=Plan)
    assert "429" in str(excinfo.value)
    assert "Rate limit" in str(excinfo.value)


def test_connection_error_names_the_provider_url_and_the_fix():
    import requests

    transport = FakeTransport(error=requests.ConnectionError("connection refused"))
    client = make_client(transport)

    with pytest.raises(LlmError) as excinfo:
        client.complete(system="s", user="u", schema=Plan)
    assert "Could not reach the LLM provider at https://openrouter.ai/api/v1/chat/completions" in str(
        excinfo.value
    )


def test_unparseable_content_is_an_llm_error_not_a_crash():
    """After one repair pass, an irreparably unparseable reply is an
    LlmError naming the schema — not an exception from the JSON parser."""
    transport = FakeTransport(
        responses=[chat_response("I cannot answer that."), chat_response("Still not JSON.")]
    )
    client = make_client(transport)

    with pytest.raises(LlmError) as excinfo:
        client.complete(system="s", user="u", schema=Plan)
    assert "parseable JSON" in str(excinfo.value)
    assert "Plan" in str(excinfo.value)
    assert len(transport.calls) == 2


def test_a_broken_first_reply_is_repaired_on_a_second_completion():
    """Live solar-pro4 emits strings whose closing quote is dropped at the
    line break: the client replays the broken reply as the assistant's turn
    and accepts the corrected JSON."""
    broken = (
        '{\n  "targets": [\n'
        '    "AI Act high-risk AI systems biometric health insurance classification,\n'
        '    "AI Act fundamental rights impact assessment obligations high-risk deployers,\n'
        '    "GDPR Article 22 automated individual decision-making including profiling safeguards"\n'
        "  ]\n}"
    )
    transport = FakeTransport(
        responses=[chat_response(broken), chat_response('{"targets": ["a", "b"]}')]
    )
    client = make_client(transport)

    assert client.complete(system="s", user="u", schema=Plan) == Plan(targets=["a", "b"])

    assert len(transport.calls) == 2
    _, _, payload = transport.calls[1]
    messages = payload["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[2]["content"] == broken
    assert "corrected JSON" in messages[3]["content"]


def test_control_characters_inside_strings_are_tolerated():
    """A string value wrapping across lines is accepted as JSON rather than
    triggering a repair pass for a mere control character."""
    transport = FakeTransport(responses=[chat_response('{"targets": ["a\nb"]}')])
    client = make_client(transport)

    assert client.complete(system="s", user="u", schema=Plan) == Plan(targets=["a\nb"])
    assert len(transport.calls) == 1


def test_schema_violation_is_an_llm_error_with_the_offending_content():
    transport = FakeTransport(responses=[chat_response('{"wrong_key": 1}')])
    client = make_client(transport)

    with pytest.raises(LlmError):
        client.complete(system="s", user="u", schema=Plan)


def test_schema_violation_is_reported_as_a_shape_mismatch_not_a_parse_failure():
    """ValidationError subclasses ValueError — without its own branch a
    well-formed but wrongly-shaped reply masquerades as unparseable."""
    transport = FakeTransport(responses=[chat_response('{"targets": {"not": "a list"}}')])
    client = make_client(transport)
    with pytest.raises(LlmError) as excinfo:
        client.complete(system="s", user="u", schema=Plan)
    assert "does not match schema" in str(excinfo.value)


# --- Per-call token usage exposure (spec #9, ticket #19) -----------------------


def usage_response(content: str) -> dict:
    body = chat_response(content)
    body["usage"] = {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}
    return body


def test_completed_completions_record_provider_token_usage():
    transport = FakeTransport(responses=[usage_response('{"targets": ["a"]}')])
    client = make_client(transport)

    client.complete(system="s", user="u", schema=Plan)

    assert client.usage == [{"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}]


def test_usage_is_recorded_even_when_the_reply_cannot_be_parsed():
    """Replies we fail to parse still spent tokens — including the repair
    pass: usage lands before any parsing so per-request cost stays
    observable."""
    transport = FakeTransport(
        responses=[usage_response("I cannot answer that."), usage_response("Still not JSON.")]
    )
    client = make_client(transport)

    with pytest.raises(LlmError):
        client.complete(system="s", user="u", schema=Plan)

    assert len(client.usage) == 2
    assert client.usage[0]["total_tokens"] == 14


def test_responses_without_usage_record_nothing():
    transport = FakeTransport(responses=[chat_response('{"targets": []}')])
    client = make_client(transport)

    client.complete(system="s", user="u", schema=Plan)

    assert client.usage == []
