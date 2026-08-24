"""LLM access via OpenRouter (spec #9, ticket #17).

``Llm`` is the second provider protocol at the composition root: tests
inject deterministic fakes, production wires ``OpenRouterClient``. The
model is pinned to the free Nemotron tier — no env override onto a paid
model — and every completion asks for JSON that is validated against a
Pydantic schema before it can cross a workflow boundary.
"""

from typing import Protocol, runtime_checkable, TypeVar

import json

import requests
from pydantic import BaseModel, ValidationError

# The locked model: free tier via OpenRouter only (spec #9).
PINNED_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

# The locked single-pass budget (~8 Chunks, ~2K reasoning): the completion
# cap bounds reasoning + answer tokens together, so one workflow pass can
# never spend more than the budget on the model side.
MAX_COMPLETION_TOKENS = 2048

_TIMEOUT_SECONDS = 120

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LlmError(RuntimeError):
    """The LLM call failed in a way the operator must see verbatim."""


class LlmUnreachableError(LlmError):
    """OpenRouter could not be reached at all — a genuine network outage.

    A reachable provider that rejected or malformed the answer is a plain
    ``LlmError``; only this subclass counts as the provider being down, so
    callers can offer recovery advice without masking real bugs.
    """


def _requests_transport(url: str, headers: dict, payload: dict) -> dict:
    response = requests.post(url, headers=headers, json=payload, timeout=_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def _extract_json(content: str) -> dict:
    """Parse the JSON object out of a chat completion's content.

    Models wrap answers in markdown fences or prose; the first '{' to the
    last '}' is the payload regardless of decoration.
    """
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in content")
    parsed = json.loads(content[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("content JSON is not an object")
    return parsed


def _schema_example(schema: type[BaseModel]) -> str:
    """A compact JSON-shape template derived from the schema itself.

    Nested models render structurally — '{"targets": [{"query": <string>}]}'
    — so the model knows not just the top-level keys but what each array
    element must look like.
    """
    defs: dict = {}

    def render_object(model_schema: dict) -> str:
        fields = [
            f'"{name}": {render_prop(prop)}'
            for name, prop in model_schema.get("properties", {}).items()
        ]
        return "{" + ", ".join(fields) + "}"

    def render_prop(prop: dict) -> str:
        if "$ref" in prop:
            return render_object(defs[prop["$ref"].rsplit("/", 1)[-1]])
        if prop.get("type") == "array":
            items = prop.get("items", {})
            if "$ref" in items:
                return "[" + render_object(defs[items["$ref"].rsplit("/", 1)[-1]]) + "]"
            return f'[<{items.get("type", "value")}>]'
        return f'<{prop.get("type", "value")}>'

    full = schema.model_json_schema()
    defs.update(full.get("$defs", {}))
    return render_object(full)


@runtime_checkable
class Llm(Protocol):
    """Structured completion: prompt in, validated Pydantic model out."""

    def complete(self, system: str, user: str, schema: type[SchemaT]) -> SchemaT: ...


class OpenRouterClient:
    """Posts chat completions to OpenRouter and validates the JSON reply.

    The model and the completion budget are module-level locks, not
    parameters — nothing can point the client at a paid model or widen
    the single-pass budget by construction.
    """

    def __init__(self, api_key: str, transport=None):
        self._api_key = api_key
        self._transport = transport or _requests_transport

    def complete(self, system: str, user: str, schema: type[SchemaT]) -> SchemaT:
        instruction = (
            "Respond with ONLY a single JSON object of exactly this shape — "
            f"no prose, no code fences: {_schema_example(schema)}"
        )
        payload = {
            "model": PINNED_MODEL,
            "max_tokens": MAX_COMPLETION_TOKENS,
            "messages": [
                {"role": "system", "content": f"{system}\n\n{instruction}"},
                {"role": "user", "content": user},
            ],
        }
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        try:
            body = self._transport(OPENROUTER_CHAT_URL, headers, payload)
        except requests.ConnectionError as error:
            raise LlmUnreachableError(
                f"Could not reach OpenRouter at {OPENROUTER_CHAT_URL}: {error}. "
                "Check the network connection."
            ) from error
        except requests.HTTPError as error:
            detail = error.response.text.strip()[:300] if error.response is not None else ""
            status = error.response.status_code if error.response is not None else "?"
            raise LlmError(f"OpenRouter rejected the request (HTTP {status}): {detail}") from error
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise LlmError(f"OpenRouter returned an unexpected response body: {body!r}") from error
        try:
            data = _extract_json(content)
        except (ValueError, json.JSONDecodeError) as error:
            raise LlmError(
                f"Model '{PINNED_MODEL}' did not return parseable JSON for schema "
                f"{schema.__name__} ({error}): {content!r}"
            ) from error
        try:
            return schema.model_validate(data)
        except ValidationError as error:
            raise LlmError(
                f"Model '{PINNED_MODEL}' returned JSON that does not match schema "
                f"{schema.__name__}: {content!r} ({error})"
            ) from error
