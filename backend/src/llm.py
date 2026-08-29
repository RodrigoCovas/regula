"""LLM access over the OpenAI-compatible chat-completions convention (spec #9, ticket #17, ADR-0009).

``Llm`` is the second provider protocol at the composition root: tests
inject deterministic fakes, production wires ``OpenRouterClient``. The
model and base URL are environment configuration (``LLM_MODEL``,
``LLM_BASE_URL``) with the historical defaults baked in as fallbacks —
Solar Pro 4 via OpenRouter, the tested path. The client speaks plain
OpenAI chat-completions with Bearer auth and no provider-specific
headers, and every completion asks for JSON that is validated against a
Pydantic schema before it can cross a workflow boundary.
"""

from typing import Protocol, runtime_checkable, TypeVar

import json

import requests
from pydantic import BaseModel, ValidationError

# The default chat model: Solar Pro 4 via OpenRouter (spec #9, amended
# 2026-08-26 — nemotron's hybrid-reasoning style truncated answers against
# the token budget). ADR-0009: LLM_MODEL overrides it without code changes.
DEFAULT_LLM_MODEL = "upstage/solar-pro4"

# The OpenAI-compatible convention (ADR-0009): the base URL ends at the
# version root and the chat-completions path is appended by the client —
# the convention OpenAI, Groq, Together, vLLM, LM Studio, and Ollama's
# OpenAI endpoint assume.
DEFAULT_LLM_BASE_URL = "https://openrouter.ai/api/v1"
CHAT_COMPLETIONS_PATH = "/chat/completions"

# The locked single-pass budget: measured empirically — the old 2K cap
# truncated answers mid-object more often than it bounded them.
# ADR-0003: depth outranks latency, and the cap only has to stop runaway
# generations, not ration the answer.
MAX_COMPLETION_TOKENS = 8192

_TIMEOUT_SECONDS = 300

# The correction nudge appended as a user turn after a reply that failed
# to parse: the model sees its own broken JSON in the conversation and is
# asked to re-emit it as valid JSON. One repair pass, then the error is
# surfaced verbatim.
_REPAIR_USER_MESSAGE = (
    "The JSON object in your previous reply is invalid and could not be parsed. "
    "Return ONLY the corrected JSON object matching the requested shape — no "
    "prose, no code fences. Close every string with its double quote on the "
    "same line you open it; never put raw newlines inside JSON strings."
)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LlmError(RuntimeError):
    """The LLM call failed in a way the operator must see verbatim."""


class LlmUnreachableError(LlmError):
    """The configured provider could not be reached at all — a genuine network outage.

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
    last '}' is the payload regardless of decoration. ``strict=False``
    admits raw control characters inside strings — a model that wraps a
    long value across lines should cost us a repair pass, not a failed
    workflow.
    """
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in content")
    parsed = json.loads(content[start : end + 1], strict=False)
    if not isinstance(parsed, dict):
        raise ValueError("content JSON is not an object")
    return parsed


def _parse_reply(schema: type[SchemaT], content: str, model: str) -> SchemaT:
    """Extract and validate the JSON payload of one completion reply.

    Parse failures propagate as ``ValueError``/``JSONDecodeError`` so the
    client can attempt a repair pass; a well-formed but wrongly-shaped
    reply is reported as ``LlmError`` immediately — no repair, since the
    model already produced JSON and disagreed on shape instead.
    """
    data = _extract_json(content)
    try:
        return schema.model_validate(data)
    except ValidationError as error:
        raise LlmError(
            f"Model '{model}' returned JSON that does not match schema "
            f"{schema.__name__}: {content!r} ({error})"
        ) from error


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

    def render_ref(ref: str) -> str:
        referenced = defs[ref.rsplit("/", 1)[-1]]
        if "enum" in referenced:
            values = " | ".join(json.dumps(value) for value in referenced["enum"])
            return f"<{values}>"
        return render_object(referenced)

    def render_prop(prop: dict) -> str:
        if "$ref" in prop:
            return render_ref(prop["$ref"])
        if prop.get("type") == "array":
            items = prop.get("items", {})
            if "$ref" in items:
                return "[" + render_ref(items["$ref"]) + "]"
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
    """Posts chat completions to an OpenAI-compatible provider and validates
    the JSON reply.

    The model and base URL are constructor parameters wired from the
    environment (ADR-0009), defaulting to the historical Solar Pro 4 /
    OpenRouter pin — nothing can silently drift from the configured
    provider, and tests can point the client anywhere. The completion
    budget stays a module-level lock.

    Every completed call that carries provider usage leaves its dictionary
    in ``usage``, so per-request observability can sum what the workflow
    spent — including replies that later fail to parse, which still cost
    tokens.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_LLM_MODEL,
        base_url: str = DEFAULT_LLM_BASE_URL,
        transport=None,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._transport = transport or _requests_transport
        self.usage: list[dict] = []

    def complete(self, system: str, user: str, schema: type[SchemaT]) -> SchemaT:
        instruction = (
            "Respond with ONLY a single JSON object of exactly this shape — "
            "no prose, no code fences: "
            f"{_schema_example(schema)}. Close every string with its double "
            "quote on the same line you open it; never put raw newlines "
            "inside JSON strings."
        )
        messages = [
            {"role": "system", "content": f"{system}\n\n{instruction}"},
            {"role": "user", "content": user},
        ]
        content = self._chat(messages)
        try:
            return _parse_reply(schema, content, self._model)
        except (ValueError, json.JSONDecodeError):
            pass
        # One repair pass: replay the broken reply as the assistant's turn
        # and ask for the corrected JSON, so a transient formatting slip
        # does not fail the workflow (live solar-pro4 emits unclosed
        # strings wrapped across lines).
        repaired = self._chat(
            messages
            + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": _REPAIR_USER_MESSAGE},
            ]
        )
        try:
            return _parse_reply(schema, repaired, self._model)
        except (ValueError, json.JSONDecodeError) as error:
            raise LlmError(
                f"Model '{self._model}' did not return parseable JSON for schema "
                f"{schema.__name__} ({error}): {repaired!r}"
            ) from error

    def _chat(self, messages: list[dict]) -> str:
        """One completion over the given conversation, returning the reply text."""
        payload = {
            "model": self._model,
            "max_tokens": MAX_COMPLETION_TOKENS,
            "messages": messages,
        }
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        url = f"{self._base_url}{CHAT_COMPLETIONS_PATH}"
        try:
            body = self._transport(url, headers, payload)
        except requests.ConnectionError as error:
            raise LlmUnreachableError(
                f"Could not reach the LLM provider at {url}: {error}. "
                "Check the network connection."
            ) from error
        except requests.HTTPError as error:
            detail = error.response.text.strip()[:300] if error.response is not None else ""
            status = error.response.status_code if error.response is not None else "?"
            raise LlmError(f"The LLM provider rejected the request (HTTP {status}): {detail}") from error
        # Usage lands before any parsing: a reply we cannot parse still spent
        # tokens the operator pays for.
        usage = body.get("usage") if isinstance(body, dict) else None
        if isinstance(usage, dict):
            self.usage.append(usage)
        try:
            choice = body["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise LlmError(f"The LLM provider returned an unexpected response body: {body!r}") from error
        if choice.get("finish_reason") == "length":
            raise LlmError(
                f"Model '{self._model}' hit the {MAX_COMPLETION_TOKENS}-token completion "
                f"budget before finishing (finish_reason=length); hidden reasoning tokens "
                f"count against the cap. Reply so far: {content[:200]!r}"
            )
        return content
