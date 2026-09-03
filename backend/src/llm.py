"""LLM access over the OpenAI-compatible chat-completions convention (spec #9, ticket #17, ADR-0009).

``Llm`` is the second provider protocol at the composition root: tests
inject deterministic fakes, production wires ``OpenRouterClient``. The
model and base URL are environment configuration (``LLM_MODEL``,
``LLM_BASE_URL``) with the historical defaults baked in as fallbacks —
Solar Pro 4 via OpenRouter, the tested path. The client speaks plain
OpenAI chat-completions with Bearer auth and no provider-specific
headers, and every completion asks for JSON that is validated against a
Pydantic schema before it can cross a workflow boundary. An OpenRouter
deployment additionally sends the provider routing preference configured
through ``LLM_PROVIDER``, and retries the transient provider failures
OpenRouter reports from its upstreams.
"""

from typing import Optional, Protocol, runtime_checkable, TypeVar

import json
import time

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

# The OpenRouter provider routing preference (ADR-0009): the provider slug the
# client prefers via the routing "order" (OpenRouter tries it first and keeps
# its usual fallbacks), empty meaning no preference at all. Chosen
# 2026-09-03: DeepInfra serves z-ai/glm-5.3-flash at 99%+ uptime, while the
# price-based default routed the same model to GMICloud (92.96%), which
# dropped completions mid-generation twice in one eval run
# (queries-glm53_v4.jsonl). The preference is an OpenRouter feature: other
# OpenAI-compatible providers receive no routing object.
DEFAULT_LLM_PROVIDER = "deepinfra"

# The locked single-pass budget: measured empirically — the old 2K cap
# truncated answers mid-object more often than it bounded them.
# ADR-0003: depth outranks latency, and the cap only has to stop runaway
# generations, not ration the answer.
MAX_COMPLETION_TOKENS = 131072

_TIMEOUT_SECONDS = 300

# The transient-failure retry: OpenRouter reports an upstream provider dying
# mid-generation as a finish_reason='error' body (HTTP 200, the real failure
# in the choice's error object), and passes HTTP 5xx/429 through the same
# way. OpenRouter re-routes to another healthy provider on the next attempt,
# so a bounded retry with a doubling backoff turns one flaky endpoint into a
# recoverable blip — queries-glm53_v4.jsonl 2026-09-03: GMICloud dropped the
# Researcher's and the Summarizer's completions in the same run.
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 1.0

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


class LlmTransientError(LlmError):
    """A provider-side failure that a retry can plausibly clear.

    An upstream that died mid-generation (OpenRouter's finish_reason='error'
    body), an HTTP 5xx, and a rate limit (429) are all upstream conditions —
    the request itself was well-formed. The client retries these itself; the
    subclass exists so a caller that exhausts the retries can tell a flaky
    upstream from a rejected request.
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
        provider: Optional[str] = None,
        transport=None,
    ):
        self._api_key = api_key
        self._model = model
        # The OpenRouter provider routing slug (ADR-0009): sent as the
        # routing "order" so OpenRouter tries it first, keeping its default
        # fallbacks. None sends no routing object at all — the plain
        # OpenAI-compatible convention other providers assume.
        self._provider = provider
        # Forgiving join: a base that already carries the chat-completions
        # path (a pasted full endpoint) is trimmed back to the version root
        # so the append in _chat cannot double it.
        base = base_url.rstrip("/")
        if base.endswith(CHAT_COMPLETIONS_PATH):
            base = base[: -len(CHAT_COMPLETIONS_PATH)]
        self._base_url = base
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
        """One completion over the given conversation, returning the reply text.

        Transient provider failures — the finish_reason='error' bodies
        OpenRouter returns when an upstream dies mid-generation, HTTP 5xx,
        and HTTP 429 — are retried with a doubling backoff up to
        ``_MAX_ATTEMPTS`` attempts: OpenRouter re-routes to another healthy
        provider on the next attempt, so one flaky endpoint must not fail
        the call. Every other failure raises on the first attempt.
        """
        payload = {
            "model": self._model,
            "max_tokens": MAX_COMPLETION_TOKENS,
            "messages": messages,
        }
        if self._provider:
            payload["provider"] = {"order": [self._provider]}
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        url = f"{self._base_url}{CHAT_COMPLETIONS_PATH}"
        attempt = 0
        while True:
            attempt += 1
            try:
                return self._post_once(url, headers, payload)
            except LlmTransientError:
                if attempt >= _MAX_ATTEMPTS:
                    raise
                time.sleep(_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))

    def _post_once(self, url: str, headers: dict, payload: dict) -> str:
        """One transport round-trip, classified: the reply text, or an error.

        Transient provider failures raise ``LlmTransientError`` (retried by
        ``_chat``); everything else raises its ``LlmError`` on the spot.
        """
        try:
            body = self._transport(url, headers, payload)
        except requests.ConnectionError as error:
            raise LlmUnreachableError(
                f"Could not reach the LLM provider at {url}: {error}. "
                "Check the network connection."
            ) from error
        except requests.HTTPError as error:
            detail = error.response.text.strip()[:300] if error.response is not None else ""
            status = error.response.status_code if error.response is not None else None
            if isinstance(status, int) and (status >= 500 or status == 429):
                raise LlmTransientError(
                    f"The LLM provider failed transiently (HTTP {status}): {detail}"
                ) from error
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
        finish = choice.get("finish_reason")
        # A provider error rides the choice (and sometimes the body) with
        # finish_reason='error' — read verbatim into the error instead of the
        # 300-character body snippet that truncated it away
        # (queries-glm53_v4.jsonl 2026-09-03). Partial content from a dead
        # stream is retried, never parsed.
        upstream = choice.get("error") or (body.get("error") if isinstance(body, dict) else None)
        if finish == "error" or upstream is not None:
            detail = upstream if upstream is not None else str(body)[:300]
            raise LlmTransientError(
                f"Model '{self._model}' failed upstream (finish_reason={finish!r}, "
                f"provider={body.get('provider')!r}): {detail!r}"
            )
        if not isinstance(content, str):
            # Reasoning models occasionally spend the whole turn on hidden
            # reasoning and emit no answer at all (live z-ai/glm-5.3-flash,
            # queries-glm53_v2.jsonl 2026-09-02T22:36:44): content arrives
            # as null. There is nothing to parse or repair — report the
            # finish_reason verbatim instead of crashing the parser. (A
            # finish_reason='error' body never reaches this branch: it is a
            # provider failure, retried above.)
            raise LlmError(
                f"Model '{self._model}' returned no content (finish_reason="
                f"{finish!r}); the turn likely went entirely "
                f"to hidden reasoning. Body: {str(body)[:300]!r}"
            )
        if finish == "length":
            raise LlmError(
                f"Model '{self._model}' hit the {MAX_COMPLETION_TOKENS}-token completion "
                f"budget before finishing (finish_reason=length); hidden reasoning tokens "
                f"count against the cap. Reply so far: {content[:200]!r}"
            )
        return content
