"""Per-request observability appended as JSON Lines (spec #9, ticket #19).

The documented query-log convention — ``logs/queries.jsonl``, one JSON
object per line — carries the cost and behaviour basics per request across
both modes: latency, retrieved-Chunk counts, token usage when the LLM ran,
and an explicit success/failure status. Minimal by design: a stdlib file
append per request, no platform, no new infrastructure.

Two rules keep the log honest:

- A request that fails mid-workflow is still recorded, with the tokens it
  already spent — cost stays observable precisely when things go wrong.
- Observability never takes serving down: an unwritable destination
  degrades to a logged warning.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# The documented location (README project structure). Relative paths resolve
# against the process working directory — /app in the Docker stack, where
# ./logs is volume-mounted.
DEFAULT_QUERY_LOG_PATH = "logs/queries.jsonl"

STATUS_SUCCESS = "success"
STATUS_FAILURE = "failure"

_ERROR_SNIPPET_LIMIT = 300


@dataclass
class RequestObservation:
    """What one request observed so far, filled in incrementally.

    Created per request in the endpoint and handed down the dispatch path so
    counts land even when a later stage fails: a request that dies after
    retrieving Chunks still reports them.
    """

    retrieved_chunks: int = 0


def aggregate_token_usage(llm: Any) -> Optional[dict[str, int]]:
    """Sum the per-call token usage an LLM client recorded this request.

    Clients expose their per-call provider usage dictionaries duck-typed as
    an optional ``usage`` list; anything without one — Demo mode's absence
    of an LLM, fakes that spent nothing — aggregates to ``None``.
    """
    usages = getattr(llm, "usage", None)
    if not usages:
        return None

    prompt = completion = total = 0
    for entry in usages:
        if not isinstance(entry, dict):
            continue
        entry_prompt = int(entry.get("prompt_tokens") or 0)
        entry_completion = int(entry.get("completion_tokens") or 0)
        # Derive the call's total only when the provider omitted it — a
        # reported 0 is a 0, not an absence.
        entry_total = int(entry.get("total_tokens") or 0) or entry_prompt + entry_completion
        prompt += entry_prompt
        completion += entry_completion
        total += entry_total
    return {"prompt": prompt, "completion": completion, "total": total}


def build_query_record(
    *,
    mode: str,
    scenario_id: Optional[str],
    workflow: Optional[str],
    status: str,
    latency_ms: float,
    retrieved_chunks: int,
    tokens: Optional[dict[str, int]],
    error: Optional[str] = None,
) -> dict[str, Any]:
    """One uniform JSONL record: every key always present, null when N/A."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "scenario_id": scenario_id,
        "workflow": workflow,
        "status": status,
        "latency_ms": round(latency_ms, 3),
        "retrieved_chunks": retrieved_chunks,
        "tokens": tokens,
        "error": error[:_ERROR_SNIPPET_LIMIT] if error else None,
    }


def append_query_record(path: str | Path, record: dict[str, Any]) -> bool:
    """Append one record as a single terminated JSON line; did it land?

    Best-effort by contract: a broken destination logs a warning and returns
    False instead of raising into the request path.
    """
    try:
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as query_log:
            query_log.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except OSError as error:
        logger.warning("Could not append to the query log at %s: %s", path, error)
        return False
