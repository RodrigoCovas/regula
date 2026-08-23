"""Pure Corpus -> Chunk transform (spec #9, ticket #11).

Turns one lexplorer JSON document into retrievable Chunks. No database, no
network: this module never touches PostgreSQL, Ollama, or the filesystem, so
it runs offline and is trivially testable against the real corpus files.

Locked hybrid chunking policy:

- Every Article, every Recital that carries text, and every Annex becomes at
  least one Chunk. Nothing with content is dropped; the known corpus gap
  (AI Act / GDPR recitals ship without text, see data/PROVENANCE.md) means
  those placeholder recitals yield no chunks.
- Each Chunk targets exactly one provision (Article XOR Recital XOR Annex),
  carrying the structured provision numbers Citations are later derived from.
- Long Articles split at paragraph boundaries. A single paragraph that alone
  exceeds the budget falls back to its own points (definitions lists), and a
  leaf unit with no internal structure falls back to sentence packing. Every
  fallback preserves all source text verbatim.
- Recitals are never split: recital integrity outranks the ~400-token target,
  so an occasional oversized Recital Chunk is the only sanctioned exception.
- Annexes pack their line structure (list items) under one heading.

Token counts are estimated deterministically (length / 4) so the transform
needs no tokenizer download and produces byte-stable output.
"""

import json
import math
import re
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any, Callable, Optional

from src.models import Chunk, ProvisionKind

MAX_CHUNK_TOKENS = 400

_CHARACTERS_PER_TOKEN = 4
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.;])\s+")
_PART_MARKER_LIMIT = 4  # tokens to reserve for a "(part n/m)" suffix


def estimate_tokens(text: str) -> int:
    """Deterministic offline token estimate (~4 characters per token)."""
    return math.ceil(len(text) / _CHARACTERS_PER_TOKEN)


def chunk_regulation(document: Mapping[str, Any]) -> list[Chunk]:
    """Transform one lexplorer document dict into its Chunks."""
    source_id = str(document["metadata"]["id"])
    chunks: list[Chunk] = []
    for article in _iter_articles(document):
        chunks.extend(
            _chunks_for_provision(
                source_id,
                ProvisionKind.article,
                int(article["number"]),
                _optional_text(article.get("title")),
                partial(_article_units, article),
            )
        )
    for recital in document.get("recitals", []):
        text = (recital.get("text") or "").strip()
        if not text:
            continue  # AI Act / GDPR recitals carry no quotable text this iteration
        chunks.extend(
            _chunks_for_provision(
                source_id,
                ProvisionKind.recital,
                int(recital["number"]),
                None,
                partial(_whole_text_units, text),
                allow_oversize=True,
            )
        )
    for annex in document.get("annexes", []):
        chunks.extend(
            _chunks_for_provision(
                source_id,
                ProvisionKind.annex,
                _annex_number(annex),
                _optional_text(annex.get("title")),
                partial(_annex_units, annex),
            )
        )
    return chunks


def _whole_text_units(text: str, budget: int) -> list[str]:
    """Recital units: the whole text, never split (recital integrity is locked)."""
    return [text]


# --- Provision walkers -------------------------------------------------------


def _iter_articles(document: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Chapter-nested articles in document order, deduplicated by id.

    The lexplorer JSON repeats two AI Act article ids with identical content;
    identical duplicates are skipped, conflicting ones fail loudly.
    """
    articles: list[Mapping[str, Any]] = []
    fingerprints: dict[str, str] = {}
    for chapter in document.get("chapters", []):
        for section in chapter.get("sections", []):
            for article in section.get("articles", []):
                article_id = str(article["id"])
                fingerprint = json.dumps(article, sort_keys=True)
                if article_id in fingerprints:
                    if fingerprints[article_id] != fingerprint:
                        raise ValueError(
                            f"Corpus lists article {article['number']} twice "
                            "with conflicting content"
                        )
                    continue
                fingerprints[article_id] = fingerprint
                articles.append(article)
    return articles


def _annex_number(annex: Mapping[str, Any]) -> int:
    """Derive the annex number from its id (the corpus ids are 'annex-N')."""
    _, _, suffix = str(annex["id"]).rpartition("-")
    try:
        return int(suffix)
    except ValueError as error:
        raise ValueError(f"Annex id '{annex['id']}' carries no usable number") from error


# --- Unit extraction ---------------------------------------------------------


def _article_units(article: Mapping[str, Any], budget: int) -> list[str]:
    """Ordered text units for one article: paragraphs, split further only when oversized."""
    units: list[str] = []
    for paragraph in article.get("paragraphs", []):
        text = (paragraph.get("text") or "").strip()
        points = paragraph.get("points") or []
        if not text and not points:
            continue
        prefix = f"{paragraph['number']}. " if isinstance(paragraph.get("number"), int) else ""
        if not points:
            units.extend(_leaf_units(prefix + text, budget))
            continue
        whole = "\n".join([prefix + text] + [_point_unit(point) for point in points])
        if estimate_tokens(whole) <= budget:
            units.append(whole)
            continue
        # A definitions-style paragraph: its points become the atomic units,
        # with the paragraph's own lead-in sentence kept ahead of them.
        if text:
            units.extend(_leaf_units(prefix + text, budget))
        for point in points:
            units.extend(_leaf_units(_point_unit(point), budget))
    if not units:
        raise ValueError(f"corpus article {article['number']} carries no text to chunk")
    return units


def _point_unit(point: Mapping[str, Any]) -> str:
    """One point (plus any sub-points) as a labelled multi-line unit."""
    lines = [_labelled_line(point)]
    lines.extend(_labelled_line(sub_point) for sub_point in point.get("subPoints") or [])
    return "\n".join(line for line in lines if line)


def _labelled_line(item: Mapping[str, Any]) -> str:
    label = item.get("label")
    text = (item.get("text") or "").strip()
    if not text:
        return ""
    return f"({label}) {text}" if label else text


def _annex_units(annex: Mapping[str, Any], budget: int) -> list[str]:
    """Line-structured units for one annex; blank-line blocks stay intact."""
    units: list[str] = []
    block: list[str] = []
    for line in str(annex.get("content") or "").splitlines():
        if line.strip():
            block.append(line.strip())
            continue
        if block:
            units.extend(_leaf_units("\n".join(block), budget))
            block = []
    if block:
        units.extend(_leaf_units("\n".join(block), budget))
    if not units:
        raise ValueError(f"corpus annex '{annex['id']}' carries no content to chunk")
    return units


def _leaf_units(text: str, budget: int) -> list[str]:
    """Keep a unit whole unless it busts the budget; then pack it by sentence."""
    text = text.strip()
    if not text:
        return []
    if estimate_tokens(text) <= budget:
        return [text]
    return _pack_sentences(text, budget)


def _pack_sentences(text: str, budget: int) -> list[str]:
    """Greedy sentence packing for unstructured oversized leaves."""
    sentences = [
        sentence.strip() for sentence in _SENTENCE_BOUNDARY.split(text) if sentence.strip()
    ]
    return [" ".join(group) for group in _pack(sentences, budget)]


# --- Packing and assembly ----------------------------------------------------


def _pack(units: Sequence[str], budget: int) -> list[list[str]]:
    """Greedy consecutive packing: never reorder, never mix, split only on overflow."""
    groups: list[list[str]] = []
    current: list[str] = []
    size = 0
    for unit in units:
        tokens = estimate_tokens(unit)
        if current and size + tokens > budget:
            groups.append(current)
            current, size = [], 0
        current.append(unit)
        size += tokens
    if current:
        groups.append(current)
    return groups


def _chunks_for_provision(
    source_id: str,
    kind: ProvisionKind,
    number: int,
    title: Optional[str],
    build_units: Callable[[int], list[str]],
    allow_oversize: bool = False,
) -> list[Chunk]:
    """Assemble the Chunks of one provision from its text units.

    The heading (plus part marker) is reserved off the budget so every finished
    Chunk stays inside MAX_CHUNK_TOKENS once labelled. Recitals pass
    ``allow_oversize`` to stay whole regardless of size.
    """
    heading = _heading(kind, number, title)
    body_budget = max(MAX_CHUNK_TOKENS - estimate_tokens(heading) - _PART_MARKER_LIMIT, 1)
    units = build_units(body_budget)
    if not units:
        raise ValueError(f"{kind.value} {number} of '{source_id}' carries no text to chunk")
    if allow_oversize:
        groups = [units]
    else:
        groups = _pack(units, body_budget)
    numbers = {
        ProvisionKind.article: {"article_number": number},
        ProvisionKind.recital: {"recital_number": number},
        ProvisionKind.annex: {"annex_number": number},
    }[kind]
    return [
        Chunk(
            source_id=source_id,
            kind=kind,
            title=title,
            chunk_index=index,
            num_chunks=len(groups),
            text=_labelled_heading(heading, index, len(groups)) + "\n" + "\n".join(group),
            **numbers,
        )
        for index, group in enumerate(groups)
    ]


def _heading(kind: ProvisionKind, number: int, title: Optional[str]) -> str:
    label = {"article": "Article", "recital": "Recital", "annex": "Annex"}[kind]
    return f"{label} {number}: {title}" if title else f"{label} {number}"


def _labelled_heading(heading: str, index: int, total: int) -> str:
    return f"{heading} (part {index + 1}/{total})" if total > 1 else heading


def _optional_text(value: Any) -> Optional[str]:
    cleaned = value.strip() if isinstance(value, str) else ""
    return cleaned or None
