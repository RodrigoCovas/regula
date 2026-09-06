"""The repo-local Corpus documents (ADR-0002): the one loader both paths share.

Demo mode reads full documents to resolve its lookup targets; Live mode needs
only the source id → short name map for Citation metadata, plus the Corpus
inventory that grounds the Planner (ADR-0012). Loading is lazy and cached once
per process; a missing or unreadable corpus degrades to an empty Corpus —
never an error on the request path.
"""

import json
import logging
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Optional

from .chunking import chunk_regulation
from .models import PROVISION_NOUNS, ProvisionKind, ProvisionTarget

logger = logging.getLogger(__name__)

# Repo-local, fixed set of regulation documents (ADR-0002).
CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "regulations"

_documents: Optional[dict[str, dict]] = None
_inventory: Optional[str] = None
_perimeter: Optional[dict[str, frozenset[ProvisionTarget]]] = None


class TitledProvision(NamedTuple):
    """One titled provision of the Corpus inventory: the Chunk metadata
    collapsed to a single entry per provision."""

    kind: ProvisionKind
    number: int
    title: str


def load_documents() -> dict[str, dict]:
    """source id → document, loaded lazily once per process.

    Documents that cannot be read or parsed are logged and skipped: a broken
    corpus file degrades to a smaller Corpus, never a request-path error.
    """
    global _documents
    if _documents is None:
        if not CORPUS_DIR.is_dir():
            logger.warning("Corpus path %s does not exist; retrieval will be limited", CORPUS_DIR)
            _documents = {}
        else:
            loaded: dict[str, dict] = {}
            for path in sorted(CORPUS_DIR.glob("*.json")):
                try:
                    document = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    logger.warning("Skipping corpus file %s: %s", path, error)
                    continue
                doc_id = (document.get("metadata") or {}).get("id")
                if doc_id:
                    loaded[str(doc_id)] = document
            _documents = loaded
    return _documents


def source_short_names() -> dict[str, str]:
    """source id → short name, from each document's metadata."""
    names: dict[str, str] = {}
    for doc_id, document in load_documents().items():
        short_name = (document.get("metadata") or {}).get("shortName")
        if short_name:
            names[doc_id] = str(short_name)
    return names


def perimeter_provisions() -> dict[str, frozenset[ProvisionTarget]]:
    """source id → the Perimeter provisions that decide who the Regulation
    covers (issue #86): the engagement gate reads them to compute a
    Regulation's Engagement state from the kept Findings — a kept Finding
    citing one of these targets carries engagement evidence, applying or as
    an open question, or asserts the exclusion over it. Each document
    declares its own list in its metadata (the operator's curated judgment,
    recorded with the gate's ADR); the loader parses the entries into the
    same structural targets Citations and ground truth compare on, skipping
    malformed ones with a warning — a broken declaration degrades the gate's
    knowledge, never the request path. Cached once per process like the
    documents themselves."""
    global _perimeter
    if _perimeter is None:
        table: dict[str, frozenset[ProvisionTarget]] = {}
        for doc_id, document in load_documents().items():
            targets: set[ProvisionTarget] = set()
            for entry in (document.get("metadata") or {}).get("perimeter") or []:
                parsed = _perimeter_entry(doc_id, entry)
                if parsed is not None:
                    targets.add(parsed)
                else:
                    logger.warning("Skipping malformed perimeter entry in %s: %r", doc_id, entry)
            table[doc_id] = frozenset(targets)
        _perimeter = table
    return _perimeter


def _perimeter_entry(doc_id: str, entry: Any) -> Optional[ProvisionTarget]:
    """One metadata perimeter entry as a structural target, or None when the
    entry is not a well-formed provision reference."""
    if not isinstance(entry, Mapping):
        return None
    try:
        kind = ProvisionKind(entry.get("kind"))
        number = int(entry["number"])
    except (KeyError, TypeError, ValueError):
        return None
    return ProvisionTarget(doc_id, kind, number)


def _titled_provisions(document: Mapping[str, Any]) -> list[TitledProvision]:
    """One TitledProvision per titled provision of one document, in document
    order — the Chunk metadata, collapsed to one entry per provision (a
    multi-part Article shares its title across its Chunks). A provision whose
    chunks disagree on the title fails loudly, mirroring the chunking
    transform's own duplicate policy."""
    entries: list[TitledProvision] = []
    titles: dict[tuple[ProvisionKind, int], str] = {}
    for chunk in chunk_regulation(document):
        if chunk.title is None:
            continue  # recitals carry no titles and stay reachable through search
        number = chunk.provision_number
        if number is None:  # unreachable: the Chunk validator enforces exactly-one-target
            raise ValueError(
                f"titled {chunk.kind.value} chunk of '{chunk.source_id}' carries no provision number"
            )
        key = (chunk.kind, number)
        first_title = titles.get(key)
        if first_title is not None:
            if first_title != chunk.title:
                raise ValueError(
                    f"{chunk.kind.value} {number} of '{chunk.source_id}' carries conflicting titles: "
                    f"{first_title!r} vs {chunk.title!r}"
                )
            continue
        titles[key] = chunk.title
        entries.append(TitledProvision(chunk.kind, number, chunk.title))
    return entries


def corpus_inventory() -> str:
    """The Corpus inventory (CONTEXT.md, ADR-0012): the Corpus's own table of
    contents, rendered from Chunk metadata — every titled Article and Annex,
    grouped by source under its short name. The Planner lifts the inventory's
    exact vocabulary into Research-target keywords; it is vocabulary guidance,
    not a constraint. Rendered through the same pure chunking transform ingest
    runs, so the inventory always matches what the Corpus actually stores, and
    cached once per process like the documents themselves. An unreadable
    Corpus degrades to an empty block — never a request-path error."""
    global _inventory
    if _inventory is None:
        groups: list[str] = []
        short_names = source_short_names()
        for source_id, document in load_documents().items():
            name = short_names.get(source_id, source_id)
            entries = _titled_provisions(document)
            if not entries:
                continue
            lines = [f"{name}:"] + [
                f"- {PROVISION_NOUNS[entry.kind]} {entry.number}: {entry.title}"
                for entry in entries
            ]
            groups.append("\n".join(lines))
        _inventory = "\n\n".join(groups)
    return _inventory
