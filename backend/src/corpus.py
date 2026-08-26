"""The repo-local Corpus documents (ADR-0002): the one loader both paths share.

Demo mode reads full documents to resolve its lookup targets; Live mode needs
only the source id → short name map for Citation metadata. Loading is lazy
and cached once per process; a missing or unreadable corpus degrades to an
empty Corpus — never an error on the request path.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Repo-local, fixed set of regulation documents (ADR-0002).
CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "regulations"

_documents: Optional[dict[str, dict]] = None


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
