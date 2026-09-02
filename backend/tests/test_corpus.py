"""Tests for the Corpus inventory (issue #62, ADR-0012).

The inventory is the Corpus's own table of contents — one entry per titled
provision (Articles and Annexes), rendered from Chunk metadata and grouped by
source — the Planner's grounding for target formulation. Recitals carry no
titles and stay reachable through search. The transform is pure, so these
tests exercise it directly against the real corpus files offline.
"""

import sys
from pathlib import Path

# Ensure backend.src is importable when running tests from repo root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import re

from src.corpus import corpus_inventory

# The fixed curated Corpus (ADR-0002): every Article and Annex of all three
# regulations carries a title, so each source's entry count is pinned by the
# same structural totals test_chunking pins (test_chunking.EXPECTED_*).
EXPECTED_ENTRY_COUNTS = {"EU AI Act": 126, "DORA": 64, "GDPR": 99}

_ENTRY_PATTERN = re.compile(r"^-(?: (Article|Annex) (\d+)(?:: (.+))?)$")


def inventory_sections() -> dict[str, list[str]]:
    """The rendered block split into per-source entry lists."""
    result: dict[str, list[str]] = {}
    current: str | None = None
    for line in corpus_inventory().splitlines():
        if not line.strip():
            continue
        if line.startswith("-"):
            assert current is not None, f"entry outside any source group: {line!r}"
            result[current].append(line)
        else:
            current = line.rstrip(":")
            result[current] = []
    return result


def test_inventory_groups_titled_provisions_by_source():
    """One group per source under the corpus's own short names, entries in
    document order beneath their group."""
    entries = inventory_sections()
    assert set(entries) == {"EU AI Act", "DORA", "GDPR"}
    # Entries stay grouped: the same provision number appears under each
    # source with that source's own title.
    assert "- Article 3: Definitions" in entries["EU AI Act"]
    assert "- Article 3: Territorial scope" in entries["GDPR"]
    assert "- Article 6: ICT risk management framework" in entries["DORA"]


def test_inventory_lists_titled_articles_and_annexes_never_recitals():
    """Every entry is a titled Article or Annex — recitals carry no titles and
    never enter the inventory (they stay reachable through search)."""
    for source, entry_lines in inventory_sections().items():
        assert len(entry_lines) == EXPECTED_ENTRY_COUNTS[source], source
        for line in entry_lines:
            match = _ENTRY_PATTERN.match(line)
            assert match is not None, f"{source}: malformed inventory entry {line!r}"
            assert match.group(1) in {"Article", "Annex"}
            assert match.group(3), f"{source}: a titled provision names its title {line!r}"
        assert not any("Recital" in line for line in entry_lines), source


def test_inventory_lists_each_provision_once_per_source():
    """One entry per provision, not per Chunk: a multi-part Article occupies
    one line, and no provision number repeats within a source group."""
    for source, entry_lines in inventory_sections().items():
        keys = [
            (match.group(1), match.group(2))
            for line in entry_lines
            if (match := _ENTRY_PATTERN.match(line))
        ]
        assert len(keys) == len(set(keys)), f"{source}: a provision is listed twice"


def test_inventory_names_the_ai_act_annexes():
    """The AI Act's Annexes enter the inventory with their titles — Annex
    provisions are titled and must be liftable into target keywords."""
    annex_entries = [
        line for line in inventory_sections()["EU AI Act"] if line.startswith("- Annex")
    ]
    assert len(annex_entries) == 13
    assert any("Annex 1: List of Union Harmonisation Legislation" in line for line in annex_entries)
    assert any("Annex 3: High-Risk AI Systems Referred to in Article 6(2)" in line for line in annex_entries)
