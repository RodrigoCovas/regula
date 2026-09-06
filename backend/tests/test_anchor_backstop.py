"""The reserved-anchor backstop's pure seam (issue #84): the checker that
names a reserved engagement-threshold target's uncited Evidence.

The pipeline seam is covered in ``test_live_pipeline.py``; these tests pin
the checker's own edges directly — which anchors are flagged, and which are
never flaggable because the Researcher never saw their Evidence.
"""

import sys
from pathlib import Path

# Ensure backend.src is importable when running tests from repo root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from src.live_workflow import LiveState, _uncited_reserved_anchors
from src.models import Chunk, Citation, Finding, ProvisionKind, Strength


def anchor_chunk(source_id: str, number: int, title: str, text: str) -> Chunk:
    return Chunk(
        source_id=source_id,
        kind=ProvisionKind.article,
        text=text,
        title=title,
        article_number=number,
    )


PRINCIPLES_CHUNK = anchor_chunk(
    "gdpr", 5, "Principles relating to processing", "Processing must be lawful, fair and transparent."
)
PERIMETER_CHUNK = anchor_chunk(
    "dora", 2, "Scope", "This Regulation applies to financial entities as defined in Article 3."
)
DUTY_CHUNK = anchor_chunk(
    "gdpr", 33, "Notification of a breach", "Notification of a personal data breach."
)


def pool(*labeled: tuple[str, Chunk]):
    return [
        {"label": label, "chunk": chunk.model_dump()}
        for label, chunk in labeled
    ]


def plan_with_reserved_queries(queries: list[str]) -> list[dict]:
    return [{"query": query, "reserved": True} for query in queries]


def finding_citing(source_id: str, number: int) -> Finding:
    return Finding(
        statement="A finding.",
        strength=Strength.moderate,
        citations=[Citation.model_validate({"source_id": source_id, "article_number": number})],
    )


def test_uncited_reserved_anchor_is_flagged_with_its_labels_and_provisions():
    """A reserved target whose pool labels no kept Finding cites is flagged:
    the query, the Evidence labels its own retrieval surfaced, and the
    provisions those labels carry — what the corrective re-prompt names."""
    state = LiveState.model_validate({
        "plan": plan_with_reserved_queries(["principles relating to processing", "financial-entity perimeter"]),
        "evidence": pool(("E1", PRINCIPLES_CHUNK), ("E2", PERIMETER_CHUNK)),
        "reserved_anchor_labels": {
            "principles relating to processing": ["E1"],
            "financial-entity perimeter": ["E2"],
        },
    })
    # The only kept Finding cites the reserved perimeter's provision — the
    # principles anchor is the uncited one.
    findings = [finding_citing("dora", 2)]

    anchors = _uncited_reserved_anchors(state, findings)

    assert len(anchors) == 1
    anchor = anchors[0]
    assert anchor.query == "principles relating to processing"
    assert anchor.labels == ["E1"]
    assert anchor.provisions == ["gdpr Article 5"]


def test_a_cited_reserved_anchor_is_never_flagged():
    """A reserved target whose provision some kept Finding cites is not
    flagged — the anchor landed, no re-prompt is owed."""
    state = LiveState.model_validate({
        "plan": plan_with_reserved_queries(["principles relating to processing"]),
        "evidence": pool(("E1", PRINCIPLES_CHUNK)),
        "reserved_anchor_labels": {"principles relating to processing": ["E1"]},
    })
    findings = [finding_citing("gdpr", 5)]

    assert _uncited_reserved_anchors(state, findings) == []


def test_evidence_the_researcher_never_saw_is_never_flagged():
    """A reserved target whose Evidence never reached the pool (crowded out
    of the fair-share seats) is not flaggable: the Researcher never saw that
    Evidence, and a re-prompt could only name labels it was shown."""
    state = LiveState.model_validate({
        "plan": plan_with_reserved_queries(["principles relating to processing"]),
        "evidence": pool(("E1", DUTY_CHUNK)),
        "reserved_anchor_labels": {},  # nothing seated for the reserved target
    })

    assert _uncited_reserved_anchors(state, []) == []


def test_unreserved_targets_are_never_flagged():
    """Only reserved engagement-threshold targets are anchors: a duty-area
    target whose Evidence goes uncited is not flagged."""
    state = LiveState.model_validate({
        "plan": [{"query": "breach notification duties", "reserved": False}],
        "evidence": pool(("E1", DUTY_CHUNK)),
        "reserved_anchor_labels": {"breach notification duties": ["E1"]},
    })

    assert _uncited_reserved_anchors(state, []) == []
