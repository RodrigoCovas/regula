"""The Summarizer stage (issue #47) at the pure seam.

Three pieces live here away from HTTP: the max-rule Citation strength
(CONTEXT.md) shared by the product and the eval, the Answer's deduplicated
Citation list, and the Summarizer's grounding gate — the application code
that lets only cited provisions receive a relevance statement.
"""

import sys
from pathlib import Path

# Ensure backend.src is importable when running tests from repo root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import pytest

from src.live_workflow import ProvisionSummary, Summaries, _validate_summaries
from src.models import (
    Citation,
    Finding,
    ProvisionKind,
    ProvisionTarget,
    Strength,
    answer_citations,
    max_rule_strengths,
)


def _citation(source_id: str, kind: ProvisionKind, number: int) -> Citation:
    return Citation.model_validate({
        "source_id": source_id,
        **{f"{kind.value}_number": number},
    })


def _finding(statement: str, strength: Strength, *citations: Citation) -> Finding:
    return Finding(statement=statement, strength=strength, citations=list(citations))


# --- The max-rule (CONTEXT.md: Citation strength) ---


def test_the_max_rule_keeps_the_strongest_strength_per_target():
    strengths = max_rule_strengths([
        (ProvisionTarget("ai-act", ProvisionKind.article, 6), Strength.weak),
        (ProvisionTarget("ai-act", ProvisionKind.article, 6), Strength.strong),
        (ProvisionTarget("ai-act", ProvisionKind.article, 6), Strength.moderate),
    ])
    assert strengths == {ProvisionTarget("ai-act", ProvisionKind.article, 6): Strength.strong}


def test_the_max_rule_is_per_target():
    strengths = max_rule_strengths([
        (ProvisionTarget("ai-act", ProvisionKind.article, 6), Strength.weak),
        (ProvisionTarget("gdpr", ProvisionKind.article, 22), Strength.strong),
    ])
    assert strengths == {
        ProvisionTarget("ai-act", ProvisionKind.article, 6): Strength.weak,
        ProvisionTarget("gdpr", ProvisionKind.article, 22): Strength.strong,
    }


def test_same_number_different_kind_or_source_are_different_targets():
    strengths = max_rule_strengths([
        (ProvisionTarget("ai-act", ProvisionKind.article, 6), Strength.weak),
        (ProvisionTarget("ai-act", ProvisionKind.annex, 6), Strength.strong),
    ])
    assert strengths[ProvisionTarget("ai-act", ProvisionKind.article, 6)] == Strength.weak
    assert strengths[ProvisionTarget("ai-act", ProvisionKind.annex, 6)] == Strength.strong


# --- The Answer's citation list ---


def test_answer_citations_carry_one_entry_per_cited_provision():
    """Two Findings citing the same provision collapse to one Answer entry —
    one Provision relevance statement per cited provision, not per Finding."""
    high_risk = _citation("ai-act", ProvisionKind.article, 6)
    findings = [
        _finding("Strong claim.", Strength.strong, high_risk),
        _finding("Weak framing claim.", Strength.weak, high_risk),
    ]
    citations = answer_citations(findings)
    assert len(citations) == 1
    assert citations[0].article_number == 6


def test_answer_citations_keep_first_mention_order():
    article_6 = _citation("ai-act", ProvisionKind.article, 6)
    recital_71 = _citation("gdpr", ProvisionKind.recital, 71)
    findings = [
        _finding("f1", Strength.strong, article_6),
        _finding("f2", Strength.moderate, recital_71),
    ]
    assert [c.provision_target for c in answer_citations(findings)] == [
        ProvisionTarget("ai-act", ProvisionKind.article, 6),
        ProvisionTarget("gdpr", ProvisionKind.recital, 71),
    ]


def test_answer_citations_carry_the_max_rule_strength():
    high_risk = _citation("ai-act", ProvisionKind.article, 6)
    definitions = _citation("ai-act", ProvisionKind.article, 3)
    findings = [
        _finding("Strong claim.", Strength.strong, high_risk),
        _finding("Weak framing claim.", Strength.weak, high_risk, definitions),
    ]
    strengths = {c.provision_target: c.strength for c in answer_citations(findings)}
    assert strengths[ProvisionTarget("ai-act", ProvisionKind.article, 6)] == Strength.strong
    assert strengths[ProvisionTarget("ai-act", ProvisionKind.article, 3)] == Strength.weak


def test_answer_citations_attach_the_provisions_relevance():
    high_risk = _citation("ai-act", ProvisionKind.article, 6)
    findings = [_finding("Strong claim.", Strength.strong, high_risk)]
    target = ProvisionTarget("ai-act", ProvisionKind.article, 6)
    citations = answer_citations(findings, {target: "Names creditworthiness evaluation as high-risk."})
    assert citations[0].relevance == "Names creditworthiness evaluation as high-risk."


def test_answer_citations_without_relevance_stay_none():
    high_risk = _citation("ai-act", ProvisionKind.article, 6)
    findings = [_finding("Strong claim.", Strength.strong, high_risk)]
    citations = answer_citations(findings)
    assert citations[0].relevance is None


def test_answer_citations_never_mutate_the_findings_own_citations():
    """Relevance and strength are answer-wide: they ride the Answer's list,
    never the per-Finding Citations the Findings section renders."""
    high_risk = _citation("ai-act", ProvisionKind.article, 6)
    findings = [_finding("Strong claim.", Strength.strong, high_risk)]
    answer_citations(findings, {high_risk.provision_target: "relevance"})
    assert findings[0].citations[0].relevance is None
    assert findings[0].citations[0].strength is None


# --- The Summarizer's grounding gate ---


def test_a_reference_to_a_cited_provision_is_kept():
    target = ProvisionTarget("ai-act", ProvisionKind.article, 6)
    relevance, decisions = _validate_summaries(
        Summaries(summaries=[ProvisionSummary(ref="P1", relevance="Names the situation outright.")]),
        {"P1": target},
    )
    assert relevance == {target: "Names the situation outright."}
    assert [(d.ref, d.status) for d in decisions] == [("P1", "kept")]


def test_a_reference_to_no_cited_provision_is_rejected():
    relevance, decisions = _validate_summaries(
        Summaries(summaries=[ProvisionSummary(ref="P9", relevance="Invented grounding.")]),
        {"P1": ProvisionTarget("ai-act", ProvisionKind.article, 6)},
    )
    assert relevance == {}
    assert decisions[0].status == "rejected"
    assert decisions[0].reason is not None


def test_finding_and_citation_labels_name_no_cited_provision():
    """The Summarizer is shown P-labels only: F1 and C1 references cannot
    smuggle Findings or per-Finding Citations into the relevance map."""
    relevance, decisions = _validate_summaries(
        Summaries(summaries=[
            ProvisionSummary(ref="F1", relevance="One."),
            ProvisionSummary(ref="C1", relevance="Two."),
        ]),
        {"P1": ProvisionTarget("ai-act", ProvisionKind.article, 6)},
    )
    assert relevance == {}
    assert [d.status for d in decisions] == ["rejected", "rejected"]


def test_one_provision_keeps_only_its_first_relevance_statement():
    """One grounded statement per cited provision: a second statement for the
    same provision is rejected, never merged or overwritten."""
    target = ProvisionTarget("ai-act", ProvisionKind.article, 6)
    relevance, decisions = _validate_summaries(
        Summaries(summaries=[
            ProvisionSummary(ref="P1", relevance="First statement."),
            ProvisionSummary(ref="P1", relevance="Second statement."),
        ]),
        {"P1": target},
    )
    assert relevance == {target: "First statement."}
    assert [d.status for d in decisions] == ["kept", "rejected"]


def test_an_empty_relevance_statement_is_rejected():
    target = ProvisionTarget("ai-act", ProvisionKind.article, 6)
    relevance, decisions = _validate_summaries(
        Summaries(summaries=[ProvisionSummary(ref="P1", relevance="   ")]),
        {"P1": target},
    )
    assert relevance == {}
    assert decisions[0].status == "rejected"


def test_distinct_provisions_keep_their_own_statements():
    article_6 = ProvisionTarget("ai-act", ProvisionKind.article, 6)
    recital_71 = ProvisionTarget("gdpr", ProvisionKind.recital, 71)
    relevance, decisions = _validate_summaries(
        Summaries(summaries=[
            ProvisionSummary(ref="P1", relevance="Article statement."),
            ProvisionSummary(ref="P2", relevance="Recital statement."),
        ]),
        {"P1": article_6, "P2": recital_71},
    )
    assert relevance == {article_6: "Article statement.", recital_71: "Recital statement."}
    assert [d.status for d in decisions] == ["kept", "kept"]


# --- The Summarizer's prompt block ---


def test_a_finding_citing_one_provision_twice_is_listed_once():
    """One Finding citing the same provision through several Citations lists
    its statement once under the ref — duplication in the prompt can only
    weight the material, never inform."""
    from src.live_workflow import _labelled_provisions

    statement = "A finding citing one provision through two Citations."
    findings = [
        _finding(
            statement,
            Strength.strong,
            _citation("ai-act", ProvisionKind.article, 6),
            _citation("ai-act", ProvisionKind.article, 6),
        )
    ]
    block, targets_by_ref = _labelled_provisions(findings)
    assert list(targets_by_ref.values()) == [ProvisionTarget("ai-act", ProvisionKind.article, 6)]
    assert block.count(statement) == 1
