"""The Summarizer stage (issue #47, ADR-0011) at the pure seam.

Three pieces live here away from HTTP: the max-rule Citation strength the
eval's expected side reads (CONTEXT.md), the Answer's deduplicated Citation
list badged with the rated strength the Summarizer carried — never derived
from Finding strength — and the Summarizer's grounding gate, the
application code that lets only cited provisions receive a relevance
statement and only a valid rating ride beside it.
"""

import sys
from pathlib import Path

# Ensure backend.src is importable when running tests from repo root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import pytest

from src.live_workflow import (
    ProvisionSummary,
    Summaries,
    _SUMMARIZER_SYSTEM,
    _validate_summaries,
)
from src.models import (
    Citation,
    Finding,
    GroundedSummary,
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


def _grounded(target: ProvisionTarget, relevance: str, strength: Strength | None = None) -> GroundedSummary:
    return GroundedSummary(target=target, relevance=relevance, strength=strength)


# --- The max-rule (CONTEXT.md: the eval's expected side only) ---


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


def test_answer_citations_carry_the_rated_strength():
    """The rated Citation strength rides the Answer's list (ADR-0011): the
    rating the Summarizer carried, never a derivation from the citing
    Findings' Strengths."""
    high_risk = _citation("ai-act", ProvisionKind.article, 6)
    definitions = _citation("ai-act", ProvisionKind.article, 3)
    findings = [
        _finding("Strong claim.", Strength.strong, high_risk),
        _finding("Weak framing claim.", Strength.weak, high_risk, definitions),
    ]
    citations = answer_citations(findings, {
        high_risk.provision_target: _grounded(high_risk.provision_target, "rated.", Strength.moderate),
    })
    strengths = {c.provision_target: c.strength for c in citations}
    assert strengths[ProvisionTarget("ai-act", ProvisionKind.article, 6)] == Strength.moderate


def test_an_unrated_provision_keeps_relevance_but_carries_no_strength():
    """No default value anywhere (ADR-0011): a provision the Summarizer never
    rated carries no strength even when strong Findings cite it — nothing is
    derived from Finding strength any more."""
    high_risk = _citation("ai-act", ProvisionKind.article, 6)
    findings = [_finding("Strong claim.", Strength.strong, high_risk)]
    citations = answer_citations(findings, {high_risk.provision_target: _grounded(high_risk.provision_target, "Names the situation.")})
    assert citations[0].relevance == "Names the situation."
    assert citations[0].strength is None


def test_answer_citations_attach_the_provisions_relevance():
    high_risk = _citation("ai-act", ProvisionKind.article, 6)
    findings = [_finding("Strong claim.", Strength.strong, high_risk)]
    target = ProvisionTarget("ai-act", ProvisionKind.article, 6)
    citations = answer_citations(findings, {target: _grounded(target, "Names creditworthiness evaluation as high-risk.")})
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
    answer_citations(findings, {high_risk.provision_target: _grounded(high_risk.provision_target, "relevance")})
    assert findings[0].citations[0].relevance is None
    assert findings[0].citations[0].strength is None


# --- The Summarizer's grounding gate ---


def test_a_reference_to_a_cited_provision_is_kept_with_its_rating():
    target = ProvisionTarget("ai-act", ProvisionKind.article, 6)
    grounded, decisions = _validate_summaries(
        Summaries(summaries=[
            ProvisionSummary(ref="P1", relevance="Names the situation outright.", strength=Strength.strong)
        ]),
        {"P1": target},
    )
    assert grounded == {target: _grounded(target, "Names the situation outright.", Strength.strong)}
    assert [(d.ref, d.status, d.strength) for d in decisions] == [("P1", "kept", Strength.strong)]


def test_a_bare_rating_string_is_carried_as_its_level():
    """The schema keeps the rating loose, so a plain JSON string arrives
    uncoerced; the gate validates it into the Strength it names."""
    target = ProvisionTarget("ai-act", ProvisionKind.article, 6)
    grounded, _ = _validate_summaries(
        Summaries(summaries=[ProvisionSummary(ref="P1", relevance="Grounded.", strength="weak")]),
        {"P1": target},
    )
    assert grounded[target].strength == Strength.weak


def test_a_wrapped_rating_object_is_unwrapped():
    """The live model wraps the level with a rationale — the same repair the
    Verifier's strength already applies keeps the bare level."""
    target = ProvisionTarget("ai-act", ProvisionKind.article, 6)
    grounded, decisions = _validate_summaries(
        Summaries(summaries=[
            ProvisionSummary(
                ref="P1",
                relevance="Grounded.",
                strength={"level": "moderate", "rationale": "supporting duty"},
            )
        ]),
        {"P1": target},
    )
    assert grounded[target].strength == Strength.moderate
    assert decisions[0].status == "kept"


def test_a_missing_rating_keeps_the_relevance_without_a_strength():
    """A partial Summarizer pass degrades silently, never failing the run
    (ADR-0011): the relevance ships, no strength is defaulted, and the gap
    is recorded with a reason in the detailed trace."""
    target = ProvisionTarget("ai-act", ProvisionKind.article, 6)
    grounded, decisions = _validate_summaries(
        Summaries(summaries=[ProvisionSummary(ref="P1", relevance="Grounded statement.")]),
        {"P1": target},
    )
    assert grounded == {target: _grounded(target, "Grounded statement.", None)}
    decision = decisions[0]
    assert decision.status == "kept"
    assert decision.strength is None
    assert decision.reason, "the missing rating is recorded, never defaulted"


def test_an_invalid_rating_keeps_the_relevance_without_a_strength():
    """A value outside the three levels is no rating: the relevance ships,
    the strength does not, and the reason names the offending value."""
    target = ProvisionTarget("ai-act", ProvisionKind.article, 6)
    grounded, decisions = _validate_summaries(
        Summaries(summaries=[
            ProvisionSummary(ref="P1", relevance="Grounded statement.", strength="decisive")
        ]),
        {"P1": target},
    )
    assert grounded == {target: _grounded(target, "Grounded statement.", None)}
    decision = decisions[0]
    assert decision.status == "kept"
    assert decision.strength is None
    assert "decisive" in (decision.reason or "")


def test_a_rating_gap_is_recorded_per_provision_not_per_run():
    """A rated and an unrated provision in one pass: the rated one carries
    its strength, the unrated one keeps only its relevance — independent
    degradation (parent spec, user story 11)."""
    article_6 = ProvisionTarget("ai-act", ProvisionKind.article, 6)
    recital_71 = ProvisionTarget("gdpr", ProvisionKind.recital, 71)
    grounded, decisions = _validate_summaries(
        Summaries(summaries=[
            ProvisionSummary(ref="P1", relevance="Article statement.", strength=Strength.strong),
            ProvisionSummary(ref="P2", relevance="Recital statement."),
        ]),
        {"P1": article_6, "P2": recital_71},
    )
    assert grounded[article_6].strength == Strength.strong
    assert grounded[recital_71].strength is None
    assert [d.strength for d in decisions] == [Strength.strong, None]


def test_a_reference_to_no_cited_provision_is_rejected():
    grounded, decisions = _validate_summaries(
        Summaries(summaries=[ProvisionSummary(ref="P9", relevance="Invented grounding.")]),
        {"P1": ProvisionTarget("ai-act", ProvisionKind.article, 6)},
    )
    assert grounded == {}
    assert decisions[0].status == "rejected"
    assert decisions[0].reason is not None


def test_finding_and_citation_labels_name_no_cited_provision():
    """The Summarizer is shown P-labels only: F1 and C1 references cannot
    smuggle Findings or per-Finding Citations into the relevance map."""
    grounded, decisions = _validate_summaries(
        Summaries(summaries=[
            ProvisionSummary(ref="F1", relevance="One."),
            ProvisionSummary(ref="C1", relevance="Two."),
        ]),
        {"P1": ProvisionTarget("ai-act", ProvisionKind.article, 6)},
    )
    assert grounded == {}
    assert [d.status for d in decisions] == ["rejected", "rejected"]


def test_one_provision_keeps_only_its_first_relevance_statement():
    """One grounded statement per cited provision: a second statement for the
    same provision is rejected, never merged or overwritten."""
    target = ProvisionTarget("ai-act", ProvisionKind.article, 6)
    grounded, decisions = _validate_summaries(
        Summaries(summaries=[
            ProvisionSummary(ref="P1", relevance="First statement.", strength=Strength.weak),
            ProvisionSummary(ref="P1", relevance="Second statement.", strength=Strength.strong),
        ]),
        {"P1": target},
    )
    assert grounded == {target: _grounded(target, "First statement.", Strength.weak)}
    assert [d.status for d in decisions] == ["kept", "rejected"]


def test_an_empty_relevance_statement_is_rejected():
    target = ProvisionTarget("ai-act", ProvisionKind.article, 6)
    grounded, decisions = _validate_summaries(
        Summaries(summaries=[ProvisionSummary(ref="P1", relevance="   ")]),
        {"P1": target},
    )
    assert grounded == {}
    assert decisions[0].status == "rejected"


def test_distinct_provisions_keep_their_own_statements():
    article_6 = ProvisionTarget("ai-act", ProvisionKind.article, 6)
    recital_71 = ProvisionTarget("gdpr", ProvisionKind.recital, 71)
    grounded, decisions = _validate_summaries(
        Summaries(summaries=[
            ProvisionSummary(ref="P1", relevance="Article statement."),
            ProvisionSummary(ref="P2", relevance="Recital statement."),
        ]),
        {"P1": article_6, "P2": recital_71},
    )
    assert grounded == {
        article_6: _grounded(article_6, "Article statement."),
        recital_71: _grounded(recital_71, "Recital statement."),
    }
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


# --- The Summarizer's rubric (issue #67: the all-strong calibration bias) ---


@pytest.fixture
def rubric() -> dict[str, str]:
    """The one place the recalibrated rubric's load-bearing phrases live:
    each test pins its rule through this mapping, so a wording edit that
    preserves intent updates this dict and nothing else."""
    return {
        "scale": "whole cited set",
        "not in isolation": "isolation",
        "reserve strong": "Reserve 'strong'",
        "strong per glossary": "the obligations the Answer turns on",
        "levels used": "The levels exist to be used",
        "no default upward": "defaulting upward",
        "weighed failure": "all 'strong' because nothing was weighed",
        "honest ratings": "rate each honestly",
        "honest all-strong": "correctly all 'strong'",
        "definitional weak": "definitional or framing",
        "weak however cited": "no matter how",
        "grounding per provision": "never on other provisions",
        "scale not grounding": "the cited set calibrates the scale, never the grounding",
        "named": "Citation strength",
        "three levels": "exactly one of 'strong', 'moderate', ",
        "bare string": "or 'weak', never an object or rationale",
    }


def test_the_rubric_sets_the_scale_against_the_whole_cited_set(rubric):
    """The all-strong bias came from rating each provision in isolation: the
    rubric must set the scale against the whole cited set, so a provision is
    judged relative to what else the Answer cites, not on its own."""
    assert rubric["scale"] in _SUMMARIZER_SYSTEM
    assert rubric["not in isolation"] in _SUMMARIZER_SYSTEM


def test_the_rubric_reserves_strong_for_the_provisions_the_answer_turns_on(rubric):
    """Strong is scarce by definition — the provisions the Answer's
    conclusions stand on — so the rubric must say what strong is reserved
    for, not merely what it means, in the glossary's own words."""
    assert rubric["reserve strong"] in _SUMMARIZER_SYSTEM
    assert rubric["strong per glossary"] in _SUMMARIZER_SYSTEM


def test_the_rubric_makes_moderate_and_weak_actually_occur(rubric):
    """The issue's ask is occurrence, not merely scarcity: the rubric maps
    the levels onto the material — supporting duties rate 'moderate' rather
    than defaulting upward to 'strong', definitional or framing material
    rates 'weak' — so both levels occur whenever their material does, while
    the honesty clause keeps genuine load-bearers 'strong'."""
    assert rubric["levels used"] in _SUMMARIZER_SYSTEM
    assert rubric["no default upward"] in _SUMMARIZER_SYSTEM


def test_the_rubric_names_an_all_strong_ratings_set_as_a_failure(rubric):
    """The exact failure the eval surfaced (~90% strong, run
    glm53-v5-2026-09-03) is named in the rubric as what it is: an
    all-'strong' set produced *without weighing*. The clause stays
    conditional, never categorical — a small cited set that is genuinely
    load-bearing throughout is correctly all 'strong', so the clause cannot
    force the inverse bias (everything moderate/weak)."""
    assert rubric["weighed failure"] in _SUMMARIZER_SYSTEM
    assert rubric["honest ratings"] in _SUMMARIZER_SYSTEM
    assert rubric["honest all-strong"] in _SUMMARIZER_SYSTEM


def test_the_rubric_widens_the_scale_never_the_grounding(rubric):
    """Setting the scale against the whole cited set is calibration, not new
    material: statements and ratings stay grounded in the content of the
    Findings citing their own provision, and other provisions stay out
    (CONTEXT.md; ADR-0011)."""
    assert rubric["grounding per provision"] in _SUMMARIZER_SYSTEM
    assert rubric["scale not grounding"] in _SUMMARIZER_SYSTEM


def test_the_rubric_keeps_weak_for_definitional_or_framing_provisions(rubric):
    """A definitional or framing provision stays 'weak' no matter how often
    its citing Findings lean on its vocabulary — the rubric must hold that
    line explicitly, or heavily-cited definitions rate strong."""
    assert rubric["definitional weak"] in _SUMMARIZER_SYSTEM
    assert rubric["weak however cited"] in _SUMMARIZER_SYSTEM


def test_the_rubric_keeps_the_three_levels_and_the_bare_string_contract(rubric):
    """The recalibration tightens the scale, never the schema: a bare
    strong/moderate/weak string, never an object or rationale (ADR-0011)."""
    assert rubric["named"] in _SUMMARIZER_SYSTEM
    assert rubric["three levels"] in _SUMMARIZER_SYSTEM
    assert rubric["bare string"] in _SUMMARIZER_SYSTEM
