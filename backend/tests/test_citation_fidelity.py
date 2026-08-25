"""Citation-fidelity scorer (spec #9, ticket #13): synthetic known cases only.

Per ADR-0001 step 3, expected Citations ride in ground truth unscored until
real hand-authored expectations arrive; these tests prove the machinery on
synthetic cases. The verbatim demo path is pinned by test_eval_harness.py —
demo citations hit their expectations by construction, so F1 = 1.0 survives.
"""

import pytest

from src.eval_harness import (
    ExpectedCitation,
    ExpectedFinding,
    ProducedFinding,
    citation_fidelity,
    parse_provision,
    semantic_matcher,
    score_scenario,
)
from src.models import Citation, ProvisionKind, ProvisionTarget, Strength


def _citation(source_id: str = "ai-act", **target_fields: int) -> Citation:
    return Citation.model_validate({"source_id": source_id, **target_fields})


# --- Parsing ground-truth provision labels into structural targets ---


def test_article_label_with_paragraph_parses_to_article_number():
    assert parse_provision("ai-act", "Article 6(2)") == ProvisionTarget("ai-act", ProvisionKind.article, 6)


def test_bare_article_label_parses_to_article_number():
    assert parse_provision("ai-act", "Article 26") == ProvisionTarget("ai-act", ProvisionKind.article, 26)


def test_multi_part_article_label_keeps_only_the_article_number():
    assert parse_provision("ai-act", "Article 26(2), (4), (6), (11)") == ProvisionTarget(
        "ai-act", ProvisionKind.article, 26
    )


def test_recital_label_parses_to_recital_number():
    assert parse_provision("gdpr", "Recital 71") == ProvisionTarget("gdpr", ProvisionKind.recital, 71)


def test_annex_label_with_roman_numeral_and_point_parses_to_annex_number():
    assert parse_provision("ai-act", "Annex III point 5(b)") == ProvisionTarget(
        "ai-act", ProvisionKind.annex, 3
    )


def test_annex_label_with_arabic_numeral_parses_to_annex_number():
    assert parse_provision("ai-act", "Annex 3") == ProvisionTarget("ai-act", ProvisionKind.annex, 3)


def test_parsing_ignores_case():
    assert parse_provision("ai-act", "article 86(1)") == ProvisionTarget("ai-act", ProvisionKind.article, 86)


@pytest.mark.parametrize("label", ["Preamble", "point 5(b)", "Annex", "Article", ""])
def test_unparseable_label_raises_a_clear_error(label):
    with pytest.raises(ValueError, match="provision"):
        parse_provision("ai-act", label)


@pytest.mark.parametrize("label", ["Annex vx", "Annex iiii", "Annex il"])
def test_malformed_roman_numeral_fails_loudly(label):
    with pytest.raises(ValueError, match="Roman"):
        parse_provision("ai-act", label)


# --- Structural target of a produced Citation ---


def test_a_citation_exposes_its_structural_provision_target():
    assert _citation(article_number=6).provision_target == ProvisionTarget("ai-act", ProvisionKind.article, 6)
    assert _citation(recital_number=71).provision_target == ProvisionTarget("ai-act", ProvisionKind.recital, 71)
    assert _citation(annex_number=3).provision_target == ProvisionTarget("ai-act", ProvisionKind.annex, 3)


def test_parsed_labels_and_validated_metadata_meet_in_one_type():
    """The expected side (parsed ground-truth labels) and the produced side
    (validated Citation metadata) compare as the same ProvisionTarget."""
    assert parse_provision("gdpr", "Recital 71") == _citation("gdpr", recital_number=71).provision_target


# --- Fidelity of a produced citation set against an expectation set ---


def _expected(source_id: str, provision: str) -> ExpectedCitation:
    return {"source_id": source_id, "provision": provision}


def test_every_expected_citation_hit_scores_full_fidelity():
    expected = [_expected("ai-act", "Article 6(2)"), _expected("ai-act", "Annex III point 5(b)")]
    produced = [_citation(article_number=6), _citation(annex_number=3)]
    assert citation_fidelity(expected, produced) == 1.0


def test_partial_hits_score_the_hit_fraction():
    expected = [_expected("ai-act", "Article 6(2)"), _expected("ai-act", "Annex III point 5(b)")]
    produced = [_citation(article_number=6)]
    assert citation_fidelity(expected, produced) == 0.5


def test_mistargeted_number_earns_nothing():
    expected = [_expected("ai-act", "Article 26")]
    assert citation_fidelity(expected, [_citation(article_number=27)]) == 0.0


def test_mistargeted_source_earns_nothing():
    expected = [_expected("gdpr", "Article 22")]
    assert citation_fidelity(expected, [_citation("ai-act", article_number=22)]) == 0.0


def test_mistargeted_kind_earns_nothing():
    expected = [_expected("gdpr", "Recital 71")]
    assert citation_fidelity(expected, [_citation("gdpr", article_number=71)]) == 0.0


def test_extra_produced_citations_dilute_fidelity():
    """A hallucinated extra target earns nothing for its expected and dilutes
    the hits — fidelity counts the union of both target sets, so a mistargeted
    Citation in the Answer always costs credit."""
    expected = [_expected("ai-act", "Article 6(2)")]
    assert citation_fidelity(expected, [_citation(article_number=6), _citation(article_number=99)]) == 0.5


def test_no_produced_citations_score_zero():
    expected = [_expected("ai-act", "Article 6(2)")]
    assert citation_fidelity(expected, []) == 0.0


def test_labels_naming_the_same_target_are_one_expectation():
    """Distinct-target semantics: two ground-truth labels collapsing onto the
    same structural target are counted once, so an authoring slip cannot
    distort fidelity."""
    expected = [_expected("ai-act", "Article 6(2)"), _expected("ai-act", "Article 6")]
    assert citation_fidelity(expected, []) == 0.0
    assert citation_fidelity(expected, [_citation(article_number=6)]) == 1.0


def test_no_expectations_and_no_production_is_vacuously_full():
    assert citation_fidelity([], []) == 1.0


def test_citation_against_no_expectation_earns_nothing():
    """A produced Citation where ground truth expects none is fully mistargeted."""
    assert citation_fidelity([], [_citation(article_number=6)]) == 0.0


# --- Integration: matched-Finding credit scales by citation fidelity ---

_HIGH_RISK = "Loan scoring is high-risk"


def _strong_expected(*citations: ExpectedCitation) -> ExpectedFinding:
    return ExpectedFinding(statement=_HIGH_RISK, strength=Strength.strong, citations=list(citations))


def _strong_produced(*citations: Citation) -> ProducedFinding:
    return ProducedFinding(statement=_HIGH_RISK, strength=Strength.strong, citations=list(citations))


def test_matched_finding_with_all_citations_hit_scores_one():
    expected = [_strong_expected(_expected("ai-act", "Article 6(2)"))]
    produced = [_strong_produced(_citation(article_number=6))]
    assert score_scenario(expected, produced)["f1"] == 1.0


def test_half_the_citations_hit_halves_the_credit():
    expected = [
        _strong_expected(_expected("ai-act", "Article 6(2)"), _expected("ai-act", "Annex III point 5(b)"))
    ]
    result = score_scenario(expected, [_strong_produced(_citation(article_number=6))])
    assert result["recall"] == 0.5
    assert result["precision"] == 0.5
    assert result["f1"] == 0.5


def test_absent_citations_score_zero_but_keep_the_finding_matched():
    """The Finding itself is real; only its citation credit vanishes."""
    expected = [_strong_expected(_expected("ai-act", "Article 6(2)"))]
    result = score_scenario(expected, [_strong_produced()])
    assert result["recall"] == 0.0
    assert result["precision"] == 0.0


def test_expected_finding_without_citations_is_unaffected():
    """Neither side names a Citation, so fidelity is vacuously 1.0 — the
    strength mechanics rule alone."""
    expected = [ExpectedFinding(statement="Framing definitions", strength=Strength.weak)]
    produced = [ProducedFinding(statement="Framing definitions", strength=Strength.weak)]
    assert score_scenario(expected, produced)["f1"] == 1.0


def test_fidelity_composes_with_semantic_matching():
    """A Live-style case: paraphrase pairing through semantic_matcher plus a
    mistargeted citation drops the credit to zero."""
    expected = [
        ExpectedFinding(
            statement=(
                "An AI system that evaluates the creditworthiness of natural persons or "
                "establishes their credit score is a high-risk AI system under the AI Act."
            ),
            strength=Strength.strong,
            citations=[_expected("ai-act", "Annex III point 5(b)")],
        )
    ]
    paraphrase = ProducedFinding(
        statement=(
            "Credit scoring systems that assess people's creditworthiness count as "
            "high-risk AI systems under the AI Act."
        ),
        strength=Strength.strong,
        citations=[_citation(article_number=5)],
    )
    result = score_scenario(expected, [paraphrase], matcher=semantic_matcher)
    assert result["recall"] == 0.0

    on_target = ProducedFinding(
        statement=paraphrase.statement,
        strength=Strength.strong,
        citations=[_citation(annex_number=3)],
    )
    assert score_scenario(expected, [on_target], matcher=semantic_matcher)["f1"] == 1.0
