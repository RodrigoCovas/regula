"""Provision-coverage scorer (ADR-0010): the pure scoring seam.

Coverage is a deterministic set F1 over Citation targets — provision kind +
number, never label strings. Recall is strength-weighted on the expected
side; precision counts off-target produced targets against itself. Statement
similarity gates nothing: produced-versus-expected statements ride only in
the per-case audit dump (its shape is pinned with the harness suite in
test_eval_harness.py). The file also pins the pure pieces the components
read: the max-rule strengths coverage's recall weights ride, and the
expected-relevance collection the summary-fidelity judge compares against.
"""

import pytest

from src.eval_harness import (
    STRENGTH_WEIGHTS,
    ExpectedFinding,
    ProducedFinding,
    TargetRating,
    coverage_scores,
    expected_relevance_summaries,
    expected_target_ratings,
    expected_target_strengths,
    expected_target_weights,
    format_provision_target,
    parse_provision,
    produced_citation_targets,
    produced_target_strengths,
)
from src.models import Citation, ProvisionKind, ProvisionTarget, Strength


def _expected(strength: Strength, *labels: str, source_id: str = "ai-act") -> ExpectedFinding:
    return ExpectedFinding(
        statement=f"expected finding citing {', '.join(labels) or 'nothing'}",
        citations=[{"source_id": source_id, "provision": label, "strength": strength} for label in labels],
    )


def _produced(strength: Strength, *targets: dict, source_id: str = "ai-act") -> ProducedFinding:
    return ProducedFinding(
        statement=f"produced finding citing {targets or 'nothing'}",
        strength=strength,
        citations=[Citation.model_validate({"source_id": source_id, **target}) for target in targets],
    )


def _article(number: int) -> dict:
    return {"article_number": number}


def _recital(number: int) -> dict:
    return {"recital_number": number}


def _annex(number: int) -> dict:
    return {"annex_number": number}


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


def test_parsed_labels_and_validated_metadata_meet_in_one_type():
    """The expected side (parsed ground-truth labels) and the produced side
    (validated Citation metadata) compare as the same ProvisionTarget."""
    citation = Citation.model_validate({"source_id": "gdpr", "recital_number": 71})
    assert parse_provision("gdpr", "Recital 71") == citation.provision_target


# --- Strength weighting of the expected side ---


def test_a_target_is_weighted_by_its_strongest_rated_citation():
    """Two ground-truth Findings citing one target: the max rule of CONTEXT.md's
    Citation strength — the target carries the strongest rated Citation's weight."""
    weights = expected_target_weights([
        _expected(Strength.weak, "Article 6"),
        _expected(Strength.strong, "Article 6"),
    ])
    assert weights == {ProvisionTarget("ai-act", ProvisionKind.article, 6): STRENGTH_WEIGHTS[Strength.strong]}


def test_each_expected_target_carries_its_own_weight():
    weights = expected_target_weights([
        _expected(Strength.weak, "Article 3"),
        _expected(Strength.strong, "Article 6"),
    ])
    assert weights == {
        ProvisionTarget("ai-act", ProvisionKind.article, 3): STRENGTH_WEIGHTS[Strength.weak],
        ProvisionTarget("ai-act", ProvisionKind.article, 6): STRENGTH_WEIGHTS[Strength.strong],
    }


def test_a_target_cited_by_several_findings_enters_the_denominator_once():
    """Set semantics: a target cited by two weakly-rated Citations weighs the
    weak weight once (the max), never the sum — repeating an expectation
    cannot inflate its worth."""
    weights = expected_target_weights([
        _expected(Strength.weak, "Article 6"),
        _expected(Strength.weak, "Article 6"),
    ])
    assert sum(weights.values()) == STRENGTH_WEIGHTS[Strength.weak]


# --- The expected side's max-rule strengths (coverage's weight source) ---


def test_expected_target_strengths_follow_the_max_rule():
    """Per expected target, the strongest Strength among the ground-truth
    Citations citing it — the operator's per-provision ratings under the
    max-rule, which since ADR-0011 serves the expected side only."""
    strengths = expected_target_strengths([
        _expected(Strength.weak, "Article 6"),
        _expected(Strength.strong, "Article 6"),
    ])
    assert strengths == {ProvisionTarget("ai-act", ProvisionKind.article, 6): Strength.strong}


def test_expected_weights_still_read_the_max_rule_alone():
    """ADR-0011: the max-rule serves the expected side only — coverage's
    strength weights — while the produced side carries its ratings as given."""
    expected = [_expected(Strength.weak, "Article 6"), _expected(Strength.strong, "Article 6")]
    strengths = expected_target_strengths(expected)
    weights = expected_target_weights(expected)
    assert strengths == {ProvisionTarget("ai-act", ProvisionKind.article, 6): Strength.strong}
    assert weights == {ProvisionTarget("ai-act", ProvisionKind.article, 6): STRENGTH_WEIGHTS[Strength.strong]}


# --- The expected side's combined rating: strength + weight in one walk ---


def test_expected_target_ratings_carry_strength_and_weight_together():
    """The one map both coverage's weighted recall and the ledger's missed
    weights read: the max-rule Strength and its weight, derived together so
    no two consumers can disagree about what a target is worth."""
    ratings = expected_target_ratings([
        _expected(Strength.weak, "Article 3"),
        _expected(Strength.strong, "Article 6"),
    ])
    assert ratings == {
        ProvisionTarget("ai-act", ProvisionKind.article, 3): TargetRating(
            Strength.weak, STRENGTH_WEIGHTS[Strength.weak]
        ),
        ProvisionTarget("ai-act", ProvisionKind.article, 6): TargetRating(
            Strength.strong, STRENGTH_WEIGHTS[Strength.strong]
        ),
    }


def test_expected_target_weights_read_the_ratings_map():
    weights = expected_target_weights([_expected(Strength.moderate, "Article 6")])
    assert weights == {
        ProvisionTarget("ai-act", ProvisionKind.article, 6): STRENGTH_WEIGHTS[Strength.moderate]
    }


# --- The produced side's target set and given ratings (scorer + ledger) ---


def _rated_produced(number: int, rating: Strength) -> ProducedFinding:
    """One produced Finding whose Citation carries the Summarizer's rating —
    the shape an artifact's rebuilt dump has."""
    return ProducedFinding(
        statement="produced finding with a rated citation",
        strength=Strength.weak,
        citations=[Citation.model_validate({
            "source_id": "ai-act", "article_number": number, "strength": rating,
        })],
    )


def test_produced_citation_targets_form_a_set():
    produced = [
        _produced(Strength.strong, _article(6)),
        _produced(Strength.moderate, _article(6)),
        _produced(Strength.weak, _article(3)),
    ]
    assert produced_citation_targets(produced) == {
        ProvisionTarget("ai-act", ProvisionKind.article, 6),
        ProvisionTarget("ai-act", ProvisionKind.article, 3),
    }


def test_produced_target_strengths_read_the_answer_rating_as_given():
    """ADR-0011: the produced side's ratings are read as given — strongest
    per target only where a dump disagrees, never derived from the Findings.
    Display material for the ledger's accounting; no score consumes it."""
    produced = [
        _rated_produced(6, Strength.weak),
        _rated_produced(6, Strength.strong),
        _rated_produced(3, Strength.moderate),
    ]
    assert produced_target_strengths(produced) == {
        ProvisionTarget("ai-act", ProvisionKind.article, 6): Strength.strong,
        ProvisionTarget("ai-act", ProvisionKind.article, 3): Strength.moderate,
    }


def test_unrated_produced_citations_name_no_strength():
    """The in-memory produced side carries unrated Citations (the Answer's
    ratings attach later, through the Summarizer): no target enters the map."""
    assert produced_target_strengths([_produced(Strength.strong, _article(6))]) == {}


# --- The expected relevance summaries the judge reads (ADR-0010, issue #50) ---


def test_expected_relevance_summaries_key_by_parsed_target():
    expected = [
        ExpectedFinding(
            statement="expected finding",
            citations=[{
                "source_id": "gdpr",
                "provision": "Article 22",
                "relevance": "Article 22 restricts the solely automated loan decision.",
                "strength": Strength.strong,
            }],
        ),
    ]
    summaries = expected_relevance_summaries(expected)
    assert summaries == {
        ProvisionTarget("gdpr", ProvisionKind.article, 22): "Article 22 restricts the solely automated loan decision.",
    }


def test_expected_citations_without_relevance_are_ignored():
    """The judge only compares provisions ground truth actually summarizes —
    labels are authored once, in #51, and never required by the scorer."""
    expected = [_expected(Strength.strong, "Article 6")]
    assert expected_relevance_summaries(expected) == {}


def test_conflicting_ground_truth_summaries_fail_loudly():
    """Two expected Findings citing one target with different summaries is an
    authoring bug: it must never be silently last-wins."""
    expected = [
        ExpectedFinding(
            statement="first",
            citations=[{"source_id": "ai-act", "provision": "Article 6", "relevance": "one story", "strength": Strength.strong}],
        ),
        ExpectedFinding(
            statement="second",
            citations=[{"source_id": "ai-act", "provision": "Article 6(2)", "relevance": "another story", "strength": Strength.moderate}],
        ),
    ]
    with pytest.raises(ValueError, match="Article 6"):
        expected_relevance_summaries(expected)


def test_identical_duplicate_summaries_deduplicate():
    expected = [
        ExpectedFinding(
            statement="first",
            citations=[{"source_id": "ai-act", "provision": "Article 6", "relevance": "same story", "strength": Strength.strong}],
        ),
        ExpectedFinding(
            statement="second",
            citations=[{"source_id": "ai-act", "provision": "Article 6(2)", "relevance": "same story", "strength": Strength.moderate}],
        ),
    ]
    assert expected_relevance_summaries(expected) == {
        ProvisionTarget("ai-act", ProvisionKind.article, 6): "same story",
    }


# --- Coverage arithmetic ---


def test_perfect_coverage_scores_one():
    expected = [_expected(Strength.strong, "Article 6", "Annex III point 5(b)")]
    produced = [_produced(Strength.strong, _article(6), _annex(3))]
    assert coverage_scores(expected, produced) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_missed_expected_target_lowers_recall_by_its_weight():
    expected = [_expected(Strength.strong, "Article 6"), _expected(Strength.weak, "Article 3")]
    produced = [_produced(Strength.strong, _article(6))]
    result = coverage_scores(expected, produced)
    assert result["recall"] == pytest.approx(50 / 51)


def test_missing_a_strong_target_hurts_more_than_missing_a_weak_one():
    expected = [_expected(Strength.strong, "Article 6"), _expected(Strength.weak, "Article 3")]
    miss_strong = coverage_scores(expected, [_produced(Strength.weak, _article(3))])
    miss_weak = coverage_scores(expected, [_produced(Strength.strong, _article(6))])
    assert miss_strong["recall"] < miss_weak["recall"]


def test_off_target_produced_citation_counts_against_precision():
    """A produced target outside the expected set earns nothing and dilutes
    precision — pessimistic by construction (ADR-0010): it is not necessarily
    wrong, and the audit reviews the spurious list before numbers are quoted."""
    expected = [_expected(Strength.strong, "Article 6")]
    result = coverage_scores(expected, [_produced(Strength.strong, _article(6), _article(99))])
    assert result["recall"] == 1.0
    assert result["precision"] == 0.5


def test_mistargeted_provision_earns_nothing():
    """Same number, wrong kind or wrong source: a different structural target."""
    expected = [_expected(Strength.strong, "Recital 71", source_id="gdpr")]
    wrong_kind = coverage_scores(expected, [_produced(Strength.strong, _article(71), source_id="gdpr")])
    wrong_source = coverage_scores(expected, [_produced(Strength.strong, _recital(71))])
    assert wrong_kind["precision"] == 0.0
    assert wrong_kind["recall"] == 0.0
    assert wrong_source["f1"] == 0.0


def test_produced_targets_form_a_set_repeats_earn_nothing():
    """Several produced Findings citing the same target: one hit, and only one
    entry in the precision denominator."""
    expected = [_expected(Strength.strong, "Article 6")]
    produced = [_produced(Strength.strong, _article(6)), _produced(Strength.moderate, _article(6))]
    result = coverage_scores(expected, produced)
    assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_f1_is_the_harmonic_mean_of_precision_and_recall():
    expected = [_expected(Strength.strong, "Article 6"), _expected(Strength.strong, "Article 27")]
    result = coverage_scores(expected, [_produced(Strength.strong, _article(6), _article(99))])
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)


def test_nothing_produced_is_vacuous_full_precision_at_zero_recall():
    expected = [_expected(Strength.strong, "Article 6")]
    result = coverage_scores(expected, [])
    assert result == {"precision": 1.0, "recall": 0.0, "f1": 0.0}


def test_empty_expected_nothing_produced_scores_one():
    """A Scenario expecting no Findings and producing none is a perfect run."""
    assert coverage_scores([], []) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_empty_expected_with_leakage_scores_zero():
    """Demo Findings leaking into a foreign Scenario must score zero, not perfect."""
    result = coverage_scores([], [_produced(Strength.strong, _article(6))])
    assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_expectation_without_citations_is_a_loud_zero():
    """A ground-truth Finding naming no provision can never be covered — the
    authoring bug scores zero instead of passing silently."""
    expected = [ExpectedFinding(statement="no provisions named")]
    assert coverage_scores(expected, [])["f1"] == 0.0
    assert coverage_scores(expected, [_produced(Strength.strong, _article(6))])["f1"] == 0.0


# --- Statements never gate the score ---


def test_scores_ignore_statement_text_entirely():
    """ADR-0010: statement similarity left the scoring path. Production may
    word its Findings however it likes — only the Citation targets grade."""
    expected = [
        ExpectedFinding(
            statement="The completely unrelated produced wording below still covers this provision",
            citations=[{"source_id": "gdpr", "provision": "Article 22", "strength": Strength.strong}],
        )
    ]
    produced = [_produced(Strength.weak, _article(22), source_id="gdpr")]
    assert coverage_scores(expected, produced) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


# --- The audit dump's rendering of a structural target ---


def test_provision_target_renders_for_the_audit():
    assert format_provision_target(ProvisionTarget("ai-act", ProvisionKind.article, 6)) == "ai-act Article 6"
    assert format_provision_target(ProvisionTarget("gdpr", ProvisionKind.recital, 71)) == "gdpr Recital 71"
    assert format_provision_target(ProvisionTarget("ai-act", ProvisionKind.annex, 3)) == "ai-act Annex 3"
