"""Semantic matching scorer (spec #9, ticket #12): synthetic known cases only.

Per ADR-0001 the matcher's behaviour on the real curated suite can only be
validated once #7's hand-authored ground truths exist; these tests prove the
machinery with synthetic paraphrases. The verbatim demo path is pinned by
test_eval_harness.py and must stay untouched.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import pytest

import src.eval_harness as eval_harness
import src.main as main
from src.eval_harness import (
    SEMANTIC_MATCH_THRESHOLD,
    EvalScenario,
    ExpectedFinding,
    ProducedFinding,
    semantic_matcher,
    statement_similarity,
    score_scenario,
)
from src.models import AnalyzeResponse, Answer, Finding, Strength, Trace

HIGH_RISK_EXPECTED = (
    "An AI system that evaluates the creditworthiness of natural persons or "
    "establishes their credit score is a high-risk AI system under the AI Act."
)
HIGH_RISK_PARAPHRASE = (
    "Credit scoring systems that assess people's creditworthiness count as "
    "high-risk AI systems under the AI Act."
)
OVERSIGHT_EXPECTED = (
    "As deployer, the company must assign human oversight to the system and "
    "keep the logs the system automatically generates."
)
OVERSIGHT_PARAPHRASE = (
    "The deployer has to provide human supervision and retain automatically "
    "generated logs."
)
UNRELATED = "The company must appoint a data protection officer and keep records."


def test_verbatim_statements_have_full_similarity():
    assert statement_similarity(HIGH_RISK_EXPECTED, HIGH_RISK_EXPECTED) == 1.0


def test_similarity_is_symmetric():
    assert statement_similarity(HIGH_RISK_EXPECTED, HIGH_RISK_PARAPHRASE) == (
        statement_similarity(HIGH_RISK_PARAPHRASE, HIGH_RISK_EXPECTED)
    )


def test_paraphrase_clears_threshold_unrelated_does_not():
    paraphrase = statement_similarity(HIGH_RISK_EXPECTED, HIGH_RISK_PARAPHRASE)
    unrelated = statement_similarity(HIGH_RISK_EXPECTED, UNRELATED)
    assert paraphrase >= SEMANTIC_MATCH_THRESHOLD
    assert unrelated < SEMANTIC_MATCH_THRESHOLD
    assert paraphrase > unrelated


def test_disjoint_statements_score_zero_similarity():
    assert statement_similarity("Providers must keep logs.", "Users may object.") == 0.0


def test_inflection_and_case_do_not_block_similarity():
    assert statement_similarity("obligations apply", "Obligation applies!") == 1.0


def test_semantic_matcher_pairs_one_expected_with_one_produced():
    expected = [
        ExpectedFinding(statement=HIGH_RISK_EXPECTED, strength=Strength.strong),
        ExpectedFinding(statement=OVERSIGHT_EXPECTED, strength=Strength.moderate),
    ]
    produced = [
        ProducedFinding(statement=HIGH_RISK_PARAPHRASE, strength=Strength.strong),
        ProducedFinding(statement=UNRELATED, strength=Strength.weak),
    ]
    matches = semantic_matcher(expected, produced)
    assert matches == {0: 0}


def test_semantic_matcher_gives_each_produced_statement_to_one_expected_only():
    """Two produced statements both above the threshold: the closer one wins,
    whichever position it sits in; the loser stays unmatched (and is scored
    as spurious)."""
    echo_a = HIGH_RISK_PARAPHRASE
    echo_b = (
        "The AI Act classifies systems that establish the credit score of "
        "persons as high-risk."
    )
    expected = [ExpectedFinding(statement=HIGH_RISK_EXPECTED, strength=Strength.strong)]
    assert statement_similarity(HIGH_RISK_EXPECTED, echo_b) > statement_similarity(
        HIGH_RISK_EXPECTED, echo_a
    ) >= SEMANTIC_MATCH_THRESHOLD

    a_first = [
        ProducedFinding(statement=echo_a, strength=Strength.strong),
        ProducedFinding(statement=echo_b, strength=Strength.strong),
    ]
    assert semantic_matcher(expected, a_first) == {0: 1}

    b_first = [
        ProducedFinding(statement=echo_b, strength=Strength.strong),
        ProducedFinding(statement=echo_a, strength=Strength.strong),
    ]
    assert semantic_matcher(expected, b_first) == {0: 0}


def test_produced_statement_paired_with_the_expected_it_fits_best():
    """A produced statement sitting between two expecteds pairs with the
    closer one, leaving the other expected unmatched."""
    oversightish = (
        "Companies deploying high-risk AI systems must assign human oversight "
        "to the system and keep its logs."
    )
    expected = [
        ExpectedFinding(statement=HIGH_RISK_EXPECTED, strength=Strength.strong),
        ExpectedFinding(statement=OVERSIGHT_EXPECTED, strength=Strength.moderate),
    ]
    matches = semantic_matcher(expected, [ProducedFinding(statement=oversightish, strength=Strength.moderate)])
    assert matches == {1: 0}


def test_paraphrased_finding_matches_but_unrelated_does_not():
    expected = [ExpectedFinding(statement=HIGH_RISK_EXPECTED, strength=Strength.strong)]

    paraphrased = score_scenario(
        expected,
        [ProducedFinding(statement=HIGH_RISK_PARAPHRASE, strength=Strength.strong)],
        matcher=semantic_matcher,
    )
    assert paraphrased["f1"] == 1.0

    unrelated_run = score_scenario(
        expected,
        [ProducedFinding(statement=UNRELATED, strength=Strength.moderate)],
        matcher=semantic_matcher,
    )
    assert unrelated_run["recall"] == 0.0
    assert unrelated_run["precision"] == 0.0


def test_strength_mistag_penalties_scale_with_distance_semantically():
    """Distance 0 / 1 / 2 on weak < moderate < strong keeps full / half / no
    credit — the same penalty curve as the verbatim scorer."""
    expected = [ExpectedFinding(statement=HIGH_RISK_EXPECTED, strength=Strength.strong)]
    full = score_scenario(
        expected,
        [ProducedFinding(statement=HIGH_RISK_PARAPHRASE, strength=Strength.strong)],
        matcher=semantic_matcher,
    )
    half = score_scenario(
        expected,
        [ProducedFinding(statement=HIGH_RISK_PARAPHRASE, strength=Strength.moderate)],
        matcher=semantic_matcher,
    )
    none = score_scenario(
        expected,
        [ProducedFinding(statement=HIGH_RISK_PARAPHRASE, strength=Strength.weak)],
        matcher=semantic_matcher,
    )
    assert full["recall"] == 1.0
    assert half["recall"] == 0.5
    assert none["recall"] == 0.0


def test_spurious_findings_subtract_their_produced_weight_semantically():
    expected = [ExpectedFinding(statement=HIGH_RISK_EXPECTED, strength=Strength.strong)]
    matched = ProducedFinding(statement=HIGH_RISK_PARAPHRASE, strength=Strength.strong)

    spurious_weak = score_scenario(
        expected,
        [matched, ProducedFinding(statement=UNRELATED, strength=Strength.weak)],
        matcher=semantic_matcher,
    )
    assert spurious_weak["precision"] == 0.5

    spurious_strong = score_scenario(
        expected,
        [matched, ProducedFinding(statement=UNRELATED, strength=Strength.strong)],
        matcher=semantic_matcher,
    )
    assert spurious_strong["precision"] == 0.0
    assert spurious_strong["recall"] == 1.0


def test_empty_expected_rule_survives_semantic_matching():
    assert score_scenario([], [], matcher=semantic_matcher) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    leaked = score_scenario(
        [],
        [ProducedFinding(statement=HIGH_RISK_PARAPHRASE, strength=Strength.strong)],
        matcher=semantic_matcher,
    )
    assert leaked == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_default_matcher_stays_verbatim():
    """Without an explicit matcher, only verbatim statements match — the demo
    tripwire property is untouched by the semantic machinery."""
    expected = [ExpectedFinding(statement=HIGH_RISK_EXPECTED, strength=Strength.strong)]
    assert score_scenario(expected, [ProducedFinding(statement=HIGH_RISK_EXPECTED, strength=Strength.strong)])["f1"] == 1.0
    paraphrased = score_scenario(
        expected,
        [ProducedFinding(statement=HIGH_RISK_PARAPHRASE, strength=Strength.strong)],
    )
    assert paraphrased["f1"] == 0.0


def test_second_verbatim_copy_of_a_matched_finding_is_spurious():
    """The pairing-based scorer credits each expected once; a repeated
    production of the same statement adds no credit and subtracts its full
    weight as leakage."""
    expected = [ExpectedFinding(statement=HIGH_RISK_EXPECTED, strength=Strength.strong)]
    duplicated = [
        ProducedFinding(statement=HIGH_RISK_EXPECTED, strength=Strength.strong),
        ProducedFinding(statement=HIGH_RISK_EXPECTED, strength=Strength.strong),
    ]
    result = score_scenario(expected, duplicated)
    assert result["recall"] == 1.0
    assert result["precision"] == 0.0


def _canned_response(*findings: Finding) -> AnalyzeResponse:
    return AnalyzeResponse(
        answer=Answer(findings=list(findings), actions=[], citations=[]),
        trace=Trace(workflow="fake", summary="canned"),
    )


@pytest.mark.parametrize("use_semantic,f1", [(True, 1.0), (False, 0.0)])
def test_per_case_matcher_selection_inside_the_harness(monkeypatch, use_semantic, f1):
    """Each scenario scores through its own matcher: a live-style case opts
    into semantic matching and credits the paraphrase; a verbatim case does
    not move off exact equality."""
    async def fake_analyze(_request):
        return _canned_response(Finding(statement=HIGH_RISK_PARAPHRASE, strength=Strength.strong))

    monkeypatch.setattr(main, "analyze", fake_analyze)
    scenario = EvalScenario(
        id="live-style",
        scenario_id="any-scenario",
        question="What applies?",
        expected=[ExpectedFinding(statement=HIGH_RISK_EXPECTED, strength=Strength.strong)],
        matcher=semantic_matcher if use_semantic else None,
    )
    monkeypatch.setattr(eval_harness, "CURATED_SCENARIOS", [scenario])
    report = eval_harness.run_eval()
    assert report.mean_f1 == f1
