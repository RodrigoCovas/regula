"""The summary-fidelity judge (ADR-0010, issue #50): the rubric seam.

The judge compares provision-aligned expected and produced Provision
relevance — same role, same obligation direction — in one batched call per
case, and a contradiction between the two statements forces the floor
score. The rubric derivation is deterministic application code: the LLM
only classifies, so run-to-run variance is bounded by the verdict schema.
"""

import pytest

from src.eval_judge import (
    JudgeVerdict,
    PairVerdict,
    RelevancePair,
    SummaryFidelityJudge,
    pair_fidelity,
)
from src.llm import LlmError, LlmUnreachableError

from fakes import (
    CONTRADICTION_VERDICT,
    FULL_AGREEMENT_VERDICT,
    POLARITY_FLIP_VERDICT,
    ROLE_MISMATCH_VERDICT,
    ScriptedJudgeLlm,
)


def _pair(ref: str = "P1") -> RelevancePair:
    return RelevancePair(
        ref=ref,
        provision="ai-act Article 6",
        expected="Article 6 is the classification rule that puts the system under the high-risk regime.",
        produced="Article 6(2) brings the company's credit-scoring system into the AI Act's high-risk obligations.",
    )


# --- The rubric derivation: verdict in, fidelity score out ---


def test_full_agreement_scores_one():
    assert pair_fidelity(FULL_AGREEMENT_VERDICT) == 1.0


def test_role_mismatch_scores_half():
    """The produced statement states the right obligation direction but plays
    the wrong role — half the rubric's two dimensions hold."""
    assert pair_fidelity(ROLE_MISMATCH_VERDICT) == 0.5


def test_polarity_flip_scores_half():
    """The produced statement explains the provision's role in the Answer but
    flips the obligation direction — the other rubric dimension alone."""
    assert pair_fidelity(POLARITY_FLIP_VERDICT) == 0.5


def test_both_dimensions_mismatched_score_the_floor():
    verdict = PairVerdict(ref="P1", same_role=False, same_direction=False, contradiction=False)
    assert pair_fidelity(verdict) == 0.0


def test_contradiction_forces_the_floor_even_when_both_dimensions_hold():
    """A contradiction is worse than a mismatched dimension: it overrides any
    would-be score, so an opposite statement can never ride a role/direction
    technicality."""
    assert pair_fidelity(CONTRADICTION_VERDICT) == 0.0


# --- The judge over the provider: one batched call, schema-validated verdicts ---


def _judge_reply(**verdicts: PairVerdict) -> JudgeVerdict:
    return JudgeVerdict(verdicts=list(verdicts.values()))


def test_the_judge_makes_one_batched_call_for_all_pairs():
    """One call per case carries every pair: the judge never makes a
    per-pair request (ADR-0010)."""
    llm = ScriptedJudgeLlm(_judge_reply(
        P1=FULL_AGREEMENT_VERDICT,
        P2=PairVerdict(ref="P2", same_role=True, same_direction=True, contradiction=False),
    ))
    judge = SummaryFidelityJudge(llm)

    records = judge.compare([_pair("P1"), _pair("P2")])

    assert records["P1"]["score"] == 1.0
    assert records["P2"]["score"] == 1.0
    assert len(llm.calls) == 1
    system, user, schema = llm.calls[0]
    assert schema is JudgeVerdict
    # The rubric travels in the system prompt; both pairs travel in the one
    # user message with their expected and produced statements.
    assert "same_role" in system and "same_direction" in system and "contradiction" in system
    for ref in ("P1", "P2"):
        assert f"[{ref}]" in user
    assert "classification rule" in user and "high-risk obligations" in user


def test_the_judge_scores_verdicts_through_the_rubric():
    """The judge derives each pair's score with the same rubric the pure
    seam pins: contradiction floors, mismatch halves, agreement ones."""
    llm = ScriptedJudgeLlm(_judge_reply(
        P1=FULL_AGREEMENT_VERDICT,
        P2=CONTRADICTION_VERDICT.model_copy(update={"ref": "P2"}),
    ))
    records = SummaryFidelityJudge(llm).compare([_pair("P1"), _pair("P2")])
    assert [record["score"] for record in records.values()] == [1.0, 0.0]


def test_a_pair_record_names_its_provision_and_its_rubric_flags():
    """The verdict record that leaves the judge carries the pair's provision
    and the rubric flags behind its score, so a floored or half-scored pair
    names its provision and failure mode wherever the record persists
    (issue #83)."""
    llm = ScriptedJudgeLlm(_judge_reply(
        P1=ROLE_MISMATCH_VERDICT,
        P2=CONTRADICTION_VERDICT.model_copy(update={"ref": "P2"}),
    ))
    records = SummaryFidelityJudge(llm).compare([_pair("P1"), _pair("P2")])

    assert records["P1"] == {
        "ref": "P1",
        "provision": "ai-act Article 6",
        "role_match": False,
        "direction_match": True,
        "contradiction": False,
        "score": 0.5,
    }
    assert records["P2"] == {
        "ref": "P2",
        "provision": "ai-act Article 6",
        "role_match": True,
        "direction_match": True,
        "contradiction": True,
        "score": 0.0,
    }


def test_the_judge_demands_exactly_one_verdict_per_pair():
    llm = ScriptedJudgeLlm(_judge_reply(
        P1=FULL_AGREEMENT_VERDICT,
        P1b=ROLE_MISMATCH_VERDICT,
    ))
    with pytest.raises(LlmError, match="P1"):
        SummaryFidelityJudge(llm).compare([_pair("P1")])


def test_the_judge_rejects_a_verdict_for_an_unknown_ref():
    llm = ScriptedJudgeLlm(_judge_reply(
        P9=FULL_AGREEMENT_VERDICT.model_copy(update={"ref": "P9"}),
    ))
    with pytest.raises(LlmError, match="P9"):
        SummaryFidelityJudge(llm).compare([_pair("P1")])


def test_the_judge_rejects_a_reply_that_skips_a_pair():
    llm = ScriptedJudgeLlm(_judge_reply(
        P2=FULL_AGREEMENT_VERDICT.model_copy(update={"ref": "P2"}),
    ))
    with pytest.raises(LlmError, match="P1"):
        SummaryFidelityJudge(llm).compare([_pair("P1"), _pair("P2")])


def test_the_judge_calls_nothing_over_an_empty_pair_list():
    """No comparable pairs — nothing to judge, no provider call to waste."""
    llm = ScriptedJudgeLlm(_judge_reply())
    assert SummaryFidelityJudge(llm).compare([]) == {}
    assert llm.calls == []


def test_an_unreachable_provider_surfaces_as_the_unreachable_error():
    """The judge does not swallow provider outages: the eval aborts on them."""
    class UnreachableLlm:
        def complete(self, system, user, schema):
            raise LlmUnreachableError("Could not reach the LLM provider at https://x: refused")

    with pytest.raises(LlmUnreachableError):
        SummaryFidelityJudge(UnreachableLlm()).compare([_pair("P1")])


def test_a_malformed_reply_surfaces_as_an_llm_error():
    """A judge reply that is not a valid verdict (schema-invalid) aborts the
    run — it can never be scored."""
    class GarbageLlm:
        def complete(self, system, user, schema):
            raise LlmError("Model 'x' returned JSON that does not match schema JudgeVerdict")

    with pytest.raises(LlmError, match="JudgeVerdict"):
        SummaryFidelityJudge(GarbageLlm()).compare([_pair("P1")])
