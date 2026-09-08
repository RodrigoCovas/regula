"""The summary-fidelity judge (ADR-0010): a strict rubric over Provision
relevance.

The judge is the eval's second LLM (ADR-0010): same configured provider as
the workflow, one batched call per case, and a schema-validated verdict. It
compares provision-aligned expected and produced Provision relevance —
hand-authored ground truth against the Summarizer's answer-wide statements —
under a strict rubric with exactly three verdict flags: ``same_role`` (both
statements explain why the provision matters to the overall Answer),
``same_direction`` (both point obligations the same way), and
``contradiction`` (the two statements cannot both be true).

The LLM only classifies; the fidelity score is derived deterministically
from the verdict — contradiction forces the floor, each rubric dimension
carries half the score — so run-to-run variance is bounded by the schema,
never by a free-form number. A judge reply that skips a pair, repeats one,
or invents a reference is judge misconduct and fails loudly: a verdict that
cannot be trusted must abort the run, never dress silence as a score.
"""

from typing import Dict, List, Protocol, TypedDict

from pydantic import BaseModel

from .llm import Llm, LlmError

# The score a contradiction forces (ADR-0010): an opposite statement is
# maximally infidelitous regardless of any other verdict flag.
FIDELITY_FLOOR = 0.0


class RelevancePair(BaseModel):
    """One provision-aligned pair the judge reads: the ground-truth and
    produced Provision relevance for the same cited provision, under the
    stable reference label the verdict must reply with."""

    ref: str
    provision: str
    expected: str
    produced: str


class PairVerdict(BaseModel):
    """The judge's verdict on one pair, under the strict rubric."""

    ref: str
    same_role: bool
    same_direction: bool
    contradiction: bool


class JudgeVerdict(BaseModel):
    """The schema every judge reply must satisfy (ADR-0010): one verdict per
    compared pair, no more and no fewer."""

    verdicts: List[PairVerdict] = []


class PairFidelityVerdict(TypedDict):
    """One pair's judged outcome as plain data (built by
    ``pair_fidelity_verdict``): the record the result shape persists
    (issue #83), so a floored or half-scored pair names its provision and
    failure mode wherever the artifact or a checkpoint carries it."""

    ref: str
    provision: str
    role_match: bool
    direction_match: bool
    contradiction: bool
    score: float


def pair_fidelity_verdict(pair: RelevancePair, verdict: PairVerdict) -> PairFidelityVerdict:
    """The plain-data record of one judged pair: the pair's stable reference
    and the provision it names, the rubric flags behind the score, and the
    score itself. One builder for every judge — real or scripted — so no
    consumer's record shape can drift from another's."""
    return PairFidelityVerdict(
        ref=pair.ref,
        provision=pair.provision,
        role_match=verdict.same_role,
        direction_match=verdict.same_direction,
        contradiction=verdict.contradiction,
        score=pair_fidelity(verdict),
    )


def pair_fidelity(verdict: PairVerdict) -> float:
    """The deterministic rubric derivation: contradiction forces the floor;
    otherwise the two rubric dimensions (role, obligation direction) carry
    the score equally — full agreement 1.0, one dimension gone 0.5."""
    if verdict.contradiction:
        return FIDELITY_FLOOR
    return (verdict.same_role + verdict.same_direction) / 2


class SummaryJudge(Protocol):
    """What the eval harness consumes: provision-aligned pairs in, one
    verdict record per reference out — the rubric flags plus the derived
    score, so nothing the judge decided is averaged away inside the
    consumer. The real judge crosses the configured provider; tests script
    this seam (parent spec, seam 2)."""

    def compare(self, pairs: List[RelevancePair]) -> Dict[str, PairFidelityVerdict]: ...


def judge_naming_its_failures(judge: SummaryJudge) -> SummaryJudge:
    """The judge with its failures annotated: when an LLM call aborts the run,
    the message must say which call it was — the judge's LlmErrors are
    re-raised prefixed with "judge", so they read differently from the
    workflow stages' own failures under the CLI's single abort message."""
    class _BlamedJudge:
        def compare(self, pairs):
            try:
                return judge.compare(pairs)
            except LlmError as error:
                raise LlmError(f"summary-fidelity judge: {error}") from error

    return _BlamedJudge()


_JUDGE_SYSTEM = (
    "You are the strict judge of a regulatory research assistant's evaluation. You compare "
    "pairs of Provision relevance statements — an expected one (hand-authored ground truth) "
    "and a produced one (the system's output) — each about one provision the Answer cites. "
    "For every pair return one verdict with exactly these fields: ref (the pair's label), "
    "same_role, same_direction, contradiction. "
    "same_role: both statements play the same role — they explain why the provision matters "
    "to the overall Answer, aggregating across the Findings that cite it. A produced statement "
    "that instead restates the provision's own text, relays a single Finding, or serves any "
    "other purpose marks same_role false. "
    "same_direction: both statements point obligations the same way — who must do what, what "
    "applies, what is restricted. A produced statement that flips the obligation direction "
    "(obliges what the expected excuses, permits what the expected restricts, applies what "
    "the expected exempts) marks same_direction false. "
    "contradiction: the two statements cannot both be true — the produced asserts the opposite "
    "of the expected. Mark it only for genuine opposites, never for wording, emphasis, or "
    "level of detail."
)


def _pairs_block(pairs: List[RelevancePair]) -> str:
    blocks = [
        f"[{pair.ref}] {pair.provision}\n"
        f"Expected: {pair.expected}\n"
        f"Produced: {pair.produced}"
        for pair in pairs
    ]
    return "\n\n".join(blocks)


class SummaryFidelityJudge:
    """The real judge: one batched completion over the configured provider,
    verdicts validated against the schema and against the pairs actually sent."""

    def __init__(self, llm: Llm):
        self._llm = llm

    def compare(self, pairs: List[RelevancePair]) -> Dict[str, PairFidelityVerdict]:
        """Verdict every pair under the rubric, keyed by reference.

        No pairs — nothing to judge and no provider call. Otherwise one
        batched call; a reply that fails the schema surfaces as ``LlmError``
        from the client, and a reply that keys its verdicts wrong fails here,
        also as ``LlmError``: either way the caller aborts the run. Each
        record carries the pair's provision and its rubric flags beside the
        derived score, so the verdict escapes with nothing averaged away.
        """
        if not pairs:
            return {}
        verdict = self._llm.complete(system=_JUDGE_SYSTEM, user=_pairs_block(pairs), schema=JudgeVerdict)
        refs = [pair.ref for pair in pairs]
        by_ref: Dict[str, PairVerdict] = {}
        for pair_verdict in verdict.verdicts:
            if pair_verdict.ref not in refs:
                raise LlmError(
                    f"The judge returned a verdict for unknown ref {pair_verdict.ref!r} "
                    f"(expected one of {refs})"
                )
            if pair_verdict.ref in by_ref:
                raise LlmError(f"The judge returned two verdicts for ref {pair_verdict.ref!r}")
            by_ref[pair_verdict.ref] = pair_verdict
        missing = [ref for ref in refs if ref not in by_ref]
        if missing:
            raise LlmError(f"The judge returned no verdict for ref(s) {missing}")
        return {
            pair.ref: pair_fidelity_verdict(pair, by_ref[pair.ref])
            for pair in pairs
        }
