"""Offline evaluation harness for the deterministic demo.

Scores produced Findings against curated ground-truth scenarios using
Strength-weighted precision, recall, and their harmonic mean (F1).

Matching is a per-case strategy producing expected→produced pairings; the
Strength-penalty and spurious-subtraction mechanics below are shared by
every matcher. The default ``verbatim_matcher`` pairs by exact statement
equality — the Demo tripwire contract of ADR-0001. ``semantic_matcher``
pairs paraphrases by lexical similarity (token-set cosine over stopword-
stripped, singularized statements), one-to-one, closest pairs first at
SEMANTIC_MATCH_THRESHOLD — step 2 of the ADR's fixed sequence, proven by
synthetic cases until #7's hand-authored ground truths arrive.

A matched Finding's credit also scales by its citation fidelity (step 3):
an expected Citation is hit when a produced Citation targets the same
source and structural provision (kind + number), compared via
``parse_provision`` and ``Citation.provision_target`` — never by label
string. Fidelity is the hit fraction of the union of both target sets, so
missed, mistargeted, and extra produced Citations all lower it; vacuously
1.0 when neither side names any. Demo production derives from the same
locked targets as its ground truth, so demo fidelity is 1.0 while every
expected target still resolves in the Corpus; one that stops resolving
lowers the score loudly — the tripwire doing its job (see ADR-0001's
step-3 update).

Scoring interpretation: the mis-tag penalty ("a matched Finding retains
weight x (1 - distance/2) of its credit") and the citation-fidelity scale
apply to the same matched-Finding credit, which feeds BOTH the precision
and recall numerators — the only reading that gives either mechanism any
effect on the score. Spurious Findings subtract
weight(produced strength) from the precision numerator, and pairing is
one-to-one per the matcher: a repeated production of an already-paired
statement adds no credit and its full weight subtracts as spurious
leakage.

Empty-expected Scenarios follow their own locked rule: they score 1.0
only when nothing was produced; any production is spurious leakage and
scores 0.0 across the board. Recall is never defaulted to 1.0 outside
that explicit rule.
"""

import json
import math
import re
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, TypedDict

from .models import (
    PROVISION_NUMBER_FIELDS,
    AnalyzeRequest,
    AnalyzeResponse,
    Citation,
    ProvisionKind,
    ProvisionTarget,
    Scenario,
    Strength,
)


class ExpectedCitation(TypedDict):
    """Ground-truth reference to one provision: the document and its official provision label."""

    source_id: str
    provision: str

STRENGTH_WEIGHTS: Dict[Strength, int] = {
    Strength.strong: 3,
    Strength.moderate: 2,
    Strength.weak: 1,
}

_STRENGTH_ORDER = [Strength.weak, Strength.moderate, Strength.strong]


@dataclass
class ExpectedFinding:
    statement: str
    strength: Strength
    citations: List[ExpectedCitation] = field(default_factory=list)


@dataclass
class ProducedFinding:
    statement: str
    strength: Strength
    citations: List[Citation] = field(default_factory=list)


def strength_distance(a: Strength, b: Strength) -> int:
    """Step count between two Strengths on the ordered scale weak < moderate < strong."""
    return abs(_STRENGTH_ORDER.index(a) - _STRENGTH_ORDER.index(b))


# Ground-truth provision label grammar per ProvisionKind. The number is the
# document's structural provision number — paragraphs and points like "(2)" or
# "point 5(b)" refine one target and never change which Article/Recital/Annex
# is cited. Keyed by the same kinds PROVISION_NUMBER_FIELDS maps, so a new kind
# must add both a Citation field there and a grammar here.
_PROVISION_PATTERNS: Dict[ProvisionKind, re.Pattern] = {
    ProvisionKind.article: re.compile(r"article\s+(\d+)", re.IGNORECASE),
    ProvisionKind.recital: re.compile(r"recital\s+(\d+)", re.IGNORECASE),
    ProvisionKind.annex: re.compile(r"annex\s+([0-9]+|[ivxlcdm]+)\b", re.IGNORECASE),
}

if set(_PROVISION_PATTERNS) != set(PROVISION_NUMBER_FIELDS):
    raise RuntimeError(
        "every ProvisionKind needs both a Citation number field "
        "(models.PROVISION_NUMBER_FIELDS) and a ground-truth label pattern"
    )

_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}

# Strict Roman-numeral grammar, so a malformed label like "Annex vx" fails
# loudly instead of scoring as some accidental value.
_ROMAN_NUMERAL = re.compile(r"m*(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})", re.IGNORECASE)


def _roman_to_int(numeral: str) -> int:
    """The value of a strictly-formed Roman numeral; anything else fails loudly."""
    if not _ROMAN_NUMERAL.fullmatch(numeral):
        raise ValueError(f"{numeral!r} is not a valid Roman numeral")
    values = [_ROMAN_VALUES[ch] for ch in numeral.lower()]
    return sum(-v if v < nxt else v for v, nxt in zip(values, values[1:] + [0]))


def parse_provision(source_id: str, label: str) -> ProvisionTarget:
    """The structural target a hand-authored ground-truth provision label
    names for the given source.

    Understands the shapes ground truth uses — "Article 6(2)",
    "Article 26(2), (4)", "Recital 71", "Annex III point 5(b)", "Annex 3".
    Anything else is a ground-truth authoring bug and fails loudly.
    """
    for kind in PROVISION_NUMBER_FIELDS:
        match = _PROVISION_PATTERNS[kind].search(label)
        if match:
            raw = match.group(1)
            return ProvisionTarget(source_id, kind, int(raw) if raw.isdigit() else _roman_to_int(raw))
    raise ValueError(
        f"Cannot parse ground-truth provision label {label!r}: expected an "
        "Article/Recital/Annex label such as 'Article 6(2)', 'Recital 71', "
        "or 'Annex III point 5(b)'"
    )


def citation_fidelity(expected: List[ExpectedCitation], produced: List[Citation]) -> float:
    """Structural agreement between the expected and produced Citation targets:
    the hit fraction of their union. A missed, mistargeted, or extra produced
    Citation lowers fidelity; vacuously 1.0 when neither side names any."""
    expected_targets = {
        parse_provision(expectation["source_id"], expectation["provision"])
        for expectation in expected
    }
    produced_targets = {citation.provision_target for citation in produced}
    if not expected_targets and not produced_targets:
        return 1.0
    return len(expected_targets & produced_targets) / len(expected_targets | produced_targets)


def _matched_finding_credit(expected: ExpectedFinding, produced: ProducedFinding) -> float:
    """Credit a matched Finding retains after the distance-scaled mis-tag
    penalty, further scaled by its citation fidelity."""
    fidelity_scale = (
        1 - strength_distance(expected.strength, produced.strength) / 2
    ) * citation_fidelity(expected.citations, produced.citations)
    return STRENGTH_WEIGHTS[expected.strength] * fidelity_scale


# Similarity at or above which two statements count as the same Finding. Only
# synthetic cases calibrate it this iteration; #7's ground truths are the real
# validation.
SEMANTIC_MATCH_THRESHOLD = 0.5

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# Polarity words that must agree before two statements can be the same
# Finding: flipping one flips its meaning while barely moving token overlap.
_NEGATORS = frozenset({"not", "no", "nor", "never", "without", "cannot"})

_STOP_WORDS = frozenset(
    """
    a an the and or of to in on for with as by at from is are be been was were
    that this it its their they them we you your our but if then
    than so such can may must shall will would should do does did have has had
    any all each other into under when where which who whom whose what whether
    only also before after between through during above below up down out off
    over again further once here there own same s t
    """.split()
)


def _singular(word: str) -> str:
    """Naive singular form so plural drift never blocks a lexical match."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _content_tokens(statement: str) -> frozenset:
    """Content tokens of a statement: lowercased, stopword-stripped, singularized."""
    words = _TOKEN_PATTERN.findall(statement.lower())
    return frozenset(_singular(word) for word in words if word not in _STOP_WORDS)


def statement_similarity(a: str, b: str) -> float:
    """Token-set cosine between two statements, gated on negation stance:
    1.0 only for verbatim or re-inflected echoes, 0.0 when exactly one side
    carries a negator (a polarity flip is never the same Finding), and 0.0
    for disjoint vocabulary.

    The gate is scope-free — "not only ... but also" style constructions are
    treated as plain negation; real validation against #7's ground truths
    will tell whether that needs refining.
    """
    left, right = _content_tokens(a), _content_tokens(b)
    if not left or not right:
        return 0.0
    if bool(left & _NEGATORS) != bool(right & _NEGATORS):
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))


Matcher = Callable[[List[ExpectedFinding], List[ProducedFinding]], Dict[int, int]]


def verbatim_matcher(expected: List[ExpectedFinding], produced: List[ProducedFinding]) -> Dict[int, int]:
    """The Demo tripwire pairing: exact statement equality."""
    produced_by_statement = {p.statement: j for j, p in enumerate(produced)}
    return {
        i: produced_by_statement[e.statement]
        for i, e in enumerate(expected)
        if e.statement in produced_by_statement
    }


def semantic_matcher(expected: List[ExpectedFinding], produced: List[ProducedFinding]) -> Dict[int, int]:
    """Pair paraphrased-but-equivalent statements one-to-one, closest pairs
    first; ties break deterministically by position."""
    scored = sorted(
        (
            (statement_similarity(e.statement, p.statement), i, j)
            for i, e in enumerate(expected)
            for j, p in enumerate(produced)
        ),
        key=lambda candidate: (-candidate[0], candidate[1], candidate[2]),
    )
    matches: Dict[int, int] = {}
    taken: set[int] = set()
    for similarity, i, j in scored:
        if similarity >= SEMANTIC_MATCH_THRESHOLD and i not in matches and j not in taken:
            matches[i] = j
            taken.add(j)
    return matches


def score_scenario(
    expected: List[ExpectedFinding],
    produced: List[ProducedFinding],
    *,
    matcher: Optional[Matcher] = None,
) -> Dict[str, float]:
    """Score one eval scenario: weighted recall, weighted precision (spurious
    Findings subtract credit by the Strength they were produced with), and
    their harmonic mean.

    Pairing comes from the matcher — verbatim statement equality by default;
    the mis-tag penalty, the citation-fidelity scale on matched credit, and
    spurious subtraction are the same for every matcher.

    A Scenario with no expected Findings scores 1.0 only when nothing was
    produced; any production is spurious leakage and scores 0.0.
    """
    if not expected:
        if not produced:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pairings = (matcher or verbatim_matcher)(expected, produced)

    expected_weight_total = sum(STRENGTH_WEIGHTS[e.strength] for e in expected)
    produced_weight_total = sum(STRENGTH_WEIGHTS[p.strength] for p in produced)

    matched_credit = sum(
        _matched_finding_credit(expected[i], produced[j]) for i, j in pairings.items()
    )
    matched_produced = set(pairings.values())
    spurious_weight = sum(
        STRENGTH_WEIGHTS[p.strength] for j, p in enumerate(produced) if j not in matched_produced
    )

    recall = matched_credit / expected_weight_total
    if produced_weight_total == 0:
        precision = 1.0
    else:
        precision = max(0.0, matched_credit - spurious_weight) / produced_weight_total
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


# Ground truth for the canonical Spanish fintech demo answer, hand-copied from
# research/DEMO_SCENARIO_FINDINGS.md (Findings 1-5, 8, D1-D3 subset as shipped).
# Deliberately NOT derived from DEMO_FINDING_DEFS: ground truth must stay an
# independent source of truth, or the eval could never disagree with the code.
_CANONICAL_EXPECTED = [
    ExpectedFinding(
        statement="An AI system that evaluates the creditworthiness of natural persons or establishes their credit score is a high-risk AI system under the AI Act, so the full high-risk obligations apply.",
        strength=Strength.strong,
        citations=[{"source_id": "ai-act", "provision": "Article 6(2)"}, {"source_id": "ai-act", "provision": "Annex III point 5(b)"}],
    ),
    ExpectedFinding(
        statement="As deployer, the company must assign human oversight, keep automatically generated logs for at least six months, and inform applicants that they are subject to a high-risk AI system.",
        strength=Strength.strong,
        citations=[{"source_id": "ai-act", "provision": "Article 26"}],
    ),
    ExpectedFinding(
        statement="Before first deployment, the deployer of the creditworthiness system must perform a Fundamental Rights Impact Assessment and notify its results to the market surveillance authority.",
        strength=Strength.strong,
        citations=[{"source_id": "ai-act", "provision": "Article 27"}],
    ),
    ExpectedFinding(
        statement="Applicants subject to a loan decision based on the system's output have a right to a clear and meaningful explanation of the role of the AI system in the decision.",
        strength=Strength.strong,
        citations=[{"source_id": "ai-act", "provision": "Article 86(1)"}],
    ),
    ExpectedFinding(
        statement="GDPR restricts decisions based solely on automated processing, including profiling, that produce legal or similarly significant effects; loan scoring is such a decision, and even the contract-necessity exception still requires human intervention and contest rights.",
        strength=Strength.strong,
        citations=[{"source_id": "gdpr", "provision": "Article 22"}, {"source_id": "gdpr", "provision": "Recital 71"}],
    ),
    ExpectedFinding(
        statement="The credit-scoring processing requires a data protection impact assessment before it starts, because it is a systematic and extensive automated evaluation on which legally effective decisions are based.",
        strength=Strength.strong,
        citations=[{"source_id": "gdpr", "provision": "Article 35"}],
    ),
    ExpectedFinding(
        statement="DORA applies in full only if the company is itself a licensed financial entity; otherwise its main relevance is through ICT third-party risk where the AI is supplied to financial entities.",
        strength=Strength.moderate,
        citations=[{"source_id": "dora", "provision": "Article 2"}],
    ),
    ExpectedFinding(
        statement="DORA governs the digital operational resilience of financial entities, not the substance of credit decisions; credit scoring itself is regulated by the AI Act and GDPR, not DORA.",
        strength=Strength.moderate,
        citations=[{"source_id": "dora", "provision": "Article 1"}],
    ),
    ExpectedFinding(
        statement="The system qualifies as an AI system; the company will be a provider and/or deployer; credit scoring is a form of profiling as cross-referenced into the AI Act.",
        strength=Strength.weak,
        citations=[{"source_id": "ai-act", "provision": "Article 3"}],
    ),
]

_NO_FINDINGS: List[ExpectedFinding] = []


@dataclass
class EvalScenario:
    id: str
    scenario_id: str
    question: str
    expected: List[ExpectedFinding]
    # None selects the verbatim default; Live ground-truth cases pass
    # semantic_matcher once #7's hand-authored expectations exist.
    matcher: Optional[Matcher] = None


@dataclass
class EvalScenarioResult:
    id: str
    precision: float
    recall: float
    f1: float


@dataclass
class EvalReport:
    scenarios: List[EvalScenarioResult]
    mean_f1: float


CURATED_SCENARIOS: List[EvalScenario] = [
    EvalScenario(id="canonical-what-applies", scenario_id="spanish-fintech", question="What regulations apply?", expected=_CANONICAL_EXPECTED),
    EvalScenario(id="canonical-loan-denial", scenario_id="spanish-fintech", question="Would an automated loan denial violate data protection requirements?", expected=_CANONICAL_EXPECTED),
    EvalScenario(id="canonical-spanish-question", scenario_id="spanish-fintech", question="¿Qué regulaciones aplican a nuestro sistema de scoring?", expected=_CANONICAL_EXPECTED),
    EvalScenario(id="non-canonical-other-id", scenario_id="other-scenario", question="Does this loan scoring violate GDPR?", expected=_NO_FINDINGS),
    EvalScenario(id="non-canonical-near-miss-id", scenario_id="spanish-fintech-demo", question="What regulations apply?", expected=_NO_FINDINGS),
    EvalScenario(id="non-canonical-unrelated", scenario_id="gdpr-audit", question="Do we need a records-of-processing register?", expected=_NO_FINDINGS),
    EvalScenario(id="non-canonical-missing-id", scenario_id="", question="What regulations apply?", expected=_NO_FINDINGS),
]

# The Live-quality cases (#20): the canonical cases' hand-authored ground
# truth, but only the cases whose expectations mean something against Live
# mode, where the workflow answers every question. They run with the semantic
# matcher and citation fidelity through the operator CLI
# (backend/src/live_eval.py) — never in CI. The non-canonical tripwires stay
# Demo-only: they pin canonical-id routing, which Live mode deliberately has
# no notion of, so their empty expectations would score 0 by construction and
# measure nothing.
_CANONICAL_CASE_IDS = {"canonical-what-applies", "canonical-loan-denial", "canonical-spanish-question"}

LIVE_EVAL_SCENARIOS: List[EvalScenario] = [
    replace(scenario, matcher=semantic_matcher)
    for scenario in CURATED_SCENARIOS
    if scenario.id in _CANONICAL_CASE_IDS
]


def score_scenario_report(
    scenarios: List[EvalScenario],
    respond: Callable[[AnalyzeRequest], AnalyzeResponse],
) -> EvalReport:
    """Run each Scenario through ``respond`` and score the produced Findings.

    The one scoring loop both runners share: ``respond`` answers one request
    (the Demo harness calls analyze directly, the Live runner crosses HTTP),
    and everything after — Findings mapping, per-case scoring with the case's
    matcher, mean F1 — happens identically for every mode.
    """
    scenario_results: List[EvalScenarioResult] = []
    for scenario in scenarios:
        request = AnalyzeRequest(scenario=Scenario(id=scenario.scenario_id), question=scenario.question)
        response = respond(request)
        produced = [
            ProducedFinding(statement=f.statement, strength=f.strength, citations=f.citations)
            for f in response.answer.findings
        ]
        result = score_scenario(scenario.expected, produced, matcher=scenario.matcher)
        scenario_results.append(EvalScenarioResult(id=scenario.id, **result))

    mean_f1 = sum(scenario_result.f1 for scenario_result in scenario_results) / len(scenario_results)
    return EvalReport(scenarios=scenario_results, mean_f1=mean_f1)


def run_eval() -> EvalReport:
    """Run every curated scenario against the analyze workflow and score the produced Findings.

    Drives the app's analyze entry point directly — offline, no HTTP.
    Per ADR-0001 the curated cases are Demo-mode tripwires, so the harness
    pins Demo mode regardless of how the app is configured and restores the
    app's settings afterwards.
    """
    import asyncio

    from . import main
    from .config import Mode, Settings
    from .main import analyze

    def respond(request: AnalyzeRequest) -> AnalyzeResponse:
        return asyncio.run(analyze(request))

    app_settings = main.settings
    main.settings = Settings(regula_mode=Mode.demo)
    try:
        return score_scenario_report(CURATED_SCENARIOS, respond)
    finally:
        main.settings = app_settings


if __name__ == "__main__":
    from dataclasses import asdict

    print(json.dumps(asdict(run_eval()), indent=2))
