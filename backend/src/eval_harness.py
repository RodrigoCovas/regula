"""Offline evaluation harness for the deterministic demo.

Scores produced Findings against curated ground-truth scenarios using
Strength-weighted precision, recall, and their harmonic mean (F1).

Scoring interpretation: the mis-tag penalty ("a matched Finding retains
weight x (1 - distance/2) of its credit") is applied to the credit that
feeds BOTH the precision and recall numerators — the only reading that
gives the penalty any effect on the score. Spurious Findings subtract
weight(produced strength) from the precision numerator.

Empty-expected Scenarios follow their own locked rule: they score 1.0
only when nothing was produced; any production is spurious leakage and
scores 0.0 across the board. Recall is never defaulted to 1.0 outside
that explicit rule.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, TypedDict

from .models import Strength


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


def strength_distance(a: Strength, b: Strength) -> int:
    """Step count between two Strengths on the ordered scale weak < moderate < strong."""
    return abs(_STRENGTH_ORDER.index(a) - _STRENGTH_ORDER.index(b))


def _matched_finding_credit(expected_strength: Strength, produced_strength: Strength) -> float:
    """Credit a matched Finding retains after the distance-scaled mis-tag penalty."""
    return STRENGTH_WEIGHTS[expected_strength] * (1 - strength_distance(expected_strength, produced_strength) / 2)


def score_scenario(expected: List[ExpectedFinding], produced: List[ProducedFinding]) -> Dict[str, float]:
    """Score one eval scenario: weighted recall, weighted precision (spurious Findings subtract
    credit by the Strength they were produced with), and their harmonic mean.

    A Scenario with no expected Findings scores 1.0 only when nothing was
    produced; any production is spurious leakage and scores 0.0.
    """
    if not expected:
        if not produced:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    produced_by_statement = {p.statement: p for p in produced}
    expected_weight_total = sum(STRENGTH_WEIGHTS[e.strength] for e in expected)
    produced_weight_total = sum(STRENGTH_WEIGHTS[p.strength] for p in produced)

    matched_credit = 0.0
    matched_expected_weight = 0.0
    for e in expected:
        p = produced_by_statement.get(e.statement)
        if p is not None:
            matched_credit += _matched_finding_credit(e.strength, p.strength)
            matched_expected_weight += STRENGTH_WEIGHTS[e.strength]

    spurious_weight = sum(
        STRENGTH_WEIGHTS[p.strength] for p in produced if p.statement not in {e.statement for e in expected}
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
    from .models import AnalyzeRequest, Scenario

    app_settings = main.settings
    main.settings = Settings(regula_mode=Mode.demo)
    try:
        scenario_results: List[EvalScenarioResult] = []
        for scenario in CURATED_SCENARIOS:
            request = AnalyzeRequest(scenario=Scenario(id=scenario.scenario_id), question=scenario.question)
            response = asyncio.run(analyze(request))
            produced = [
                ProducedFinding(statement=f.statement, strength=f.strength)
                for f in response.answer.findings
            ]
            result = score_scenario(scenario.expected, produced)
            scenario_results.append(EvalScenarioResult(id=scenario.id, **result))
    finally:
        main.settings = app_settings

    mean_f1 = sum(scenario_result.f1 for scenario_result in scenario_results) / len(scenario_results)
    return EvalReport(scenarios=scenario_results, mean_f1=mean_f1)


if __name__ == "__main__":
    from dataclasses import asdict

    print(json.dumps(asdict(run_eval()), indent=2))
