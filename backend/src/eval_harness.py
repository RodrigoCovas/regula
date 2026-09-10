"""Offline evaluation harness.

Scoring is the two never-blended components of ADR-0010, reported
separately per case and in aggregate:

1. Provision coverage — a deterministic set F1 over Citation targets
   (provision kind + number via ``parse_provision`` and
   ``Citation.provision_target``, never label strings), with the expected
   side weighted by the operator's per-provision Strength ratings on its
   Citations and every off-target produced target counted against precision.
2. Summary fidelity — the strict rubric judge (``eval_judge.SummaryFidelityJudge``,
   same configured provider, one batched call per case, schema-validated
   verdict) compares provision-aligned expected and produced Provision
   relevance; a contradiction forces the floor score. The component is
   measured where ground truth summarizes a provision (the labels #51
   authored for the Live cases) and a judge is supplied; a cited provision
   the Summarizer left bare scores the floor deterministically, with no
   judge call.

The operator's per-provision ratings carry no comparison metric: they feed
coverage's recall weights only, while the produced Citation-strength ratings
stay on the Citations and in the audit dump for the human review — never
scored against the expected side.

Statements are never matched: statement similarity left the scoring path
with ADR-0010 (ADR-0001's updates record the retirement as deletion), and
survives only as diagnostic material — the per-case audit dump
(``EvalScenarioResult.expected`` / ``.produced``) carries both sides'
statements and relevance summaries for the human review the Live eval CLI
writes out. Beside the scores ride the persistence fields (issue #83):
every judged pair's fidelity verdict — provision, rubric flags, score —
and the response's degraded-coverage Known limitation verbatim, so the
per-pair data the judge decided is never averaged away inside the scorer.

Demo pinning (ADR-0001/0008): the curated cases are Demo-mode tripwires.
Demo production derives its Citations from the same locked targets its
ground truth transcribes, so coverage F1 is 1.0 by construction while every
expected target still resolves in the Corpus — one that stops resolving
lowers the score loudly. Empty-expected Scenarios follow their own locked
rule: they score 1.0 only when nothing was produced; any production is
spurious leakage and scores 0.0 across the board.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, NamedTuple, NotRequired, Optional, Set, TypedDict

from .eval_judge import FIDELITY_FLOOR, PairFidelityVerdict, RelevancePair, SummaryJudge
from .models import (
    PROVISION_NOUNS,
    PROVISION_NUMBER_FIELDS,
    AnalyzeRequest,
    AnalyzeResponse,
    Citation,
    Mode,
    ProvisionKind,
    ProvisionTarget,
    Scenario,
    Strength,
    max_rule_strengths,
)


class ExpectedCitation(TypedDict):
    """Ground-truth reference to one provision: the document and its official
    provision label, the hand-written relevance summary the fidelity judge
    compares against (authored in #51), and the operator's per-provision
    Strength rating transcribed alongside them (#56)."""

    source_id: str
    provision: str
    relevance: NotRequired[str]
    strength: Strength

# The eval's one numeric weight map (#55): strong = 50, moderate = 5, weak =
# 1, so missing a provision that names the situation outright costs fifty
# times a missed framing provision. It feeds recall only — precision stays
# unweighted.
STRENGTH_WEIGHTS: Dict[Strength, int] = {
    Strength.strong: 50,
    Strength.moderate: 5,
    Strength.weak: 1,
}


@dataclass
class ExpectedFinding:
    """One ground-truth expectation: a hand-authored statement grouping the
    expected Citations. Strengths live on the Citations — the operator's
    per-provision ratings — never on the Finding (#58)."""

    statement: str
    citations: List[ExpectedCitation] = field(default_factory=list)


@dataclass
class ProducedFinding:
    statement: str
    strength: Strength
    citations: List[Citation] = field(default_factory=list)


class CoverageScores(TypedDict):
    """The per-case coverage triple: precision, recall, and their harmonic mean."""

    precision: float
    recall: float
    f1: float


class ExpectedCitationDump(TypedDict):
    """One expected Citation's audit-dump entry: the ground-truth label as
    written, the authored relevance summary when #51 has written one, and the
    operator's per-provision rating (ADR-0011)."""

    label: str
    relevance: Optional[str]
    strength: str


class ExpectedFindingDump(TypedDict):
    """One expected Finding's audit-dump entry: the authored statement and the
    ground-truth provision labels as written. No Finding-level Strength — the
    operator's per-provision ratings ride the Citations (#58), on both sides
    of the dump since ADR-0011."""

    statement: str
    citations: List[ExpectedCitationDump]


class ProducedCitationDump(TypedDict):
    """One produced Citation's audit-dump entry: the structural target, the
    quoted evidence if any, the Provision relevance the Answer carries, and
    the rated Citation strength — ``None`` where the Summarizer shipped no
    rating (ADR-0011)."""

    target: str
    quote: Optional[str]
    relevance: Optional[str]
    strength: Optional[str]


class ProducedFindingDump(TypedDict):
    """One produced Finding's audit-dump entry: the statement as the LLM
    worded it, its Strength (finding display's own axis), and each Citation's
    structural target with its rating."""

    statement: str
    strength: str
    citations: List[ProducedCitationDump]


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


def expected_target_strengths(expected: List[ExpectedFinding]) -> Dict[ProvisionTarget, Strength]:
    """The expected side's Citation strength under the max-rule (CONTEXT.md):
    per expected Citation target, the strongest Strength among the
    ground-truth Citations citing it — each Citation's own Strength, the
    operator's per-provision rating transcribed in #56 and the single source
    of expected Citation strength since #58. The one remaining role (ADR-0011
    retired the max-rule from the produced side): coverage's strength
    weights.
    """
    return max_rule_strengths(
        (parse_provision(citation["source_id"], citation["provision"]), citation["strength"])
        for finding in expected
        for citation in finding.citations
    )


class TargetRating(NamedTuple):
    """One expected Citation target's rating: the operator's Strength and the
    recall weight that Strength carries — the two faces of the one numeric
    weight map, always read together."""

    strength: Strength
    weight: int


def expected_target_ratings(expected: List[ExpectedFinding]) -> Dict[ProvisionTarget, TargetRating]:
    """Each expected Citation target with its operator rating and recall
    weight in one walk of the max-rule (CONTEXT.md). The single source both
    coverage's weighted recall and the ledger's missed-list weights read, so
    no two consumers can disagree about what an expected target is worth."""
    return {
        target: TargetRating(strength, STRENGTH_WEIGHTS[strength])
        for target, strength in expected_target_strengths(expected).items()
    }


def expected_target_weights(expected: List[ExpectedFinding]) -> Dict[ProvisionTarget, int]:
    """Weight each expected Citation target by the strongest Strength among
    the ground-truth Citations citing it — the operator's per-provision
    ratings under CONTEXT.md's max-rule, applied to the expected side through
    the one shared max-rule implementation (models.max_rule_strengths). A
    target cited by several Findings enters once (set semantics); a Finding
    citing no target contributes nothing, so an expectation that names no
    provision can never be covered."""
    return {target: rating.weight for target, rating in expected_target_ratings(expected).items()}


def expected_relevance_summaries(expected: List[ExpectedFinding]) -> Dict[ProvisionTarget, str]:
    """The ground-truth Provision relevance summaries, keyed by the structural
    target each label parses to. Only targets #51's hand-authored summaries
    cover take part in summary fidelity; two Findings citing one target must
    agree on its summary — a conflicting pair is an authoring bug that fails
    loudly instead of silently last-wins."""
    summaries: Dict[ProvisionTarget, str] = {}
    for finding in expected:
        for citation in finding.citations:
            relevance = citation.get("relevance")
            if not relevance:
                continue
            target = parse_provision(citation["source_id"], citation["provision"])
            existing = summaries.get(target)
            if existing is not None and existing != relevance:
                raise ValueError(
                    f"Conflicting ground-truth relevance summaries for "
                    f"{format_provision_target(target)}"
                )
            summaries[target] = relevance
    return summaries


def format_provision_target(target: ProvisionTarget) -> str:
    """The deterministic audit-dump form of a structural target: the source
    and its provision kind + number — never the free-text label."""
    return f"{target.source_id} {PROVISION_NOUNS[target.kind]} {target.number}"


def format_score(score: Optional[float]) -> str:
    """Three decimals for a measured score, ``n/a`` for one that is not —
    an unmeasured component must never dress up as a zero."""
    return f"{score:.3f}" if score is not None else "n/a"


def produced_citation_targets(produced: List[ProducedFinding]) -> Set[ProvisionTarget]:
    """The produced side's unique Citation targets (set semantics): repeating
    one earns nothing extra, wherever the walk starts — the scorer and the
    ledger read the same set."""
    return {citation.provision_target for finding in produced for citation in finding.citations}


def produced_target_strengths(produced: List[ProducedFinding]) -> Dict[ProvisionTarget, Strength]:
    """The produced side's rated Citation strengths exactly as given
    (ADR-0011): per target, the rating the Answer carries — the artifact
    holds one answer-wide rating per target, so the strongest-wins walk only
    deduplicates should a dump ever disagree. Display material for the
    ledger's accounting; no score consumes it."""
    return max_rule_strengths(
        (citation.provision_target, citation.strength)
        for finding in produced
        for citation in finding.citations
        if citation.strength is not None
    )


def coverage_scores(
    expected: List[ExpectedFinding],
    produced: List[ProducedFinding],
) -> CoverageScores:
    """Provision-coverage F1 (ADR-0010): set arithmetic over Citation targets.

    Recall is the strength-weighted share of expected targets the produced
    side covers: each expected target carries the weight of its
    strongest-rated expected Citation — the operator's per-provision rating —
    and covering it anywhere in production counts in full.
    Precision is the fraction of produced targets that hit an expected one —
    every off-target produced Citation counts against it (pessimistic by
    construction: the audit reviews the spurious list before numbers are
    quoted). Produced targets form a set: repeating one earns nothing extra.
    Nothing produced is vacuously full precision (nothing spurious) at zero
    recall.

    The locked empty rule: no expected Findings scores 1.0 only when nothing
    was produced; any production is spurious leakage and scores 0.0 across
    the board. Expected Findings naming no provision can never be covered
    and score loudly zero.
    """
    if not expected:
        if not produced:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    weights = expected_target_weights(expected)
    expected_weight_total = sum(weights.values())
    produced_targets = produced_citation_targets(produced)

    hit_weight = sum(w for target, w in weights.items() if target in produced_targets)
    recall = hit_weight / expected_weight_total if expected_weight_total else 0.0
    precision = (
        len(weights.keys() & produced_targets) / len(produced_targets) if produced_targets else 1.0
    )
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
        citations=[{"source_id": "ai-act", "provision": "Article 6(2)", "strength": Strength.strong}, {"source_id": "ai-act", "provision": "Annex III point 5(b)", "strength": Strength.strong}],
    ),
    ExpectedFinding(
        statement="As deployer, the company must assign human oversight, keep automatically generated logs for at least six months, and inform applicants that they are subject to a high-risk AI system.",
        citations=[{"source_id": "ai-act", "provision": "Article 26", "strength": Strength.strong}],
    ),
    ExpectedFinding(
        statement="Before first deployment, the deployer of the creditworthiness system must perform a Fundamental Rights Impact Assessment and notify its results to the market surveillance authority.",
        citations=[{"source_id": "ai-act", "provision": "Article 27", "strength": Strength.strong}],
    ),
    ExpectedFinding(
        statement="Applicants subject to a loan decision based on the system's output have a right to a clear and meaningful explanation of the role of the AI system in the decision.",
        citations=[{"source_id": "ai-act", "provision": "Article 86(1)", "strength": Strength.strong}],
    ),
    ExpectedFinding(
        statement="GDPR restricts decisions based solely on automated processing, including profiling, that produce legal or similarly significant effects; loan scoring is such a decision, and even the contract-necessity exception still requires human intervention and contest rights.",
        citations=[{"source_id": "gdpr", "provision": "Article 22", "strength": Strength.strong}, {"source_id": "gdpr", "provision": "Recital 71", "strength": Strength.strong}],
    ),
    ExpectedFinding(
        statement="The credit-scoring processing requires a data protection impact assessment before it starts, because it is a systematic and extensive automated evaluation on which legally effective decisions are based.",
        citations=[{"source_id": "gdpr", "provision": "Article 35", "strength": Strength.strong}],
    ),
    ExpectedFinding(
        statement="DORA applies in full only if the company is itself a licensed financial entity; otherwise its main relevance is through ICT third-party risk where the AI is supplied to financial entities.",
        citations=[{"source_id": "dora", "provision": "Article 2", "strength": Strength.moderate}],
    ),
    ExpectedFinding(
        statement="DORA governs the digital operational resilience of financial entities, not the substance of credit decisions; credit scoring itself is regulated by the AI Act and GDPR, not DORA.",
        citations=[{"source_id": "dora", "provision": "Article 1", "strength": Strength.moderate}],
    ),
    ExpectedFinding(
        statement="The system qualifies as an AI system; the company will be a provider and/or deployer; credit scoring is a form of profiling as cross-referenced into the AI Act.",
        citations=[{"source_id": "ai-act", "provision": "Article 3", "strength": Strength.weak}],
    ),
]

_NO_FINDINGS: List[ExpectedFinding] = []


@dataclass
class EvalScenario:
    id: str
    scenario_id: str
    question: str
    expected: List[ExpectedFinding]
    # The company context the request carries alongside the question. Demo
    # pins routing by scenario id and ignores it; Live mode grounds the
    # Planner on it (live_workflow reads description/title).
    description: Optional[str] = None


@dataclass
class EvalScenarioResult:
    """One case's two component scores plus its audit dump and persistence
    fields.

    Coverage (precision/recall/F1) and summary fidelity are separate numbers
    (ADR-0010) — nothing blends them. A component reads ``None`` where it
    measured nothing: no authored relevance summaries or no judge for
    fidelity. The expected/produced dumps carry both sides as plain data
    (statements, provision targets, relevance summaries, the operator's
    ratings, the produced side's Finding Strengths and the rated Citation
    strengths) for the human review the Live eval CLI writes out.

    The persistence fields ride beside the scores, never inside them
    (issue #83): ``fidelity_verdicts`` holds every judged pair's verdict
    record (provision, rubric flags, score) in pair order — empty where no
    pair was judged, never fabricated — and ``summarizer_limitation`` the
    response's Provision-relevance Known limitation verbatim (``None``
    where coverage was complete), so degraded Summarizer coverage stays
    visible after the run.
    """

    id: str
    precision: float
    recall: float
    f1: float
    expected: List[ExpectedFindingDump]
    produced: List[ProducedFindingDump]
    summary_fidelity: Optional[float] = None
    fidelity_verdicts: List[PairFidelityVerdict] = field(default_factory=list)
    summarizer_limitation: Optional[str] = None


@dataclass
class EvalReport:
    """The aggregate half of the report: one mean per component, over the
    cases where that component measured — never a blended overall score."""

    scenarios: List[EvalScenarioResult]
    mean_f1: float
    mean_summary_fidelity: Optional[float] = None


CURATED_SCENARIOS: List[EvalScenario] = [
    EvalScenario(id="canonical-what-applies", scenario_id="spanish-fintech-startup-uses-9e165169", question="What regulations apply?", expected=_CANONICAL_EXPECTED),
    EvalScenario(id="canonical-loan-denial", scenario_id="spanish-fintech-startup-uses-9e165169", question="Would an automated loan denial violate data protection requirements?", expected=_CANONICAL_EXPECTED),
    EvalScenario(id="canonical-spanish-question", scenario_id="spanish-fintech-startup-uses-9e165169", question="¿Qué regulaciones aplican a nuestro sistema de scoring?", expected=_CANONICAL_EXPECTED),
    EvalScenario(id="non-canonical-other-id", scenario_id="other-scenario", question="Does this loan scoring violate GDPR?", expected=_NO_FINDINGS),
    EvalScenario(id="non-canonical-near-miss-id", scenario_id="spanish-fintech-demo", question="What regulations apply?", expected=_NO_FINDINGS),
    EvalScenario(id="non-canonical-unrelated", scenario_id="gdpr-audit", question="Do we need a records-of-processing register?", expected=_NO_FINDINGS),
    EvalScenario(id="non-canonical-missing-id", scenario_id="", question="What regulations apply?", expected=_NO_FINDINGS),
]

# The Live-quality cases (#20, #51): the operator CLI's case list
# (backend/src/live_eval.py), never part of CI. Each case describes its own
# Scenario (the description grounds the Live Planner; Demo would refuse these
# ids as unknown). Ground truth is hand-authored: every cited provision's
# relevance summary and per-provision Strength rating are transcribed
# verbatim from the operator's data/regulations/citations.json (#51, #56),
# and the Findings grouping those citations are hand-authored per case under
# the #7 precedent — statements written once, never derived from or validated
# against pipeline output, so the eval can disagree with the code; the
# Strengths ride the Citations alone (#58). The authoring conventions
# (clustering, range expansion, sub-reference collapse, the Strength rubric)
# live with the ground-truth record in data/PROVENANCE.md.
_RETAILER_BREACH_EXPECTED = [
    ExpectedFinding(
        statement="Confirmed unauthorized access to a database of customers' names, email addresses, postal addresses and order information is a personal data breach in its own right - the download question feeds the risk assessment, not the breach's existence.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 4",
             "relevance": "Article 4(12) makes the confirmed unauthorized access itself a personal data breach, so the notification analysis stands on the breach's existence while the download question feeds the risk assessment, and Article 4(1) confirms the accessed names, addresses, and order data are personal data.",
             "strength": Strength.strong},
        ],
    ),
    ExpectedFinding(
        statement="The retailer must notify the competent supervisory authority of the breach within 72 hours of becoming aware of the intrusion, phasing in information while the exfiltration investigation continues and documenting the breach so the authority can verify it.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 33",
             "relevance": "Article 33 drives the core response: the retailer must notify the supervisory authority within 72 hours of becoming aware of the intrusion, subject to the low-risk exception, while Article 33(3)-(5) supply the content, the phased-information mechanism for the ongoing exfiltration investigation, and the documentation the authority can verify.",
             "strength": Strength.strong},
        ],
    ),
    ExpectedFinding(
        statement="Affected customers must be told about the breach promptly where the exposed contact and order data are likely to result in a high risk to them, unless the data were encrypted or subsequent measures remove the risk.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 34",
             "relevance": "Article 34 obliges the retailer to promptly communicate the breach to affected customers where the exposed contact and order data create high risk, with Article 34(3)'s encryption and subsequent-measures conditions providing the recognized routes to lifting that communication duty.",
             "strength": Strength.strong},
            {"source_id": "gdpr", "provision": "Article 12",
             "relevance": "Article 12 sets the concise, clear-and-plain form in which the retailer communicates the breach to affected customers and handles their subsequent requests, the modality complement to the Article 34 communication duty.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The compromised customer data must have been secured with measures proportionate to the risk, including against unauthorized access, with the ability to restore availability and access - and the incident tests whether they were adequate.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 32",
             "relevance": "Article 32 obliges the retailer to implement encryption, availability, restoration, and testing measures proportionate to the risk, with Article 32(2) expressly listing unauthorized access among the risks to assess, framing both the incident's cause and its remediation.",
             "strength": Strength.moderate},
        ],
    ),
    ExpectedFinding(
        statement="The processing must respect the integrity and confidentiality principle, the retailer must be able to demonstrate compliance through documented measures, and its processing records must document the security measures the breached safeguards are assessed against.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 5",
             "relevance": "Article 5(1)(f)'s integrity-and-confidentiality principle grounds the retailer's duty to keep the compromised customer data secure and frames the post-incident hardening of its database protections.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 24",
             "relevance": "Article 24 obliges the retailer to implement and demonstrate measures that keep processing compliant, framing the remediation and accountability follow-up the breach response must include.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 30",
             "relevance": "Article 30(1)(g) requires the retailer's processing records to document its security measures, the baseline against which the breached safeguards are assessed.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Where the database sits with a processor, that provider must assist the retailer in responding to the breach and meeting its security obligations.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 28",
             "relevance": "Article 28(3)(f) requires a hosting provider operating the database to assist the retailer in responding to the breach and meeting its security obligations, which matters where the database sits with a processor.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The notification goes to the competent Member State supervisory authority, with the lead-authority mechanism streamlining cross-border customer bases to a single authority.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 55",
             "relevance": "Article 55 identifies the Member State supervisory authority that receives the retailer's breach notification.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 56",
             "relevance": "Article 56 designates the lead supervisory authority where the retailer's cross-border customer base engages authorities in several Member States, streamlining the notification to a single authority.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="DORA's incident regime covers financial entities only, so an online retailer's breach response stays a GDPR matter.",
        citations=[
            {"source_id": "dora", "provision": "Article 2",
             "relevance": "Article 2's financial-entity list marks DORA's perimeter, which keeps an online retailer's breach response a GDPR matter for this scenario.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The supervisory enforcement landscape frames the breach response: the authority can impose corrective measures - from ordering the breach's communication to processing bans and fines - and affected customers can lodge complaints, so the retailer should anticipate both.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 58",
             "relevance": "Article 58 lists the corrective powers - ordering the breach's communication to data subjects, limiting or banning processing, ordering rectification or erasure, imposing fines, and suspending data flows - the retailer must anticipate as the breach response's enforcement exposure.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 77",
             "relevance": "Article 77 gives affected customers the right to lodge a complaint with a supervisory authority over the breach, so the retailer should anticipate and respond to breach-related complaints and keep the complainant informed of progress and outcome.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Where the retailer has designated a data protection officer, that officer must be involved properly and in a timely manner in the breach response - informing and advising the company on its obligations, cooperating with the supervisory authority and acting as contact point.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 38",
             "relevance": "Article 38 secures the designated data protection officer's proper and timely involvement in the breach response, with the resources and access the officer needs to advise on it.",
             "strength": Strength.moderate},
            {"source_id": "gdpr", "provision": "Article 39",
             "relevance": "Article 39's advisory and monitoring tasks define the designated data protection officer's role in overseeing the breach response and the security hardening that follows.",
             "strength": Strength.weak},
        ],
    ),
]

_AI_RECRUITMENT_SCREENING_EXPECTED = [
    ExpectedFinding(
        statement="Screening applicants' CVs and recorded video interviews to score and prioritize them is a high-risk AI use under the AI Act - recruitment and candidate evaluation are named high-risk areas - and the AI Act's duties apply alongside the GDPR.",
        citations=[
            {"source_id": "ai-act", "provision": "Annex 3",
             "relevance": "Annex III point 4(a) lists recruitment screening and candidate evaluation as a high-risk use area, which brings the CV- and video-scoring system within the Chapter III compliance regime.",
             "strength": Strength.strong},
            {"source_id": "ai-act", "provision": "Article 6",
             "relevance": "Article 6(2) classifies the recruitment system as high-risk, and the profiling carve-out in Article 6(3) keeps that status because candidate scoring is profiling, locking in the Chapter III deployer obligations.",
             "strength": Strength.strong},
            {"source_id": "ai-act", "provision": "Article 2",
             "relevance": "Article 2(7) confirms the AI Act runs alongside the GDPR, so the company must satisfy both regimes' duties for the screening system simultaneously.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The AI Act's definitions fix the company's role as deployer, confirm the scoring is profiling, and frame the emotion-recognition boundary check for the video analysis.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 3",
             "relevance": "Article 3 supplies the decisive definitions for this deployment: deployer fixes the company's role, profiling sustains the high-risk classification, and emotion recognition system frames the Article 5 boundary check for the video analysis.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Automated evaluation of applicants' performance is profiling under the GDPR, which anchors both the DPIA trigger and the AI Act's profiling-based classification.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 4",
             "relevance": "Article 4(4) defines automated evaluation of performance at work as profiling, which anchors both the DPIA trigger and the AI Act's profiling-based high-risk classification for the screening system.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="If the video-interview analysis infers emotions, that workplace emotion inference is prohibited under the AI Act and carries the prohibited-practice fines, so the company must verify the system's capabilities before use.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 5",
             "relevance": "Article 5(1)(f) prohibits workplace emotion inference, so the company must verify whether the video-interview analysis infers emotions, since such a capability would bring the use within the prohibition.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Article 99",
             "relevance": "Article 99's fines for prohibited practices attach to workplace emotion inference, which sharpens the pre-deployment verification of the video analysis.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="As deployer, the company must run the screener per the provider's instructions, assign competent human oversight, keep at least six months of logs, ensure representative input data, inform candidates that AI scoring is applied, and make sure its recruiters and operators have sufficient AI literacy.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 26",
             "relevance": "Article 26 obliges the company as deployer to run the screener per the provider's instructions, assign competent human oversight, keep at least six months of logs, ensure representative input data, and inform candidates that AI scoring is applied to them.",
             "strength": Strength.strong},
            {"source_id": "ai-act", "provision": "Article 4",
             "relevance": "Article 4 obliges the company to ensure its recruiters and system operators have sufficient AI literacy to interpret and oversee the candidate scores properly.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The provider owes instructions describing the screener's capabilities, accuracy and limits, oversight by design, and the conformity assessment, CE marking and EU-database registration markers the company verifies before adopting the system.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 13",
             "relevance": "Article 13 obliges the provider to deliver instructions describing the screener's capabilities, accuracy, and limits, which the company needs to operate the system per instructions and to brief candidates accurately.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Article 14",
             "relevance": "Article 14's oversight design of automation-bias awareness and the ability to disregard outputs defines what the company's assigned human overseers must implement as recruiters rely on the scores.",
             "strength": Strength.moderate},
            {"source_id": "ai-act", "provision": "Article 16",
             "relevance": "Article 16's provider duties of conformity assessment, CE marking, and registration are the compliance markers the company verifies when selecting the screening system.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Article 43",
             "relevance": "Article 43(2) sets the internal-control conformity assessment for Annex III employment systems, the procedure the provider must complete before the company deploys the screener.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Article 49",
             "relevance": "Article 49(1) requires the provider's EU-database registration of the high-risk screener, a checkpoint the company can verify before adoption.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="After deployment the provider keeps monitoring the screener's real-world performance, and the company informs the provider of serious incidents and anomalies through the deployer duty.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 72",
             "relevance": "Article 72 obliges the provider to keep monitoring the screener's real-world performance after deployment, giving the company a channel for reporting anomalies it observes.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Article 73",
             "relevance": "Article 73's serious-incident reporting route applies where the screener causes an incident within the Article 3(49) definition, reached through the company's duty to inform the provider under Article 26(5).",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Candidates adversely affected by score-based decisions have a right to clear explanations of the AI system's role in the prioritisation.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 86",
             "relevance": "Article 86 gives candidates adversely affected by score-based decisions a right to clear explanations of the AI system's role, so the company must be able to explain how the score shaped interview prioritisation.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The fundamental-rights impact assessment is confined to public bodies and specified deployers, so this private employer's pre-use assessment runs through the GDPR DPIA instead.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 27",
             "relevance": "Article 27 confines the fundamental-rights impact assessment to public bodies, public-service providers, and Annex III 5(b)-(c) deployers, so the private employer's required pre-use assessment for recruitment scoring runs through the GDPR DPIA.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The high-risk deployer duties for the recruitment system apply from 2 August 2026.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 113",
             "relevance": "Article 113(2) fixes 2 August 2026 as the application date, confirming the high-risk deployer duties for the recruitment system are in force at the time of use.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Collecting and scoring CVs and video interviews needs a lawful basis, with the processing kept fair, transparent and minimised to the recruitment purpose, and consent - where used - easy to withdraw.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 5",
             "relevance": "Article 5's fairness, transparency, and minimisation principles govern the collection and scoring of CVs and video interviews, requiring the recording and analysis to stay proportionate to the recruitment purpose.",
             "strength": Strength.moderate},
            {"source_id": "gdpr", "provision": "Article 6",
             "relevance": "Article 6(1) requires a lawful basis, plausibly legitimate interests or consent, before the company processes applicant CVs and interview recordings through the AI screener.",
             "strength": Strength.moderate},
        ],
    ),
    ExpectedFinding(
        statement="Where consent serves as the processing basis, applicants can withdraw it as easily as they gave it.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 7",
             "relevance": "Article 7(3) guarantees applicants easy withdrawal of consent, which the company must honour where consent serves as the processing basis.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Applicants must be told at collection that an AI system scores their materials, with meaningful information about the scoring's logic and envisaged consequences, covering data obtained from other sources too, and can access information about the automated decision-making's logic.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 13",
             "relevance": "Article 13(2)(f) obliges the company to tell applicants at collection that an AI system scores their materials, with meaningful information about the scoring's logic and envisaged consequences.",
             "strength": Strength.moderate},
            {"source_id": "gdpr", "provision": "Article 14",
             "relevance": "Article 14 extends the information duties to applicant data obtained from other sources, completing transparency coverage across the screening pipeline.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 15",
             "relevance": "Article 15(1)(h) lets applicants access information about the automated decision-making and its logic, reinforcing the explanation capability the company must maintain.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Where interview prioritisation becomes effectively solely automated, the GDPR safeguards - human intervention, expressing a point of view, and contesting the decision - must be available to candidates.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 22",
             "relevance": "Article 22 engages where interview prioritisation becomes effectively solely automated, in which case its safeguards of human intervention, expressing a point of view, and contesting must be available to candidates.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The systematic profiling-based evaluation of all applicants makes a DPIA mandatory before the screening goes live, with prior consultation where high residual risk remains.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 35",
             "relevance": "Article 35(3)(a) makes a DPIA mandatory for this systematic profiling-based evaluation of all applicants, so the assessment must be completed before the screening goes live.",
             "strength": Strength.strong},
            {"source_id": "gdpr", "provision": "Article 36",
             "relevance": "Article 36 requires prior consultation with the supervisory authority where the DPIA shows high residual risk, defining the escalation path for this deployment.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The company must route any special categories of personal data the CVs or recorded interviews may reveal through an Article 9 gateway - explicit consent being the available route - before the screener processes them.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 9",
             "relevance": "Article 9 restricts processing of special categories of personal data to defined gateways such as explicit consent, so the screening design must keep CV and interview data that may reveal health, religious, or union-affiliation information within such a gateway.",
             "strength": Strength.moderate},
        ],
    ),
    ExpectedFinding(
        statement="The screening must embed data protection by design and by default - minimising what is collected from CVs and interviews - and the recordings and scores must be secured with risk-appropriate measures.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 25",
             "relevance": "Article 25 requires the screening to embed data protection by design and by default, minimising the CV and interview data drawn into candidate scoring and pseudonymising where possible.",
             "strength": Strength.moderate},
            {"source_id": "gdpr", "provision": "Article 32",
             "relevance": "Article 32 obliges the company to secure the CVs, video recordings, and scores with technical and organisational measures proportionate to the risk, a direct duty over the screening pipeline.",
             "strength": Strength.moderate},
        ],
    ),
    ExpectedFinding(
        statement="The screening's processing must appear in the company's records of activities, covering purposes, categories of data subjects and personal data, recipients, and the security measures.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 30",
             "relevance": "Article 30 requires the screening processing to appear in the company's records of activities, covering purposes, categories of data subjects and personal data, recipients, and a description of the security measures.",
             "strength": Strength.weak},
        ],
    ),
]

_BANK_CLOUD_OUTAGE_EXPECTED = [
    ExpectedFinding(
        statement="As a credit institution the bank falls within DORA, and the cloud failure is an ICT-related incident affecting a critical function supplied by an ICT third-party service provider, with the response scaled to the bank's size and risk profile.",
        citations=[
            {"source_id": "dora", "provision": "Article 2",
             "relevance": "Article 2(1)(a) brings the bank within DORA as a credit institution, engaging the ICT incident-management and third-party regimes that govern the outage response.",
             "strength": Strength.moderate},
            {"source_id": "dora", "provision": "Article 3",
             "relevance": "Article 3's definitions of ICT-related incident, major incident, ICT third-party service provider, and critical or important function establish that the cloud failure is an ICT incident affecting a critical function supplied by an external provider.",
             "strength": Strength.moderate},
            {"source_id": "dora", "provision": "Article 4",
             "relevance": "Article 4 scales the bank's incident-response and third-party obligations to its size and risk profile, the proportionality lens applied across the analysis.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The bank must run the outage through its ICT incident-management process, classify it by clients affected, duration, data losses, service criticality and economic impact and, once classified major, submit initial, intermediate and final reports within the harmonised time limits while promptly informing affected clients about the incident and mitigation measures.",
        citations=[
            {"source_id": "dora", "provision": "Article 17",
             "relevance": "Article 17 obliges the bank to run the outage through its ICT incident-management process of detection, management, notification, escalation to senior management, and response procedures for timely restoration.",
             "strength": Strength.strong},
            {"source_id": "dora", "provision": "Article 18",
             "relevance": "Article 18(1) requires the bank to classify the outage by clients affected, duration, data losses, service criticality, and economic impact, which determines whether the major-incident reporting duty is triggered.",
             "strength": Strength.strong},
            {"source_id": "dora", "provision": "Article 19",
             "relevance": "Article 19 obliges the bank, once the outage is classified major, to submit initial, intermediate, and final reports to the competent authority and to promptly inform affected clients about the incident and mitigation measures.",
             "strength": Strength.strong},
        ],
    ),
    ExpectedFinding(
        statement="The bank must activate business continuity plans that prioritise resuming the critical online-banking function - plans that cover functions outsourced to the cloud provider - and restore from backups and restoration capabilities.",
        citations=[
            {"source_id": "dora", "provision": "Article 11",
             "relevance": "Article 11 obliges the bank to activate ICT business continuity and response plans that prioritise resumption of the critical online-banking function, and Article 11(4) requires those plans to cover functions outsourced through ICT third-party providers like the cloud host.",
             "strength": Strength.moderate},
            {"source_id": "dora", "provision": "Article 12",
             "relevance": "Article 12's backup and restoration duties frame the recovery actions the bank invokes to bring online banking back within acceptable downtime.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The major incident triggers a review of the ICT risk-management framework and a post-incident review of the outage's causes and the response's effectiveness, with a communication strategy framing what clients and stakeholders are told.",
        citations=[
            {"source_id": "dora", "provision": "Article 6",
             "relevance": "Article 6(5) makes a major incident a mandatory trigger for reviewing the bank's ICT risk-management framework, alongside the recovery itself.",
             "strength": Strength.weak},
            {"source_id": "dora", "provision": "Article 13",
             "relevance": "Article 13(2) requires a post-incident review of the outage's causes and of the effectiveness of the bank's response once the incident has disrupted core activities.",
             "strength": Strength.weak},
            {"source_id": "dora", "provision": "Article 14",
             "relevance": "Article 14's communication-strategy duty frames how the bank informs clients and stakeholders about the outage, complementing the specific client-notification duty in Article 19(3).",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The bank stays fully responsible for DORA compliance despite the provider's failure and any supervisory feedback on its reports; its dependence on a single cloud provider requires concentration-risk analysis and substitutability considerations, and the contract must secure incident assistance, notification of material developments and tested contingency plans, with audit, termination and exit-strategy levers.",
        citations=[
            {"source_id": "dora", "provision": "Article 28",
             "relevance": "Article 28 keeps the bank fully responsible for DORA compliance despite the cloud provider's failure and supplies the audit, termination, and exit-strategy levers for the provider relationship going forward.",
             "strength": Strength.moderate},
            {"source_id": "dora", "provision": "Article 29",
             "relevance": "Article 29's concentration-risk duties apply because the bank depends on a single cloud provider for critical online banking, requiring substitutability analysis and consideration of alternative solutions.",
             "strength": Strength.weak},
            {"source_id": "dora", "provision": "Article 30",
             "relevance": "Article 30 requires the cloud contract to secure incident assistance, notification of material developments, and tested contingency plans from the provider, the provisions the bank invokes during and after the outage.",
             "strength": Strength.moderate},
            {"source_id": "dora", "provision": "Article 22",
             "relevance": "Article 22 provides that the bank remains fully responsible for the incident's handling and its consequences notwithstanding any supervisory feedback or guidance the competent authority gives on its notification and reports.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Client personal data must remain protected and available: the bank owes security measures and timely restoration of access after the technical incident under the GDPR's integrity and confidentiality baseline.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 32",
             "relevance": "Article 32(1)(b)-(c) obliges the bank to ensure availability of its processing systems and timely restoration of access to client personal data after the technical incident, running in parallel with DORA's recovery duties.",
             "strength": Strength.moderate},
            {"source_id": "gdpr", "provision": "Article 5",
             "relevance": "Article 5(1)(f)'s integrity-and-confidentiality principle grounds the bank's ongoing duty to protect client personal data processed through the disrupted systems.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="If the outage investigation reveals a personal data breach, the 72-hour supervisory-authority notification and, where customers face high risk, the communication duty come into play alongside the DORA reporting, with the authority's corrective powers - from ordering communication to fines - behind that track.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 4",
             "relevance": "Article 4(12) frames the assessment of whether the outage also involved a personal data breach, which turns on whether client data were destroyed, lost, altered, disclosed, or accessed.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 33",
             "relevance": "Article 33's 72-hour notification duty becomes live where the outage investigation reveals a personal data breach, an obligation the bank must keep in view alongside its DORA reporting.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 34",
             "relevance": "Article 34's communication duty attaches where the incident creates high risk to customers, complementing the DORA client notification for affected online-banking users.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 58",
             "relevance": "Article 58's corrective powers - ordering the breach's communication to data subjects, limiting or banning processing, and imposing administrative fines - hang over the conditional GDPR track the outage could open.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="As a credit institution the bank cannot use the simplified ICT risk-management framework and must comply with the general framework of Articles 5 to 15.",
        citations=[
            {"source_id": "dora", "provision": "Article 16",
             "relevance": "Article 16's simplified ICT risk-management framework is not open to a credit institution, so the bank must comply with the general framework of Articles 5 to 15, calibrating which obligations bite.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="In hosting and processing the bank's customer data on the disrupted systems, the cloud provider acts as a processor: the bank may engage it only under a contract carrying Article 28(3)'s guarantees, and the provider must assist the bank in meeting its security and breach-response obligations.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 28",
             "relevance": "Article 28 obliges the bank to engage the cloud host only as a processor under a contract carrying Article 28(3)'s guarantees, with Article 28(3)(f) requiring the provider's assistance with the breach and security duties, the GDPR counterpart to the DORA contract levers.",
             "strength": Strength.moderate},
        ],
    ),
    ExpectedFinding(
        statement="The bank must identify and document the processes dependent on the cloud provider and its interconnections, keeping those inventories updated - the identification step grounding the recovery and third-party analysis.",
        citations=[
            {"source_id": "dora", "provision": "Article 8",
             "relevance": "Article 8 requires the bank to identify and document all processes dependent on the cloud provider and their interconnections, the classification grounding the criticality and recovery analysis for the disrupted functions.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The bank's management body must approve, oversee and periodically review the ICT business continuity policy and the response and recovery plans the outage engages and the post-incident revisions produce.",
        citations=[
            {"source_id": "dora", "provision": "Article 5",
             "relevance": "Article 5 places the management body over the outage response: it must approve, oversee, and periodically review the ICT business continuity policy and the response and recovery plans the incident engages.",
             "strength": Strength.weak},
        ],
    ),
]

_EMPLOYEE_PRODUCTIVITY_MONITORING_EXPECTED = [
    ExpectedFinding(
        statement="Continuously recording employees' computer activity and generating individual productivity scores is high-risk under the AI Act - AI that monitors and evaluates workers' performance and behaviour - with the AI Act running alongside the GDPR.",
        citations=[
            {"source_id": "ai-act", "provision": "Annex 3",
             "relevance": "Annex III point 4(b) classifies AI that monitors and evaluates workers' performance and behaviour as high-risk, bringing the productivity-scoring system within the Chapter III regime.",
             "strength": Strength.strong},
            {"source_id": "ai-act", "provision": "Article 6",
             "relevance": "Article 6(2) makes the monitoring system high-risk, and the profiling carve-out in Article 6(3) confirms the classification because the score evaluates employees.",
             "strength": Strength.strong},
            {"source_id": "ai-act", "provision": "Article 2",
             "relevance": "Article 2(7) confirms the AI Act runs alongside the GDPR, so the employer must satisfy both the AI Act deployer duties and the GDPR employment rules for the monitoring system.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Automated evaluation of performance at work is profiling, which anchors the GDPR's automated-decision and DPIA analyses and sustains the high-risk classification.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 4",
             "relevance": "Article 4(4) expressly covers automated evaluation of performance at work, confirming the productivity score is profiling and anchoring the Article 22 and Article 35 analyses for the monitoring system.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="As deployer, the employer must inform workers' representatives and affected workers before workplace use, run the system per the provider's instructions under competent oversight, keep at least six months of logs, and tell employees when the score feeds decisions affecting them.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 26",
             "relevance": "Article 26 obliges the employer to inform workers' representatives and affected workers before workplace use, to run the system per instructions under competent oversight, to keep at least six months of logs, and to tell employees when the score feeds decisions.",
             "strength": Strength.strong},
        ],
    ),
    ExpectedFinding(
        statement="The oversight design - automation-bias awareness and the ability to disregard outputs - defines what the employer's assigned overseers must implement as managers use the scores.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 14",
             "relevance": "Article 14's oversight measures of automation-bias awareness and the ability to disregard outputs define what the employer's assigned overseers must implement as managers use the scores.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The employer must verify the provider's instructions describing capabilities and limits and the compliance markers of conformity assessment, CE marking and EU-database registration before adopting the system.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 13",
             "relevance": "Article 13 obliges the provider to deliver instructions describing the system's capabilities and limits, which the employer needs to configure oversight of the scoring properly.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Article 16",
             "relevance": "Article 16's provider duties define the compliance markers of conformity assessment, CE marking, and registration, which the employer verifies when adopting the monitoring system.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Article 49",
             "relevance": "Article 49(1) requires the provider's EU-database registration of the monitoring system, a checkpoint the employer verifies before adoption.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Article 43",
             "relevance": "Article 43(2) sets the internal-control conformity assessment for Annex III employment systems, the procedure behind the conformity marker the employer verifies.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Article 47",
             "relevance": "Article 47 obliges the provider to draw up the EU declaration of conformity stating the system meets the Section 2 requirements, part of the documentation the employer examines.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Annex 8",
             "relevance": "Annex VIII's content requirements define what the EU declaration of conformity must carry, part of the documentation behind the markers the employer verifies.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Staff operating the monitoring and managers using the scores need sufficient AI literacy.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 4",
             "relevance": "Article 4 obliges the employer to ensure that the staff operating the monitoring and the managers using the scores have sufficient AI literacy.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Should the productivity system infer workers' emotions, that workplace emotion inference is prohibited under the AI Act, so the employer must check the system's capabilities against this boundary - and where emotion recognition exists, the workers exposed must be told of the system's operation.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 5",
             "relevance": "Article 5(1)(f) prohibits workplace emotion inference, so the employer must check the productivity system's capabilities against this boundary before deployment.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Article 50",
             "relevance": "Article 50(3) obliges the deployer of an emotion recognition system to inform the natural persons exposed to it of the system's operation, the transparency complement to the Article 5 prohibition the employer checks for.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The AI Act's definitions of deployer and profiling fix the employer's role and confirm the scoring is profiling, sustaining the high-risk classification.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 3",
             "relevance": "Article 3's definitions of deployer and profiling fix the employer's role and confirm the scoring is profiling, which sustains the high-risk classification.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Employees adversely affected by score-based decisions have a right to explanations of the AI system's role in performance evaluations.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 86",
             "relevance": "Article 86 gives employees adversely affected by score-based decisions a right to explanations of the AI system's role, which the employer must deliver in performance evaluations.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="After deployment the provider keeps reviewing real-world performance, and the employer informs the provider of serious incidents through the reporting channel.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 72",
             "relevance": "Article 72 keeps the provider reviewing the system's real-world performance after deployment, complementing the employer's own monitoring duty for anomalies.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Article 73",
             "relevance": "Article 73's serious-incident reporting route applies where the scoring system causes an incident, reached through the employer's duty to inform the provider.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The fundamental-rights impact assessment is confined to public bodies and specified deployers, so the employer's pre-use assessment runs through the GDPR DPIA.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 27",
             "relevance": "Article 27 confines the fundamental-rights impact assessment to public bodies, public-service providers, and Annex III 5(b)-(c) deployers, so the employer's pre-use assessment runs through the GDPR DPIA.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The monitoring needs a lawful basis - with employee consent under close scrutiny given the employment relationship's imbalance - exercised consistently with the fairness, transparency and minimisation principles.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 6",
             "relevance": "Article 6(1) requires a lawful basis for the monitoring, with legitimate interests demanding a balancing against employees' rights and consent carrying the free-given burden set by Article 7(4).",
             "strength": Strength.strong},
            {"source_id": "gdpr", "provision": "Article 5",
             "relevance": "Article 5's fairness, transparency, minimisation, and storage-limitation principles directly constrain continuous recording of applications, websites, and time worked, requiring the monitoring to be scoped to the narrowest set serving its purpose.",
             "strength": Strength.strong},
            {"source_id": "gdpr", "provision": "Article 7",
             "relevance": "Article 7(4) directs close scrutiny of employee consent given the employment relationship's imbalance, shaping whether consent can carry the monitoring's lawfulness.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Recorded websites and applications can reveal health, religious, or union-affiliation data, which the GDPR restricts to defined gateways such as explicit consent.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 9",
             "relevance": "Article 9 governs the risk that recorded websites and applications reveal health, religious, or union-affiliation data, so the monitoring design must route any such data through a gateway like explicit consent.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Employees must be informed of the monitoring's purposes, legal basis, retention and rights - including meaningful information about the productivity score's logic - in concise, accessible form, and can access information about the automated scoring's logic.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 13",
             "relevance": "Article 13 obliges the employer to inform employees of the monitoring's purposes, legal basis, retention, and rights, including meaningful information about the productivity score's automated logic and consequences.",
             "strength": Strength.moderate},
            {"source_id": "gdpr", "provision": "Article 12",
             "relevance": "Article 12 sets the concise, accessible form in which employees receive the monitoring information and the handling of their rights requests.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 15",
             "relevance": "Article 15(1)(h) lets employees access information about the automated scoring's logic, reinforcing the transparency owed over the productivity evaluation.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 14",
             "relevance": "Article 14 completes the transparency coverage where the activity data are not obtained directly from the employees, obliging the equivalent information within a reasonable period of obtaining them.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Where a performance evaluation effectively rests solely on the score, the GDPR safeguards - human intervention, expressing a point of view, and contesting - protect the employee.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 22",
             "relevance": "Article 22 engages where a performance evaluation effectively rests solely on the score, in which case its safeguards of human intervention, expressing a point of view, and contesting protect the employee.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Systematic automated evaluation of employees makes a DPIA mandatory before deployment, with prior consultation where high residual risk remains.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 35",
             "relevance": "Article 35(3)(a) makes a DPIA mandatory before deploying the monitoring, since systematic automated evaluation of employees is its paradigm high-risk processing.",
             "strength": Strength.strong},
            {"source_id": "gdpr", "provision": "Article 36",
             "relevance": "Article 36 requires prior consultation with the supervisory authority where the DPIA leaves high residual risk, defining the escalation step for this deployment.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Regular, systematic, large-scale monitoring as a core activity makes DPO designation mandatory for the employer.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 37",
             "relevance": "Article 37(1)(b) makes DPO designation mandatory where the monitoring is regular, systematic, and large-scale and forms a core activity, a threshold the employer must assess.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Member States may layer specific workplace-monitoring safeguards over the GDPR baseline, which the employer must check.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 88",
             "relevance": "Article 88 authorises national rules on employee-data processing with specific safeguards for workplace monitoring, so the employer must layer Member State requirements over the GDPR baseline.",
             "strength": Strength.moderate},
        ],
    ),
    ExpectedFinding(
        statement="The monitoring must embed data protection by design and by default - minimising what is recorded, pseudonymising where possible, and keeping the scores inaccessible by default - and the recorded activity and productivity scores must be secured with risk-appropriate measures.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 25",
             "relevance": "Article 25 requires the monitoring to embed data protection by design and by default, so the recording of applications, websites, and time worked must be minimised to what the scoring purpose needs, with pseudonymisation where possible and the productivity scores inaccessible by default.",
             "strength": Strength.moderate},
            {"source_id": "gdpr", "provision": "Article 32",
             "relevance": "Article 32 obliges the employer to secure the recorded activity and productivity scores with technical and organisational measures proportionate to the risk, a direct duty over the monitoring pipeline's logs and outputs.",
             "strength": Strength.moderate},
        ],
    ),
]

_TELECOM_GENAI_CHATBOT_EXPECTED = [
    ExpectedFinding(
        statement="Customers interacting with the chatbot must be told they are dealing with an AI system by the time of first interaction, and the chatbot's generated text outputs carry the machine-readable marking duty.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 50",
             "relevance": "Article 50(1) requires that customers be told they are interacting with an AI system by the time of first interaction, and Article 50(2)'s machine-readable marking duty covers the chatbot's generated text outputs.",
             "strength": Strength.strong},
        ],
    ),
    ExpectedFinding(
        statement="A customer-service question-answering chatbot sits in the transparency tier rather than the high-risk regime, with the telecom as a Union deployer owing AI Act duties alongside the GDPR, and the provider/deployer definitions and the generative engine's general-purpose AI model analysis determining who bears which duty.",
        citations=[
            {"source_id": "ai-act", "provision": "Annex 3",
             "relevance": "Annex III lists the AI use areas subject to the high-risk regime, and a customer-service question-answering chatbot sits in the transparency tier instead, which routes the deployment to the Article 50 duties.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Article 2",
             "relevance": "Article 2(1)(b) and 2(7) establish that the telecom, as a Union deployer, owes AI Act duties that run alongside its GDPR obligations for the chatbot.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Article 3",
             "relevance": "Article 3's provider and deployer definitions determine whether the telecom or its vendor bears the Article 50 transparency duty, and Article 3(63) frames the general-purpose AI model analysis for the generative engine.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Staff operating and supervising the chatbot need sufficient AI literacy.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 4",
             "relevance": "Article 4 obliges the telecom to ensure its staff operating and supervising the chatbot have sufficient AI literacy.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The underlying model's provider owes documentation, downstream information and policy duties - what the telecom examines when assessing its vendor's model-level compliance.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 53",
             "relevance": "Article 53's documentation, downstream-information, and policy duties bind the general-purpose AI model provider, informing the telecom's vendor due diligence for the chatbot's underlying engine.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Annex 11",
             "relevance": "Annex XI's documentation content for general-purpose AI models is part of what the telecom examines when assessing the vendor's model-level compliance.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Annex 12",
             "relevance": "Annex XII's downstream-information requirements show what the model provider owes the chatbot's provider, supporting the telecom's assessment of its vendor chain.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The names, account numbers and addresses customers enter are personal data, so the chatbot processing needs a lawful basis - plausibly contract necessity or legitimate interests - kept fair, minimal and storage-limited, with secondary uses such as model training passing the compatibility test.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 4",
             "relevance": "Article 4(1) confirms that the names, account numbers, and addresses customers type into the chatbot are personal data, bringing the processing within the GDPR.",
             "strength": Strength.moderate},
            {"source_id": "gdpr", "provision": "Article 5",
             "relevance": "Article 5's fairness, minimisation, and storage-limitation principles govern how long conversations are kept and how narrowly they are used, keeping secondary uses such as model training tied to compatible purposes.",
             "strength": Strength.moderate},
            {"source_id": "gdpr", "provision": "Article 6",
             "relevance": "Article 6 requires a lawful basis for the chatbot processing, plausibly contract necessity for customer service or legitimate interests subject to balancing, with Article 6(4) supplying the compatibility test for further uses such as training.",
             "strength": Strength.strong},
        ],
    ),
    ExpectedFinding(
        statement="Customers must be told, at the time they enter their details, of the chatbot processing's purposes, legal basis, retention and rights, in concise, accessible form, with combined data from other sources covered too.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 13",
             "relevance": "Article 13 obliges the telecom to inform customers, at the time they enter their details, of the chatbot processing's purposes, legal basis, retention, and their rights.",
             "strength": Strength.strong},
            {"source_id": "gdpr", "provision": "Article 12",
             "relevance": "Article 12 sets the concise, accessible form in which the telecom delivers the chatbot's privacy information and handles customers' rights requests.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 14",
             "relevance": "Article 14 completes the transparency coverage for customer data the telecom combines with chatbot conversations from other sources, such as account records.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The chatbot answers questions and routes account changes to human staff, keeping the GDPR's safeguards against solely automated decisions in reserve for future capability changes.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 22",
             "relevance": "Article 22 sets the decision-making boundary for the deployment: the chatbot answers questions and routes account changes to human staff, keeping the right against solely-automated decisions in reserve for future capability changes.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The chatbot's processing must appear in the telecom's records of processing activities, and the account numbers and addresses flowing through it must be secured with proportionate measures.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 30",
             "relevance": "Article 30 requires the chatbot's processing to appear in the telecom's records of activities, covering purposes, categories, recipients, and transfers.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 32",
             "relevance": "Article 32 obliges the telecom to secure the account numbers and addresses processed through the chatbot with measures proportionate to the risk.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="DORA's financial-entity perimeter leaves the telecom's chatbot duties to the GDPR and the AI Act.",
        citations=[
            {"source_id": "dora", "provision": "Article 2",
             "relevance": "Article 2 defines DORA's financial-entity perimeter, which assigns the telecom's chatbot duties to the GDPR and AI Act for this deployment.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The chatbot must embed data protection by design and by default - minimising what the conversations capture and keeping stored data inaccessible by default.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 25",
             "relevance": "Article 25 requires the chatbot to embed data protection by design and by default, minimising what the conversations capture and pseudonymising where possible, with stored data inaccessible by default.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Customers can exercise their data protection rights over the chatbot processing - access to their data and restriction where accuracy is contested or objection is pending - and the telecom must facilitate those requests.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 15",
             "relevance": "Article 15 gives customers access to the personal data the chatbot processing holds, including the safeguards information for any transfers, a rights channel the telecom must facilitate.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 18",
             "relevance": "Article 18 lets customers obtain restriction of the chatbot processing where accuracy is contested, processing is unlawful, or objection is pending, a rights channel the telecom must honour.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="If the chatbot's inference or storage runs outside the EEA, chatbot-derived personal data may flow to the third country only under the GDPR's transfer safeguards.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 44",
             "relevance": "Article 44 conditions any transfer of chatbot-derived personal data to a third country - for instance where the underlying model or its hosting sits outside the EEA - on compliance with the GDPR's transfer chapter, a location question to settle before operating the service.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 46",
             "relevance": "Article 46 supplies the appropriate-safeguards route - standard contractual clauses or binding corporate rules - for chatbot data flowing to the model provider's hosting outside the EEA, the concrete route behind the Article 44 condition.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Customer information typed into the chatbot may include special categories of personal data, which the telecom must keep within an Article 9 gateway - explicit consent being the available route - rather than assuming the free-text channel stays ordinary data.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 9",
             "relevance": "Article 9 restricts processing of special categories of personal data to defined gateways such as explicit consent, so the telecom must treat free-text chatbot input that may reveal health, religious, or similar data as a gateway question in the chatbot's design.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The telecom must assess whether DPO designation is required - regular and systematic large-scale monitoring or large-scale special-category processing as core activities would trigger it - and, where designated, involve the officer properly and in a timely manner in the chatbot deployment.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 37",
             "relevance": "Article 37 makes DPO designation turn on its thresholds - regular and systematic large-scale monitoring or large-scale special-category processing as core activities - a status question the telecom must assess for the chatbot.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 38",
             "relevance": "Article 38 secures the designated DPO's timely involvement and support in the chatbot deployment where designation applies.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 39",
             "relevance": "Article 39's advisory and monitoring tasks define the designated DPO's role over the chatbot's processing.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The telecom must be able to demonstrate the chatbot processing's compliance with the data protection principles - the accountability duty the by-design measures and records hang from.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 24",
             "relevance": "Article 24 obliges the telecom to implement and demonstrate the measures that keep the chatbot processing compliant - the accountability frame the by-design and records duties hang from.",
             "strength": Strength.weak},
        ],
    ),
]

_FINTECH_LOAN_RECOMMENDATIONS_EXPECTED = [
    ExpectedFinding(
        statement="AI systems that evaluate the creditworthiness of natural persons are high-risk under the AI Act, so the full high-risk regime applies to the loan system alongside the GDPR.",
        citations=[
            {"source_id": "ai-act", "provision": "Annex 3",
             "relevance": "Annex III point 5(b) lists creditworthiness evaluation and credit scoring as a high-risk use area, which places the fintech's loan-recommendation system squarely within the Chapter III regime.",
             "strength": Strength.strong},
            {"source_id": "ai-act", "provision": "Article 6",
             "relevance": "Article 6(2) classifies the loan-assessment system as high-risk, and the profiling carve-out in Article 6(3) keeps that status because applicant scoring is profiling.",
             "strength": Strength.strong},
            {"source_id": "ai-act", "provision": "Article 2",
             "relevance": "Article 2(7) confirms the AI Act runs alongside the GDPR, so the fintech owes both regimes' duties for the loan-assessment system.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The provider-side duties - risk management, data governance with bias examination, technical documentation, logging, instructions, oversight design, accuracy and robustness, and the compliance markers - are what the fintech verifies when adopting the loan system, and owes itself where it developed the system under its own name.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 9",
             "relevance": "Article 9's provider-side risk-management system is the foundation the fintech relies on when adopting the loan system, covering identified and foreseeable risks across its lifecycle.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Article 10",
             "relevance": "Article 10's data-governance duties, including bias examination and mitigation, are the discrimination-protection requirement for borrower scoring, defining what the fintech should verify about the system's training data.",
             "strength": Strength.moderate},
            {"source_id": "ai-act", "provision": "Article 11",
             "relevance": "Article 11's pre-market technical documentation demonstrates the loan system's Chapter III conformity, the evidence trail available to the fintech and to authorities.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Article 12",
             "relevance": "Article 12's logging requirement underpins the traceability of the loan system's outputs and supports the fintech's Article 26(6) log-keeping.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Article 13",
             "relevance": "Article 13 obliges the provider to supply instructions covering the system's capabilities, accuracy, and oversight measures, which the fintech needs to operate the system per instructions and to inform applicants.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Article 15",
             "relevance": "Article 15's accuracy, robustness, and cybersecurity baseline is the performance standard the fintech should verify for the loan system.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Article 16",
             "relevance": "Article 16's provider duties of conformity assessment, CE marking, and registration apply to the fintech itself where it developed the system, and otherwise define the compliance markers it verifies in its vendor.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The provider must have passed the internal-control conformity assessment and registered the system in the EU database before the fintech deploys it - checkpoints the fintech verifies before adoption.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 43",
             "relevance": "Article 43(2) sets the internal-control conformity assessment for Annex III credit systems, the gate the provider passes before the fintech deploys the system.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Article 49",
             "relevance": "Article 49(1) requires the provider's EU-database registration of the high-risk loan system, a checkpoint the fintech verifies before adoption.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="As deployer, the fintech must run the system per the provider's instructions, assign competent human oversight, keep logs through the financial-institution governance route, ensure representative input data, and inform applicants they are subject to AI-based assessment.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 26",
             "relevance": "Article 26 obliges the fintech as deployer to run the system per instructions, assign competent oversight, ensure representative input data, keep logs through the financial-institution governance route of paragraphs (5)-(6), and inform applicants they are subject to AI-based assessment.",
             "strength": Strength.strong},
        ],
    ),
    ExpectedFinding(
        statement="The employees reviewing recommendations must genuinely exercise human oversight - automation-bias awareness included - before final loan decisions.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 14",
             "relevance": "Article 14's human-oversight design, including automation-bias awareness, defines what the fintech's reviewing employees must genuinely exercise before final loan decisions.",
             "strength": Strength.moderate},
        ],
    ),
    ExpectedFinding(
        statement="Staff, including the reviewing employees, need sufficient AI literacy.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 4",
             "relevance": "Article 4 obliges the fintech to ensure its staff, including the employees reviewing recommendations, have sufficient AI literacy.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Declined applicants have a right to clear explanations of the AI system's role in the decision.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 86",
             "relevance": "Article 86 gives declined applicants a right to clear explanations of the AI system's role in the decision, which the fintech must be able to deliver in its loan outcomes.",
             "strength": Strength.moderate},
        ],
    ),
    ExpectedFinding(
        statement="As a deployer of a creditworthiness system, the fintech must perform a fundamental-rights impact assessment before deployment and notify the market-surveillance authority of its results.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 27",
             "relevance": "Article 27 makes the fintech, as a deployer of an Annex III 5(b) system, perform a fundamental-rights impact assessment before deployment and notify the market-surveillance authority of its results, a mandatory pre-use step for this scenario.",
             "strength": Strength.strong},
        ],
    ),
    ExpectedFinding(
        statement="The high-risk and fundamental-rights duties for the loan system apply from 2 August 2026.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 113",
             "relevance": "Article 113(2) fixes 2 August 2026 as the application date, confirming the high-risk and FRIA duties for the loan system are in force at the time of use.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The AI Act's definitions fix the fintech's role as deployer - and provider too where it developed the system under its own name.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 3",
             "relevance": "Article 3's definitions fix the fintech's role as deployer of a profiling system, and provider as well where it developed the system under its own name, which determines whether Article 16 duties also apply.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Processing applicants' financial data needs a lawful basis - plausibly steps prior to entering a loan contract - exercised consistently with the core principles, with the financial data sitting in the ordinary personal-data regime.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 6",
             "relevance": "Article 6(1)(b) supplies the natural lawful basis, the application being steps prior to entering a loan contract, for processing applicant financial data through the system.",
             "strength": Strength.moderate},
            {"source_id": "gdpr", "provision": "Article 5",
             "relevance": "Article 5's principles govern the collection and use of income, employment, and debt data, requiring the assessment to stay proportionate to the loan-decision purpose.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 9",
             "relevance": "Article 9's special-category scope confirms that the applicant financial data sit in the ordinary personal-data regime, routing the lawfulness analysis to Article 6 and its bases.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Applicants must be told about the automated decision-making and profiling in concise, accessible form, with meaningful information about the logic and consequences, covering data obtained from other sources too, and can access information about the automated assessment's logic.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 13",
             "relevance": "Article 13(2)(f) obliges the fintech to tell applicants about the automated decision-making and profiling in the assessment, with meaningful information about the logic and consequences.",
             "strength": Strength.moderate},
            {"source_id": "gdpr", "provision": "Article 12",
             "relevance": "Article 12 sets the concise, accessible form for the information and rights responses the fintech owes loan applicants.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 15",
             "relevance": "Article 15(1)(h) lets applicants access information about the automated assessment's logic, reinforcing the explanation duty alongside AI Act Article 86.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 14",
             "relevance": "Article 14 extends the transparency duties to applicant data the fintech obtains from other sources, such as credit-bureau records, completing coverage across the assessment pipeline.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Where the employee review becomes nominal and the decision rests solely on the system, the GDPR's safeguards - human intervention, expressing a view, contesting - become mandatory, a risk the automation-bias concern makes concrete.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 22",
             "relevance": "Article 22 engages where the employee review becomes nominal and the decision rests solely on the system, in which case the Article 22(3) safeguards of human intervention, expressing a view, and contesting become mandatory; AI Act Article 14(4)(b) identifies the automation-bias risk that threatens the review's substance.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The systematic profiling-based evaluation of all applicants makes a DPIA mandatory before the loan system goes into use, with prior consultation where high residual risk remains.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 35",
             "relevance": "Article 35(3)(a) makes a DPIA mandatory for the systematic profiling-based evaluation of all applicants, required before the loan system goes into use.",
             "strength": Strength.moderate},
            {"source_id": "gdpr", "provision": "Article 36",
             "relevance": "Article 36 requires prior consultation with the supervisory authority where the DPIA leaves high residual risk, defining the escalation path.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The applicant scoring is profiling, which anchors the DPIA trigger and the automated-decision analysis.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 4",
             "relevance": "Article 4(4) confirms the applicant scoring is profiling, which anchors the DPIA trigger and the automated-decision analysis for the loan system.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="If the fintech's authorization brings it within DORA as a financial entity, its management body becomes ultimately responsible for ICT risk and it must operate a documented ICT risk-management framework with reliable, resilient systems for the loan platform.",
        citations=[
            {"source_id": "dora", "provision": "Article 2",
             "relevance": "Article 2's financial-entity list determines whether the fintech's authorization brings it within DORA, which gates the ICT-resilience duties for the loan platform.",
             "strength": Strength.weak},
            {"source_id": "dora", "provision": "Article 5",
             "relevance": "Article 5 would make the fintech's management body ultimately responsible for ICT risk once it qualifies as a financial entity.",
             "strength": Strength.weak},
            {"source_id": "dora", "provision": "Article 6",
             "relevance": "Article 6 would require the fintech, as a financial entity, to operate a documented ICT risk-management framework for the loan platform, reviewed upon major incidents.",
             "strength": Strength.weak},
            {"source_id": "dora", "provision": "Article 7",
             "relevance": "Article 7 would require the fintech to run the loan platform on reliable, resilient ICT systems with sufficient capacity for processing volumes.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="On that same condition, the fintech must map the loan system's ICT dependencies, protect the platform's availability, integrity and confidentiality, and run detection mechanisms over the platform.",
        citations=[
            {"source_id": "dora", "provision": "Article 8",
             "relevance": "Article 8 would require the fintech to identify and map the loan system's ICT dependencies, including ICT third-party providers, as the prerequisite for criticality classification.",
             "strength": Strength.weak},
            {"source_id": "dora", "provision": "Article 9",
             "relevance": "Article 9 would require the fintech to deploy ICT security policies protecting the availability, integrity, and confidentiality of loan-processing data.",
             "strength": Strength.weak},
            {"source_id": "dora", "provision": "Article 10",
             "relevance": "Article 10 would require the fintech to operate detection mechanisms that promptly surface anomalies and intrusions in the loan platform.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="On the same condition it maintains tested business continuity plans, backups with segregated restoration, post-incident reviews, and communication strategies for incidents affecting loan customers, under the harmonised technical standards - or the simplified framework if it is a small non-interconnected firm.",
        citations=[
            {"source_id": "dora", "provision": "Article 11",
             "relevance": "Article 11 would require the fintech to maintain tested ICT business continuity and response plans covering the loan-approval service.",
             "strength": Strength.weak},
            {"source_id": "dora", "provision": "Article 12",
             "relevance": "Article 12 would require the fintech to maintain backup and segregated restoration capabilities for loan-platform data.",
             "strength": Strength.weak},
            {"source_id": "dora", "provision": "Article 13",
             "relevance": "Article 13 would require the fintech to run post-incident reviews after major incidents and incorporate lessons into its risk framework.",
             "strength": Strength.weak},
            {"source_id": "dora", "provision": "Article 14",
             "relevance": "Article 14 would require the fintech to maintain communication strategies for ICT incidents affecting loan customers.",
             "strength": Strength.weak},
            {"source_id": "dora", "provision": "Article 15",
             "relevance": "Article 15 would bring the fintech within the ESAs' harmonised technical standards for ICT risk-management tools and testing.",
             "strength": Strength.weak},
            {"source_id": "dora", "provision": "Article 16",
             "relevance": "Article 16 swaps in a simplified ICT risk-management framework where the fintech is a small and non-interconnected investment firm, requiring a status check before the full framework applies.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Any ICT third-party arrangement for the loan system would require due diligence, a register entry, exit planning, concentration-risk weighing and prescribed contract provisions.",
        citations=[
            {"source_id": "dora", "provision": "Article 28",
             "relevance": "Article 28 would govern any ICT third-party arrangement for the loan system, covering due diligence, a register entry, and exit planning once the fintech is a financial entity.",
             "strength": Strength.weak},
            {"source_id": "dora", "provision": "Article 29",
             "relevance": "Article 29 would require the fintech to weigh concentration risk from hard-to-substitute providers of the loan platform.",
             "strength": Strength.weak},
            {"source_id": "dora", "provision": "Article 30",
             "relevance": "Article 30 would prescribe the contractual provisions for ICT services supporting the loan platform, including locations, data protection, and incident assistance.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="If the fintech substantially modifies the system, puts its own name or trademark on it, or changes its intended purpose such that the system becomes high-risk, it is treated as the provider and owes the provider obligations, including any required new conformity assessment.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 25",
             "relevance": "Article 25 makes the fintech the provider - subject to the Article 16 obligations and any new conformity assessment - where it substantially modifies the loan system, puts its own name or trademark on it, or changes its intended purpose so the system becomes high-risk, the role-flip risk for a fintech adapting a vendor system.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Applicants can object to legitimate-interests-based processing including the profiling, obtain restriction where accuracy is contested or objection is pending, and, where processing rests on consent or contract and is automated, receive their data in a portable format.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 21",
             "relevance": "Article 21 gives applicants the right to object to processing based on legitimate interests, including the profiling behind the loan assessment, a right the fintech must bring to their attention explicitly at the latest at the time of first communication.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 18",
             "relevance": "Article 18 lets applicants obtain restriction of the assessment processing where accuracy is contested, processing is unlawful, or objection is pending.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 20",
             "relevance": "Article 20 lets applicants receive data they provided in a structured, commonly used, machine-readable format where the processing rests on consent or contract and is carried out by automated means.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The loan-assessment processing must embed data protection by design and by default and be secured with risk-appropriate technical and organisational measures.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 25",
             "relevance": "Article 25 requires the loan assessment to embed data protection by design and by default, minimising the financial data drawn into the scoring and pseudonymising where possible.",
             "strength": Strength.moderate},
            {"source_id": "gdpr", "provision": "Article 32",
             "relevance": "Article 32 obliges the fintech to secure applicants' financial data across the assessment pipeline with technical and organisational measures proportionate to the risk.",
             "strength": Strength.moderate},
        ],
    ),
    ExpectedFinding(
        statement="The loan-assessment processing must appear in the fintech's records of processing activities.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 30",
             "relevance": "Article 30 requires the loan-assessment processing to appear in the fintech's records of activities, covering purposes, categories of data subjects and personal data, recipients, and a description of the security measures.",
             "strength": Strength.weak},
        ],
    ),
]

_INSURANCE_HEALTH_PRICING_EXPECTED = [
    ExpectedFinding(
        statement="AI-based risk assessment and pricing for life and health insurance is a named high-risk use, so the full high-risk regime applies, with health-based applicant scoring confirmed as profiling.",
        citations=[
            {"source_id": "ai-act", "provision": "Annex 3",
             "relevance": "Annex III point 5(c) lists risk assessment and pricing AI for life and health insurance as high-risk, which places the insurer's premium-recommendation system squarely within the Chapter III regime.",
             "strength": Strength.strong},
            {"source_id": "ai-act", "provision": "Article 6",
             "relevance": "Article 6(2) classifies the insurance-pricing system as high-risk, and the profiling carve-out in Article 6(3) keeps that status because the system scores applicants.",
             "strength": Strength.strong},
            {"source_id": "ai-act", "provision": "Article 3",
             "relevance": "Article 3's profiling definition, incorporating GDPR Article 4(4), confirms that health-based applicant scoring is profiling, which sustains the high-risk classification under Article 6(3).",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The insurer's underwriting staff reviewing recommendations need sufficient AI literacy.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 4",
             "relevance": "Article 4 obliges the insurer to ensure its underwriting staff reviewing recommendations have sufficient AI literacy.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The provider's bias-examination and mitigation duties are the discrimination-protection requirement for health-based scoring, which the insurer verifies about the system's data governance.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 10",
             "relevance": "Article 10's bias-examination and mitigation duties are the discrimination-protection requirement for health-based scoring, defining what the insurer should verify about the system's data governance.",
             "strength": Strength.moderate},
        ],
    ),
    ExpectedFinding(
        statement="The provider must supply instructions describing capabilities, accuracy and oversight measures, and design the human oversight the reviewing employee must genuinely exercise before a policy is issued.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 13",
             "relevance": "Article 13 obliges the provider to supply instructions describing the system's capabilities, accuracy, and oversight measures, which the insurer needs to operate the system per instructions.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Article 14",
             "relevance": "Article 14's oversight design, including automation-bias awareness, defines what the insurer's reviewing employee must genuinely exercise before issuing a policy.",
             "strength": Strength.moderate},
        ],
    ),
    ExpectedFinding(
        statement="The compliance markers - conformity assessment, CE marking, EU-database registration, and the internal-control procedure for Annex III insurance systems - are what the insurer verifies when adopting the system, together with the lifecycle-wide risk management system behind them.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 16",
             "relevance": "Article 16's provider duties define the compliance markers the insurer verifies when adopting the system, and owes itself where it developed the system.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Article 43",
             "relevance": "Article 43(2) sets the internal-control conformity assessment for Annex III insurance systems, the gate the provider passes before the insurer deploys it.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Article 49",
             "relevance": "Article 49(1) requires the provider's EU-database registration of the high-risk pricing system, a checkpoint the insurer verifies.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Article 9",
             "relevance": "Article 9's continuous, lifecycle-wide risk management system is the foundation behind the conformity markers, covering identified and foreseeable risks across the pricing system's life, which the insurer verifies in vendor due diligence.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Article 47",
             "relevance": "Article 47 obliges the provider to draw up the EU declaration of conformity stating the system meets the Section 2 requirements, part of the documentation behind the markers the insurer verifies.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Article 48",
             "relevance": "Article 48 obliges the provider to affix the CE marking - visibly, legibly, and indelibly - to the high-risk system, the market marker the insurer examines on the delivered system.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Article 71",
             "relevance": "Article 71 obliges the provider, or its authorised representative, to register the Annex III pricing system in the EU database with the Annex VIII data before placing it on the market, completing the registration checkpoint the insurer verifies.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Annex 5",
             "relevance": "Annex V's content requirements define what the EU declaration of conformity must carry, part of what the insurer examines when verifying the provider's declaration.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="As deployer, the insurer must run the system per instructions, assign competent oversight, keep logs through the financial-institution governance route, and inform applicants they are subject to AI-based assessment.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 26",
             "relevance": "Article 26 obliges the insurer as deployer to run the system per instructions, assign competent oversight, keep logs through the financial-institution governance route, and inform applicants they are subject to AI-based assessment.",
             "strength": Strength.strong},
            {"source_id": "ai-act", "provision": "Article 25",
             "relevance": "Article 25 stops the insurer from putting its name on the system, substantially modifying it, or changing its intended purpose into a high-risk use without becoming the provider - the role-flip boundary for an insurer adapting the vendor system.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="As a deployer of an insurance pricing system, the insurer must perform a fundamental-rights impact assessment before deployment and notify the market-surveillance authority of its results.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 27",
             "relevance": "Article 27 makes the insurer, as a deployer of an Annex III 5(c) system, perform a fundamental-rights impact assessment before deployment and notify the authority, a mandatory pre-use step for the pricing system.",
             "strength": Strength.strong},
        ],
    ),
    ExpectedFinding(
        statement="The provider's post-market monitoring can be integrated into the insurer's existing financial-services governance frameworks, keeping oversight after the system enters service.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 72",
             "relevance": "Article 72 lets the insurer integrate the provider's post-market monitoring into its existing financial-services governance frameworks, keeping oversight after the system enters service.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Applicants adversely affected by premium or eligibility decisions have a right to explanations of the AI system's role in the outcome.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 86",
             "relevance": "Article 86 gives applicants adversely affected by premium or eligibility decisions a right to explanations of the AI system's role, which the insurer must deliver in its outcomes.",
             "strength": Strength.moderate},
        ],
    ),
    ExpectedFinding(
        statement="The applicants' medical information is data concerning health, so the underwriting processing is restricted to the Article 9 gateways - explicit consent the clearly available route for underwriting, with Member States able to add conditions - and needs a lawful basis alongside the gateway.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 4",
             "relevance": "Article 4(15) defines data concerning health broadly enough to cover the medical information the insurer analyses, which brings the underwriting within Article 9's special-category regime.",
             "strength": Strength.moderate},
            {"source_id": "gdpr", "provision": "Article 9",
             "relevance": "Article 9 restricts the insurer's processing of applicants' medical information to defined gateways, with explicit consent under Article 9(2)(a) the clearly available route for underwriting, the decisive constraint on the AI pricing system. Article 9(4) lets Member States add further conditions on health data, a layer the insurer must check.",
             "strength": Strength.strong},
            {"source_id": "gdpr", "provision": "Article 6",
             "relevance": "Article 6 requires a lawful basis alongside the Article 9 gateway, plausibly contract-necessity steps for the insurance application or consent.",
             "strength": Strength.moderate},
            {"source_id": "gdpr", "provision": "Article 7",
             "relevance": "Article 7 conditions any consent the insurer relies on - demonstrability, a clearly distinguishable request, and easy withdrawal under Article 7(3) - shaping whether explicit consent can carry the underwriting's lawfulness.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The collection and use of applicants' medical and personal data must stay proportionate and transparent under the GDPR's principles.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 5",
             "relevance": "Article 5's principles govern the collection and use of applicants' medical and personal data, requiring the underwriting processing to stay proportionate and transparent.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Applicants must be told about the automated decision-making and profiling behind eligibility and premium determination, with meaningful information about the logic and consequences, and can access information about the automated underwriting's logic.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 13",
             "relevance": "Article 13(2)(f) obliges the insurer to inform applicants of the automated decision-making and profiling behind eligibility and premium determination, with meaningful information about the logic and consequences.",
             "strength": Strength.moderate},
            {"source_id": "gdpr", "provision": "Article 15",
             "relevance": "Article 15(1)(h) lets applicants access information about the automated underwriting's logic, reinforcing the explanation duties alongside AI Act Article 86.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 12",
             "relevance": "Article 12 sets the deadline, free-of-charge handling, and concise form governing the insurer's responses to applicants' rights requests over the underwriting.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 14",
             "relevance": "Article 14 extends the information duties to applicants' health data the insurer obtains from other sources, such as medical records, covering the source and the automated decision-making's logic.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Automated decisions resting on health data are restricted to the explicit-consent and public-interest gateways with suitable safeguards, conditioning the premium-recommendation workflow, while the employee-review step determines whether the solely-automated safeguards are engaged at all.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 22",
             "relevance": "Article 22(4) restricts automated decisions resting on health data to the explicit-consent and public-interest gateways with suitable safeguards in place, directly conditioning the premium-recommendation workflow, while the employee-review step determines whether Article 22(1) is engaged at all.",
             "strength": Strength.moderate},
        ],
    ),
    ExpectedFinding(
        statement="Large-scale processing of health data makes a DPIA mandatory before the AI underwriting begins, with prior consultation where high residual risk remains.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 35",
             "relevance": "Article 35(3)(b) makes a DPIA mandatory for large-scale processing of health data, required before the AI underwriting begins.",
             "strength": Strength.moderate},
            {"source_id": "gdpr", "provision": "Article 36",
             "relevance": "Article 36 requires prior consultation with the supervisory authority where the DPIA leaves high residual risk, the escalation step for this deployment.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Large-scale special-category processing as a core activity makes DPO designation mandatory, with the DPO's timely involvement, resources and direct reporting line secured and its advisory and monitoring tasks defined.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 37",
             "relevance": "Article 37(1)(c) makes DPO designation mandatory where the insurer's core activities involve large-scale special-category processing, which health-based underwriting plausibly is.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 38",
             "relevance": "Article 38 secures the designated DPO's timely involvement, resources, and direct reporting line within the insurer's governance.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 39",
             "relevance": "Article 39's advisory and monitoring tasks define the DPO's role in overseeing the health-data underwriting and its impact assessment.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The insurer is a financial entity within DORA, whose ICT-resilience duties attach to its operations while the AI and health-data questions are governed by the GDPR and the AI Act.",
        citations=[
            {"source_id": "dora", "provision": "Article 2",
             "relevance": "Article 2(1)(n) confirms the insurer is a financial entity within DORA's scope, whose ICT-resilience duties attach to its operations while the scenario's AI and health-data questions are governed by the GDPR and AI Act.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The insurer's reliance on the AI provider as an ICT third-party service provider must run through the third-party regime: the insurer manages the third-party risk as an integral component of its ICT risk-management framework and remains fully responsible for compliance, with the arrangement assessed and registered where it supports critical or important functions.",
        citations=[
            {"source_id": "dora", "provision": "Article 8",
             "relevance": "Article 8 obliges the insurer, as a financial entity, to manage ICT third-party risk as an integral component of its ICT risk-management framework, covering the AI provider's arrangement.",
             "strength": Strength.weak},
            {"source_id": "dora", "provision": "Article 28",
             "relevance": "Article 28 governs the insurer's reliance on the AI provider: pre-contract assessment, the register of information, information-security standards, and full institutional responsibility despite the provider's involvement, biting where the arrangement supports critical or important functions.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The underwriting must embed data protection by design and by default - minimising the medical and personal data drawn into eligibility and premium determination, with pseudonymisation where possible.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 25",
             "relevance": "Article 25 requires the underwriting to embed data protection by design and by default, minimising the medical and personal data drawn into eligibility and premium determination and pseudonymising where possible.",
             "strength": Strength.moderate},
        ],
    ),
]

_RANSOMWARE_INVESTMENT_FIRM_EXPECTED = [
    ExpectedFinding(
        statement="The investment firm is a financial entity within DORA, and the ransomware event is an ICT-related incident - a cyber-attack of the reportable category from the outset.",
        citations=[
            {"source_id": "dora", "provision": "Article 2",
             "relevance": "Article 2(1)(e) brings the investment firm within DORA as a financial entity, engaging the incident-management and reporting regime for the ransomware attack.",
             "strength": Strength.moderate},
            {"source_id": "dora", "provision": "Article 3",
             "relevance": "Article 3's definitions of ICT-related incident, cyber-attack, and major incident establish the ransomware event as an ICT incident of the reportable category from the outset.",
             "strength": Strength.moderate},
        ],
    ),
    ExpectedFinding(
        statement="The firm must run the attack through its ICT incident-management process - detection, management, notification, root-cause identification, senior-management escalation - and classify it by clients affected, duration, data losses including the open confidentiality question, criticality and economic impact to gate the reporting duty.",
        citations=[
            {"source_id": "dora", "provision": "Article 17",
             "relevance": "Article 17 obliges the firm to run the ransomware incident through its ICT incident-management process of detection, management, notification, root-cause identification, senior-management escalation, and response for timely restoration.",
             "strength": Strength.strong},
            {"source_id": "dora", "provision": "Article 18",
             "relevance": "Article 18(1) requires the firm to classify the incident by clients affected, duration, data losses including the open confidentiality question, criticality of trading services, and economic impact, which gates the reporting duty.",
             "strength": Strength.strong},
        ],
    ),
    ExpectedFinding(
        statement="Once classified major, the firm reports through initial, intermediate and final reports - the intermediate ones accommodating the evolving investigation into client-data access - within the prescribed templates and time limits, with the authority's acknowledgement and feedback closing the supervisory loop.",
        citations=[
            {"source_id": "dora", "provision": "Article 19",
             "relevance": "Article 19 obliges the firm to report the major incident through initial, intermediate, and final reports, with the intermediate reports accommodating the evolving investigation into client-data access, and Article 19(3) adds the client-notification duty where financial interests are affected.",
             "strength": Strength.strong},
            {"source_id": "dora", "provision": "Article 22",
             "relevance": "Article 22 provides for the authority's acknowledgement and feedback on the incident reports, the supervisory loop supporting the firm's handling.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The firm must contain the attack, limit damage and prioritise resumption of trading, restoring systems and data from backups with segregation and integrity checks.",
        citations=[
            {"source_id": "dora", "provision": "Article 11",
             "relevance": "Article 11(2) frames the containment, damage-limitation, and prioritised-resumption duties the firm executes while restoring trading and internal systems.",
             "strength": Strength.weak},
            {"source_id": "dora", "provision": "Article 12",
             "relevance": "Article 12's backup and segregated-restoration duties govern the recovery of systems and data, with Article 12(3)'s segregation and integrity checks applying to any restoration from backups.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The major incident triggers a review of the ICT risk-management framework and a post-incident review of response promptness, forensic quality, escalation and communication, with the management body approving the resulting revisions and a communication strategy framing messaging to clients, staff and stakeholders.",
        citations=[
            {"source_id": "dora", "provision": "Article 6",
             "relevance": "Article 6(5) makes a major incident a mandatory trigger for reviewing the firm's ICT risk-management framework, framing the post-attack hardening.",
             "strength": Strength.weak},
            {"source_id": "dora", "provision": "Article 13",
             "relevance": "Article 13(2) requires a post-incident review once the major incident has disrupted core activities, covering response promptness, forensic quality, escalation, and communication, squarely the follow-up the firm owes after this attack.",
             "strength": Strength.weak},
            {"source_id": "dora", "provision": "Article 14",
             "relevance": "Article 14's communication-strategy duty frames the firm's messaging to clients, staff, and stakeholders about the attack, complementing the Article 19(3) client notification.",
             "strength": Strength.weak},
            {"source_id": "dora", "provision": "Article 5",
             "relevance": "Article 5 places the management body over the follow-up: it must approve, oversee, and periodically review the revised ICT business continuity policy and response and recovery plans the post-incident review produces.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="A size and status check applies: small and non-interconnected investment firms follow the simplified framework, while the incident-reporting duties continue to apply in full.",
        citations=[
            {"source_id": "dora", "provision": "Article 16",
             "relevance": "Article 16 requires a size and status check: small and non-interconnected investment firms follow the simplified framework while the incident-reporting duties of Articles 17-19 continue to apply in full.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The ransomware intrusion into systems holding client data is a personal data breach on occurrence, and the firm must notify the supervisory authority within 72 hours of awareness, phasing information while the access investigation proceeds and documenting the breach.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 4",
             "relevance": "Article 4(12) makes the ransomware intrusion, with unauthorized access and potential encryption-destruction of client data, a personal data breach on occurrence, grounding the parallel GDPR track.",
             "strength": Strength.strong},
            {"source_id": "gdpr", "provision": "Article 33",
             "relevance": "Article 33 obliges the firm to notify the supervisory authority within 72 hours of awareness of the intrusion into systems holding client data, with phased information under Article 33(4) while the access investigation proceeds and documentation under Article 33(5).",
             "strength": Strength.strong},
        ],
    ),
    ExpectedFinding(
        statement="Clients whose data the attackers accessed must be informed where the breach is likely to result in high risk, with the encryption condition available where affected data were unintelligible.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 34",
             "relevance": "Article 34 requires communication to affected clients where attacker access to their data creates high risk, with Article 34(3)(a)'s encryption condition available where affected data were unintelligible.",
             "strength": Strength.moderate},
        ],
    ),
    ExpectedFinding(
        statement="Client data and trading systems must be protected with appropriate technical and organisational security measures, including timely restoration of availability and access, framed by the integrity and confidentiality principle.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 32",
             "relevance": "Article 32's security duties, including Article 32(1)(c)'s timely restoration of availability and access to client data, frame both the incident's prevention dimension and the recovery architecture.",
             "strength": Strength.moderate},
            {"source_id": "gdpr", "provision": "Article 5",
             "relevance": "Article 5(1)(f)'s integrity-and-confidentiality principle frames the security hardening the firm owes client data after the attack.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The firm's remediation must produce demonstrable compliance measures under the accountability duty.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 24",
             "relevance": "Article 24's accountability duty frames the demonstrable compliance measures the firm's remediation must produce.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Where remediating the attack entails a major change in the firm's network and information system infrastructure, the firm must perform a risk assessment upon that significant change.",
        citations=[
            {"source_id": "dora", "provision": "Article 8",
             "relevance": "Article 8 obliges the firm to run ICT risk assessments upon significant changes to network and information system infrastructure, the duty a major post-ransomware rebuild triggers.",
             "strength": Strength.weak},
        ],
    ),
]

_GENAI_CUSTOMER_SERVICE_ASSISTANT_EXPECTED = [
    ExpectedFinding(
        statement="Customer conversations flowing through the AI assistant must be processed fairly and transparently on an identified lawful basis, with any new purpose passing the compatibility test.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 5",
             "relevance": "Article 5(1)(a)'s fairness and transparency principle frames how the institution handles customer conversations now processed through an AI assistant.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 6",
             "relevance": "Article 6 requires the institution to identify a lawful basis for the customer conversations flowing through the AI service, with any new purpose passing the Article 6(4) compatibility test.",
             "strength": Strength.moderate},
        ],
    ),
    ExpectedFinding(
        statement="Customers must receive concise, accessible information that their conversations are processed through an AI-assisted service, covering purposes, legal basis, recipients and transfers, with rights responses in the same form.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 13",
             "relevance": "Article 13 obliges the institution to inform customers that their conversations are processed through an AI-assisted service, covering purposes, legal basis, recipients, and transfers.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 12",
             "relevance": "Article 12 sets the concise, accessible form for the information and rights responses the institution owes customers whose conversations are processed through the assistant.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 14",
             "relevance": "Article 14 extends the information duties to customer data the institution obtains from other sources, such as account records combined with conversations, within a reasonable period of obtaining them.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The institution needs processor contracts with the AI provider and the cloud host as sub-processor - documented instructions, security, assistance, deletion or return, audit access - records of the processing, and security across the arrangement.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 28",
             "relevance": "Article 28 requires processor contracts with the AI provider and the cloud host as sub-processor, covering documented instructions, security measures, assistance with rights and breach duties, deletion or return, and audit access, the GDPR counterpart to DORA's contractual toolkit.",
             "strength": Strength.strong},
            {"source_id": "gdpr", "provision": "Article 30",
             "relevance": "Article 30 requires the AI-assistant processing to appear in the institution's records, covering purposes, categories of data, recipients including processors, and any transfers.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 32",
             "relevance": "Article 32 obliges the institution to ensure the security of customer data processed by the providers, applying jointly to controller and processors across the arrangement.",
             "strength": Strength.moderate},
        ],
    ),
    ExpectedFinding(
        statement="Any third-country movement of customer conversations to the externally hosted AI and cloud infrastructure must rest on the Chapter V safeguards, with standard contractual clauses the appropriate-safeguards route - making the hosting locations a central fact.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 44",
             "relevance": "Article 44 conditions any third-country movement of customer conversations to the externally hosted AI and cloud infrastructure on compliance with the Chapter V safeguards, which makes the hosting locations a central fact for this scenario.",
             "strength": Strength.moderate},
            {"source_id": "gdpr", "provision": "Article 46",
             "relevance": "Article 46 supplies the appropriate-safeguards route, such as standard contractual clauses, enabling lawful transfers where the AI service or cloud hosting processes conversations outside the EEA.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="A customer-service support role leaves the assistant in the transparency tier of the AI Act's high-risk perimeter, and a shift toward creditworthiness evaluation would change that classification.",
        citations=[
            {"source_id": "ai-act", "provision": "Annex 3",
             "relevance": "Annex III defines the high-risk perimeter for the assistant: a customer-service support role leaves it in the transparency tier, and a shift toward creditworthiness evaluation under point 5(b) would change that classification.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The AI Act's definitions fix the institution's role as deployer and frame the vendor-side analysis for the underlying generative model.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 3",
             "relevance": "Article 3's definitions of deployer and general-purpose AI model fix the institution's role as deployer and frame the vendor-side analysis for the underlying generative model.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Customer-service employees using the assistant need sufficient AI literacy.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 4",
             "relevance": "Article 4 obliges the institution to ensure its customer-service employees using the assistant have sufficient AI literacy.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The disclosure duty attaches to direct customer interaction, which the employee-support design leaves conditional, so the institution should secure the disclosure capability for any customer-facing use.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 50",
             "relevance": "Article 50(1)'s disclosure duty attaches to direct customer interaction, which the assistant's employee-support design leaves conditional here, so the institution should secure the disclosure capability for any customer-facing use.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The technology company's documentation, downstream-information and policy duties as general-purpose AI model provider are what the institution examines in vendor due diligence on a service it has limited visibility into.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 53",
             "relevance": "Article 53's technical-documentation, downstream-information, and policy duties bind the technology company as general-purpose AI model provider, informing the institution's vendor due diligence on a service it has limited visibility into.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Annex 12",
             "relevance": "Annex XII's downstream-information requirements define what the model provider owes the system provider integrating it, part of the compliance chain the institution examines.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Both the AI provider and the hosting cloud fall within DORA's third-party regime, and the institution must assign a role or senior manager to monitor those arrangements - directly addressing the limited-visibility concern.",
        citations=[
            {"source_id": "dora", "provision": "Article 3",
             "relevance": "Article 3's definitions of ICT services and ICT third-party service provider bring both the AI provider and the hosting cloud within the third-party regime, the vocabulary on which the analysis rests.",
             "strength": Strength.moderate},
            {"source_id": "dora", "provision": "Article 5",
             "relevance": "Article 5(3) requires the institution to assign a role, or a senior manager, to monitor the ICT third-party arrangements, directly addressing the limited-visibility concern.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The whole reliance is governed by DORA's third-party regime: pre-contract due diligence and criticality assessment, a register covering both providers, audit rights suited to the AI service's complexity, the subcontracting chain between the AI provider and the cloud host, and contract terms securing locations, data protection, incident assistance, contingency plans and monitoring rights - with full institutional responsibility preserved.",
        citations=[
            {"source_id": "dora", "provision": "Article 28",
             "relevance": "Article 28 governs the whole reliance: pre-contract due diligence and criticality assessment, the register of information covering both providers, information-security standards, audit rights suited to the AI service's technical complexity, termination grounds, and exit strategies preserving full institutional responsibility.",
             "strength": Strength.strong},
            {"source_id": "dora", "provision": "Article 29",
             "relevance": "Article 29(2) squarely addresses the structure, since the AI provider's reliance on another cloud host is a subcontracting chain the institution must weigh, especially where long or complex chains impair its ability to fully monitor the contracted functions.",
             "strength": Strength.strong},
            {"source_id": "dora", "provision": "Article 30",
             "relevance": "Article 30 prescribes the contract content the institution must secure, covering locations and data-processing venues, personal-data protection, data return, incident assistance at defined cost, notification of material developments, tested contingency plans, and ongoing performance-monitoring rights, the toolkit for restoring visibility into the service.",
             "strength": Strength.strong},
        ],
    ),
    ExpectedFinding(
        statement="The institution is a financial entity within DORA, so its reliance on the external AI provider and the hosting cloud is governed through DORA's third-party regime, while the customer-data and AI questions are governed by the GDPR and the AI Act.",
        citations=[
            {"source_id": "dora", "provision": "Article 2",
             "relevance": "Article 2's financial-entity list brings the institution within DORA's scope, so its reliance on the AI provider and the hosting cloud is governed through DORA's third-party regime, while the customer-data and AI questions are governed by the GDPR and the AI Act.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The AI-assisted service must embed data protection by design and by default - minimising what the conversations capture and keeping stored data inaccessible by default.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 25",
             "relevance": "Article 25 requires the assistant to embed data protection by design and by default, minimising what the conversations capture and pseudonymising where possible, with stored data inaccessible by default.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The institution must identify and document the processes dependent on the AI provider and the hosting cloud, and run the ICT risk assessment before and after connecting the assistant.",
        citations=[
            {"source_id": "dora", "provision": "Article 8",
             "relevance": "Article 8 requires the institution to identify and document all processes dependent on ICT third-party providers and to run ICT risk assessments before and after connecting technologies such as the AI assistant, the identification step grounding the third-party analysis.",
             "strength": Strength.weak},
            {"source_id": "dora", "provision": "Article 6",
             "relevance": "Article 6 obliges the institution to maintain a sound, comprehensive, and well-documented ICT risk-management framework covering all information and ICT assets, including the AI assistant, the frame the identification step feeds.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Customers whose conversations flow to the externally hosted service have data protection rights over the processing - access to their data and copies on request, including the safeguards information for any transfers - which the institution must facilitate.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 15",
             "relevance": "Article 15 gives customers access to the personal data the assistant processing holds - copies of their data, including the safeguards information for any third-country transfers - a rights channel the institution must facilitate.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Customer conversations may reveal special categories of personal data or criminal-offence data, which the institution must keep within their respective gateways unless a specific ground applies.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 9",
             "relevance": "Article 9 restricts the AI-based handling of customer conversations to defined gateways for any special categories the conversations may reveal, a design constraint the institution carries over the free-text channel.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The institution must assess whether DPO designation is required - regular and systematic large-scale monitoring or large-scale special-category processing as core activities would trigger it - and, where designated, involve the officer properly and in a timely manner in the assistant's deployment.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 37",
             "relevance": "Article 37 makes DPO designation turn on its thresholds - regular and systematic large-scale monitoring or large-scale special-category processing as core activities - a status question the institution must assess for the assistant.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 38",
             "relevance": "Article 38 secures the designated DPO's timely involvement, resources, and support in the assistant's deployment where designation applies.",
             "strength": Strength.weak},
            {"source_id": "gdpr", "provision": "Article 39",
             "relevance": "Article 39's advisory and monitoring tasks define the designated DPO's role over the assistant's processing and impact assessment.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="Where the AI-assisted processing, as processing using new technologies, is likely to result in high risk, a data protection impact assessment is required - on the employee-support design the stated facts leave that trigger a check for the institution rather than a settled obligation.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 35",
             "relevance": "Article 35's DPIA duty attaches where processing using new technologies is likely to result in high risk, notably systematic evaluation producing significant effects - a threshold the assistant's processing meets only conditionally, keeping the assessment a check rather than a settled duty.",
             "strength": Strength.weak},
        ],
    ),
]

_AI_TRADING_CLOUD_ATTACK_EXPECTED = [
    ExpectedFinding(
        statement="The investment firm is a financial entity within DORA, and the cyberattack is an ICT-related incident affecting a critical or important function supplied by a provider the firm depends on - a major-incident candidate from the outset.",
        citations=[
            {"source_id": "dora", "provision": "Article 2",
             "relevance": "Article 2(1)(e) brings the investment firm within DORA as a financial entity, engaging the incident-reporting and third-party regimes for the cloud attack.",
             "strength": Strength.moderate},
            {"source_id": "dora", "provision": "Article 3",
             "relevance": "Article 3's definitions of ICT-related incident, cyber-attack, major incident, critical or important function, and ICT concentration risk frame the attack as a major-incident candidate affecting a critical function supplied by a provider the firm depends on.",
             "strength": Strength.moderate},
        ],
    ),
    ExpectedFinding(
        statement="The firm must run the attack through its ICT incident-management process, classify it by clients affected, duration of more than a day, data losses including the potential client-data access, criticality and economic impact, and report through initial, intermediate and final reports as the investigation evolves, following the ESAs' templates and time limits.",
        citations=[
            {"source_id": "dora", "provision": "Article 17",
             "relevance": "Article 17 obliges the firm to run the attack through its ICT incident-management process of detection, management, notification, senior-management escalation, and response procedures for timely restoration of trading.",
             "strength": Strength.strong},
            {"source_id": "dora", "provision": "Article 18",
             "relevance": "Article 18(1) requires the firm to classify the incident by clients affected, duration of more than a day, data losses including the potential client-data access, criticality of trading services, and economic impact, which gates the reporting duty.",
             "strength": Strength.strong},
            {"source_id": "dora", "provision": "Article 19",
             "relevance": "Article 19 obliges the firm to report the major incident through initial, intermediate, and final reports as the investigation into client-data access evolves, and Article 19(3) adds the prompt client notification where financial interests are affected.",
             "strength": Strength.strong},
        ],
    ),
    ExpectedFinding(
        statement="The firm must contain the attack, prioritise resuming trading, maintain continuity plans covering the outsourced critical function, and restore trading systems and client data from backups with segregation and redundancy, while a communication strategy frames what clients and stakeholders are told.",
        citations=[
            {"source_id": "dora", "provision": "Article 11",
             "relevance": "Article 11(2) frames the containment, prioritised resumption, and crisis-communication duties the firm executes while trading systems are down, with Article 11(4)'s outsourced-critical-function continuity plans squarely engaged.",
             "strength": Strength.weak},
            {"source_id": "dora", "provision": "Article 12",
             "relevance": "Article 12's backup, segregated-restoration, and redundancy duties frame the firm's recovery of trading systems and client data from the attack.",
             "strength": Strength.weak},
            {"source_id": "dora", "provision": "Article 14",
             "relevance": "Article 14's communication-strategy duty frames the firm's messaging to clients and stakeholders about the attack, complementing the Article 19(3) client notification.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The major incident triggers a review of the ICT risk-management framework and a post-incident review of response promptness, forensic quality, escalation and communication, with the management body approving the resulting revisions.",
        citations=[
            {"source_id": "dora", "provision": "Article 6",
             "relevance": "Article 6(5) makes a major incident a mandatory trigger for reviewing the ICT risk-management framework, and Article 6(9)'s multi-vendor strategy option is the structural answer to the cloud dependency the attack exposed.",
             "strength": Strength.weak},
            {"source_id": "dora", "provision": "Article 13",
             "relevance": "Article 13(2) requires a post-incident review, covering response promptness, forensic quality, escalation, and communication, once the major incident has disrupted core trading activities.",
             "strength": Strength.weak},
            {"source_id": "dora", "provision": "Article 5",
             "relevance": "Article 5 places the management body over the follow-up: it must approve, oversee, and periodically review the revised ICT business continuity policy and response and recovery plans the post-incident review produces.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The firm stays fully responsible for DORA compliance despite the cloud provider's failure; the critical trading infrastructure's dependence on a single provider requires concentration-risk analysis and substitutability planning, with audit rights, termination grounds and exit strategies governing the relationship, and the contract securing incident assistance and tested contingency plans.",
        citations=[
            {"source_id": "dora", "provision": "Article 28",
             "relevance": "Article 28 keeps the firm fully responsible for DORA compliance despite the cloud provider's failure, and its audit rights, termination grounds including evidenced provider weaknesses, and exit strategies now govern the relationship's future.",
             "strength": Strength.strong},
            {"source_id": "dora", "provision": "Article 29",
             "relevance": "Article 29's concentration-risk duties apply squarely, since the firm's critical trading infrastructure depends on a single cloud provider whose failure endangered critical functions, requiring substitutability analysis and alternative solutions.",
             "strength": Strength.moderate},
            {"source_id": "dora", "provision": "Article 30",
             "relevance": "Article 30 requires the cloud contract to secure incident assistance, notification of material developments, tested contingency plans, and ongoing performance monitoring, the provisions the firm invokes during and after the attack.",
             "strength": Strength.moderate},
        ],
    ),
    ExpectedFinding(
        statement="Once established, the unauthorized access to systems holding client information is a personal data breach, obliging notification to the supervisory authority within 72 hours with phased information while the investigation proceeds.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 4",
             "relevance": "Article 4(12) makes the possible unauthorized access to systems holding client information a personal data breach once established, grounding the GDPR track of the response.",
             "strength": Strength.moderate},
            {"source_id": "gdpr", "provision": "Article 33",
             "relevance": "Article 33 obliges the firm to notify the supervisory authority within 72 hours of awareness of the potential client-data access, with phased information under Article 33(4) while the investigation proceeds and documentation under Article 33(5).",
             "strength": Strength.strong},
        ],
    ),
    ExpectedFinding(
        statement="Clients must be informed where attacker access creates high risk, with the encryption condition available where affected data were unintelligible.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 34",
             "relevance": "Article 34 requires communication to affected clients where attacker access creates high risk, with Article 34(3)'s encryption condition available where affected data were unintelligible.",
             "strength": Strength.moderate},
        ],
    ),
    ExpectedFinding(
        statement="Client data and trading systems must be secured with measures covering availability and timely restoration, framed by the integrity and confidentiality principle.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 32",
             "relevance": "Article 32's security duties, including availability and timely restoration for client data, frame both the incident's prevention dimension and the recovery architecture.",
             "strength": Strength.moderate},
            {"source_id": "gdpr", "provision": "Article 5",
             "relevance": "Article 5(1)(f)'s integrity-and-confidentiality principle frames the security hardening the firm owes client data after the attack.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The firm's remediation must produce demonstrable compliance measures under the accountability duty.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 24",
             "relevance": "Article 24's accountability duty frames the demonstrable compliance measures the firm's remediation must produce.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The firm's processing of client personal data, including through the AI system, needs an Article 6 lawful basis - contract necessity or legitimate interests for the analysis - assessed before the system's use.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 6",
             "relevance": "Article 6(1) requires a lawful basis for the firm's processing of client personal data through the trading-recommendation system, with legitimate interests carrying the balancing test against clients' rights.",
             "strength": Strength.moderate},
        ],
    ),
    ExpectedFinding(
        statement="Where the cloud provider processes client personal data on the firm's behalf, the arrangement must rest on an Article 28 processor contract whose terms secure the provider's assistance with the breach response.",
        citations=[
            {"source_id": "gdpr", "provision": "Article 28",
             "relevance": "Article 28 obliges the firm to engage the cloud host only as a processor under a contract carrying Article 28(3)'s guarantees, with Article 28(3)(f) requiring the provider's assistance with the breach and security duties.",
             "strength": Strength.moderate},
        ],
    ),
    ExpectedFinding(
        statement="The firm is a Union deployer owing AI Act duties, and the definitions of deployer and intended purpose frame the classification of the trading-recommendation tool.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 2",
             "relevance": "Article 2(1)(b) establishes the firm's status as a Union deployer owing AI Act duties, the entry point for the classification analysis of the trading-recommendation system.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Article 3",
             "relevance": "Article 3's definitions of deployer and intended purpose frame the classification of a recommendation tool used by traders under the firm's authority, whose intended purpose drives the high-risk question.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The classification analysis tests the product-safety route and the Annex III creditworthiness wording, and on the stated facts leaves the market-recommendation tool in the lighter tier - a shift toward evaluating clients' creditworthiness would change that.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 6",
             "relevance": "Article 6(1) supplies the product-safety high-risk route through Annex I, which the classification analysis tests for this decision-support tool and, on the stated facts, leaves the system in the lighter tier.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Annex 1",
             "relevance": "Annex I lists the Union harmonisation legislation whose safety components trigger Article 6(1) high-risk classification, the alternative route assessed for the trading system, which on the stated facts leaves a recommendation engine in the lighter tier.",
             "strength": Strength.weak},
            {"source_id": "ai-act", "provision": "Annex 3",
             "relevance": "Annex III point 5(b) frames the decisive classification test: its creditworthiness wording leaves a market-analysis and recommendation tool in the lighter tier on these facts, and any shift toward evaluating clients' creditworthiness would change that.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The traders using the system's recommendations need sufficient AI literacy - the duty the analysis finds clearly applicable.",
        citations=[
            {"source_id": "ai-act", "provision": "Article 4",
             "relevance": "Article 4 obliges the firm to ensure its traders using the system's recommendations have sufficient AI literacy, the AI Act duty the analysis finds clearly applicable to this deployment.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="A size and status check applies: small and non-interconnected investment firms follow the simplified framework, while the incident-reporting duties continue to apply in full.",
        citations=[
            {"source_id": "dora", "provision": "Article 16",
             "relevance": "Article 16 requires a size and status check: small and non-interconnected investment firms follow the simplified framework while the incident-reporting duties of Articles 17-19 continue to apply in full.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The firm must identify, classify and document its ICT-supported business functions and the assets supporting them - including the critical trading infrastructure and its dependence on the cloud provider - and run risk assessments around significant changes.",
        citations=[
            {"source_id": "dora", "provision": "Article 8",
             "relevance": "Article 8 requires the firm to identify and document all ICT-supported business functions and the assets supporting them, the classification grounding the criticality and dependency analysis the attack exposed.",
             "strength": Strength.weak},
        ],
    ),
    ExpectedFinding(
        statement="The firm must maintain detection mechanisms that promptly surface anomalous activities, including ICT-related incidents and cyber-attacks, across the critical trading infrastructure.",
        citations=[
            {"source_id": "dora", "provision": "Article 10",
             "relevance": "Article 10 obliges the firm to operate detection mechanisms that promptly identify anomalous activities and cyber-attacks and single points of failure, the capability whose limits the attack exposed.",
             "strength": Strength.weak},
        ],
    ),
]
LIVE_EVAL_SCENARIOS: List[EvalScenario] = [
    EvalScenario(
        id="live-retailer-breach",
        scenario_id="online-retailer-breach",
        description="An online retailer discovers that attackers gained unauthorized access to a database containing customers' names, email addresses, postal addresses, and order information. The company has confirmed that the attackers were able to access the data but does not yet know whether the information was downloaded.",
        question="What regulatory obligations should the company consider following this personal data breach?",
        expected=_RETAILER_BREACH_EXPECTED,
    ),
    EvalScenario(
        id="live-ai-recruitment-screening",
        scenario_id="hr-ai-cv-screening",
        description="A company uses an AI system to screen job applications. The system analyses applicants' CVs and recorded video interviews and assigns each applicant a score that recruiters use to prioritize candidates for interviews.",
        question="What regulatory requirements should the company consider before using this AI system for recruitment?",
        expected=_AI_RECRUITMENT_SCREENING_EXPECTED,
    ),
    EvalScenario(
        id="live-bank-cloud-outage",
        scenario_id="bank-cloud-outage",
        description="A bank relies on an external cloud provider to host critical systems used for online banking. A major technical failure at the cloud provider makes the bank's online banking services unavailable to customers for several hours.",
        question="What regulatory obligations should the bank consider in relation to this incident, its reliance on the cloud provider, and its data protection obligations towards customer data on the disrupted systems?",
        expected=_BANK_CLOUD_OUTAGE_EXPECTED,
    ),
    EvalScenario(
        id="live-employee-productivity-monitoring",
        scenario_id="employee-productivity-monitoring",
        description="A company deploys software that continuously records employees' computer activity, including applications used, websites visited, and time spent working. An AI system uses this information to generate an individual productivity score for each employee, which managers can use when evaluating performance.",
        question="What legal requirements should the company consider when using this system to monitor and evaluate employees?",
        expected=_EMPLOYEE_PRODUCTIVITY_MONITORING_EXPECTED,
    ),
    EvalScenario(
        id="live-telecom-chatbot",
        scenario_id="telecom-genai-chatbot",
        description="A telecommunications company deploys a generative AI chatbot on its website to answer customer-service questions. Customers can enter information such as their name, account number, and address when describing their problems. The chatbot provides answers but cannot make changes to customer accounts. The chatbot is powered by a general-purpose AI model the company licenses from an external model provider.",
        question="What regulatory requirements should the company consider when operating this chatbot and processing the information provided by customers?",
        expected=_TELECOM_GENAI_CHATBOT_EXPECTED,
    ),
    EvalScenario(
        id="live-fintech-loan-recommendations",
        scenario_id="fintech-loan-recommendations",
        description="A fintech company uses an AI system to analyse applicants' income, employment history, existing debts, and other financial information to produce a recommendation on whether a loan should be approved. Employees review the recommendation before making the final decision. The company uses the system for all consumer loan applications in the EU.",
        question="What EU regulatory requirements should the company consider when using this system to assess and decide on consumer loan applications, including the lawfulness of processing applicants' financial data?",
        expected=_FINTECH_LOAN_RECOMMENDATIONS_EXPECTED,
    ),
    EvalScenario(
        id="live-insurance-health-pricing",
        scenario_id="insurer-health-ai-pricing",
        description="An insurance company uses an AI system to analyse applicants' medical information and other personal data when determining eligibility and calculating premiums for life insurance policies. The system produces a recommended premium automatically, which an employee can review before the policy is issued.",
        question="What regulatory requirements should the insurer consider regarding the use of AI and the processing of applicants' health information?",
        expected=_INSURANCE_HEALTH_PRICING_EXPECTED,
    ),
    EvalScenario(
        id="live-ransomware-investment-firm",
        scenario_id="investment-firm-ransomware",
        description="An investment firm suffers a ransomware attack that disrupts its trading platform and prevents employees from accessing several internal systems. Some systems affected by the attack contain client personal data. The firm restores its services after several hours and is investigating whether client data was accessed by the attackers.",
        question="What regulatory obligations should the firm consider following the incident?",
        expected=_RANSOMWARE_INVESTMENT_FIRM_EXPECTED,
    ),
    EvalScenario(
        id="live-genai-customer-service",
        scenario_id="financial-institution-genai-assistant",
        description="A European financial institution deploys a generative AI system provided by an external technology company to assist customer-service employees. The system processes customer conversations that may contain account information and personal data. The AI service is hosted on infrastructure operated by another cloud provider, and the financial institution has limited visibility into how the underlying service is operated.",
        question="What regulatory requirements should the financial institution consider regarding the AI system, the processing of customer information, its reliance on external technology providers, and any transfer of customer conversations outside the EEA?",
        expected=_GENAI_CUSTOMER_SERVICE_ASSISTANT_EXPECTED,
    ),
    EvalScenario(
        id="live-ai-trading-cloud-attack",
        scenario_id="investment-firm-ai-cloud-attack",
        description="A European investment firm uses an AI system hosted by a third-party cloud provider to analyse market and client information and generate recommendations that traders use when making investment decisions. The system processes personal data relating to clients and is integrated into the firm's critical trading infrastructure. The cloud provider suffers a major cyberattack that disrupts the firm's trading systems for more than a day. During the incident, the firm discovers that attackers may also have gained unauthorized access to systems containing client information.",
        question="What EU regulatory obligations should the investment firm consider in relation to the AI system, the processing of client data, the ICT incident, its dependence on the third-party cloud provider, and its data protection obligations for the client personal data the attackers may have accessed?",
        expected=_AI_TRADING_CLOUD_ATTACK_EXPECTED,
    ),
]


# The audit-dump form of one expected Finding: the authored statement and the
# ground-truth provision labels as written, each with its authored relevance
# summary when one exists and the operator's rating (ADR-0011).
def _expected_dump(finding: ExpectedFinding) -> ExpectedFindingDump:
    return {
        "statement": finding.statement,
        "citations": [
            {
                "label": f"{c['source_id']} {c['provision']}",
                "relevance": c.get("relevance"),
                "strength": c["strength"].value,
            }
            for c in finding.citations
        ],
    }


def scenario_definition_hash(scenario: EvalScenario) -> str:
    """SHA-256 over a case's full definition — id, routing id, question,
    description, and the expected findings exactly as the audit dump renders
    them. A checkpointed result records its case's hash so a later resume can
    refuse a stale one: ground truth that changed under a run id must never
    be silently substituted for the work already measured."""
    payload = json.dumps({
        "id": scenario.id,
        "scenario_id": scenario.scenario_id,
        "question": scenario.question,
        "description": scenario.description,
        "expected": [_expected_dump(f) for f in scenario.expected],
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# The audit-dump form of one produced Finding: the statement as the LLM worded
# it, its Strength, and each Citation's structural target with its quote and
# the answer-wide fields — Provision relevance and rated Citation strength —
# read straight off the Answer's own Citation for that target (``None`` where
# the Summarizer shipped nothing or no rating, ADR-0011).
def _produced_dump(
    finding: ProducedFinding,
    citations_by_target: Dict[ProvisionTarget, Citation],
) -> ProducedFindingDump:
    return {
        "statement": finding.statement,
        "strength": finding.strength.value,
        "citations": [
            {
                "target": format_provision_target(c.provision_target),
                "quote": c.quote,
                "relevance": citations_by_target[c.provision_target].relevance,
                "strength": (
                    answer_strength.value
                    if (answer_strength := citations_by_target[c.provision_target].strength) is not None
                    else None
                ),
            }
            for c in finding.citations
        ],
    }


def _mean(scores: Iterable[Optional[float]]) -> Optional[float]:
    """The mean over the scores that measured — ``None`` only when none did."""
    measured = [score for score in scores if score is not None]
    return sum(measured) / len(measured) if measured else None


class SummaryFidelityMeasurement(NamedTuple):
    """One case's fidelity measurement (issue #83): the score the report
    carries plus every judged pair's verdict record behind it — so the
    per-pair data the judge decided escapes with the score instead of being
    averaged away inside the scorer."""

    fidelity: float
    verdicts: List[PairFidelityVerdict]


# The Known-limitation prefix that marks degraded Summarizer coverage — the
# one limitation whose text the result shape persists (issue #83). The
# workflow's own limitation constants must keep starting with it; wording
# drift is pinned loudly by test.
SUMMARIZER_LIMITATION_PREFIX = "Known limitation: Provision relevance"


def _summarizer_limitation(known_limitations: Iterable[str]) -> Optional[str]:
    """The response's fidelity-relevant Known limitation, if any: the
    Provision-relevance limitation the Summarizer's degraded pass owes the
    Answer. The standing limitations (English-only, prototype) never match,
    and none is ever fabricated."""
    for limitation in known_limitations:
        if limitation.startswith(SUMMARIZER_LIMITATION_PREFIX):
            return limitation
    return None


def _summary_fidelity(
    scenario: EvalScenario,
    citations_by_target: Dict[ProvisionTarget, Citation],
    judge: SummaryJudge,
) -> Optional[SummaryFidelityMeasurement]:
    """One case's summary fidelity (ADR-0010): the strict rubric judge over
    provision-aligned expected and produced Provision relevance.

    Ground truth must summarize at least one cited provision for the
    component to measure at all (the labels #51 authored). Pairs align by
    structural target, one per provision: a target the Answer cites bare —
    ground truth summarizes it, the Summarizer shipped nothing — floors its
    share of the score deterministically, with no judge call; a target
    either side never touches is coverage's business, never a fidelity pair.
    The produced relevance reads off the Answer's own Citations, the bundled
    carrier of the answer-wide fields.

    The judge's verdict records travel with the score (issue #83): a judged
    pair carries its flags and its score verbatim; a floored pair carries
    nothing — its failure mode is the Known limitation's business.
    """
    expected_summaries = expected_relevance_summaries(scenario.expected)
    if not expected_summaries:
        return None
    pairs: List[RelevancePair] = []
    floored_count = 0
    for target, expected_relevance in expected_summaries.items():
        if target not in citations_by_target:
            continue
        produced_relevance = citations_by_target[target].relevance
        if not produced_relevance:
            # A provision the Answer cites without a relevance statement
            # is infidelitous by construction — the deterministic floor,
            # with nothing for a judge to compare.
            floored_count += 1
            continue
        pairs.append(RelevancePair(
            ref=f"P{len(pairs) + 1}",
            provision=format_provision_target(target),
            expected=expected_relevance,
            produced=produced_relevance,
        ))
    if not pairs and not floored_count:
        return None
    verdicts_by_ref = judge.compare(pairs) if pairs else {}
    verdicts = [verdicts_by_ref[pair.ref] for pair in pairs]
    fidelity = (
        sum(verdict["score"] for verdict in verdicts) + FIDELITY_FLOOR * floored_count
    ) / (len(verdicts) + floored_count)
    return SummaryFidelityMeasurement(fidelity=fidelity, verdicts=verdicts)


def evaluate_scenarios(
    scenarios: List[EvalScenario],
    respond: Callable[[AnalyzeRequest], AnalyzeResponse],
    mode: Mode,
    judge: Optional[SummaryJudge] = None,
    on_result: Optional[Callable[[EvalScenarioResult], None]] = None,
) -> EvalReport:
    """Run each Scenario through ``respond`` and score the produced Findings.

    The one evaluation loop both runners share: ``respond`` answers one
    request (the Demo harness calls analyze directly, the Live runner crosses
    HTTP), and everything after — Findings mapping, the two per-case
    component scores, the audit dump, the per-component aggregate means —
    happens identically for every mode. The mode is pinned per request
    (ADR-0008): every request carries it explicitly. Summary fidelity runs
    only where a judge is supplied (the Demo tripwires run judgeless); the
    fidelity judge is called at most once per case (ADR-0010).

    ``on_result`` is the checkpoint seam: invoked with each case's result the
    moment it is scored, before the next case starts, so the Live runner can
    persist progress mid-run without the harness ever touching the
    filesystem — a callback parameter, no I/O here.
    """
    scenario_results: List[EvalScenarioResult] = []
    for scenario in scenarios:
        request = AnalyzeRequest(
            scenario=Scenario(id=scenario.scenario_id, description=scenario.description),
            question=scenario.question,
            mode=mode,
        )
        response = respond(request)
        produced = [
            ProducedFinding(statement=f.statement, strength=f.strength, citations=f.citations)
            for f in response.answer.findings
        ]
        # The Answer's own Citations are the bundled carrier of the two
        # answer-wide fields: one map serves the dump and the fidelity judge.
        citations_by_target = {c.provision_target: c for c in response.answer.citations}
        measurement = (
            _summary_fidelity(scenario, citations_by_target, judge)
            if judge is not None
            else None
        )
        result = EvalScenarioResult(
            id=scenario.id,
            **coverage_scores(scenario.expected, produced),
            expected=[_expected_dump(f) for f in scenario.expected],
            produced=[_produced_dump(f, citations_by_target) for f in produced],
            summary_fidelity=measurement.fidelity if measurement is not None else None,
            fidelity_verdicts=list(measurement.verdicts) if measurement is not None else [],
            summarizer_limitation=_summarizer_limitation(response.known_limitations),
        )
        scenario_results.append(result)
        if on_result is not None:
            on_result(result)

    return EvalReport(
        scenarios=scenario_results,
        mean_f1=sum(scenario_result.f1 for scenario_result in scenario_results) / len(scenario_results),
        mean_summary_fidelity=_mean(result.summary_fidelity for result in scenario_results),
    )


def run_eval() -> EvalReport:
    """Run every curated scenario against the analyze workflow and score the produced Findings.

    Drives the app's analyze entry point directly — no HTTP, and judgeless:
    the Demo tripwires run in CI without a provider, so summary fidelity
    reports unmeasured there while coverage scores.
    Per ADR-0001 the curated cases are Demo-mode tripwires, so the harness
    pins Demo mode per request (ADR-0008) regardless of how the app booted.
    """

    from .main import analyze

    def respond(request: AnalyzeRequest) -> AnalyzeResponse:
        # No HTTP: the harness drives the analyze entry point directly and
        # carries no progress request id, so nothing is registered in the
        # progress registry.
        return analyze(request, request_id=None)

    return evaluate_scenarios(CURATED_SCENARIOS, respond, mode=Mode.demo)


if __name__ == "__main__":
    from dataclasses import asdict

    print(json.dumps(asdict(run_eval()), indent=2))
