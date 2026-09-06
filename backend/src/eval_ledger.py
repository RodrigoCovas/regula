"""The eval ledger (spec #79, ticket #80): weight-aware spurious/missed
accounting over a Live-eval run artifact, re-scored offline.

One pure seam — ``ledger_rows``: a parsed run artifact (either flavor the
Live eval writes: the ``--output`` report artifact or a run checkpoint
under ``data/eval-runs/``, both carrying the same per-case records) plus
the expected keys to score against; out: one row per artifact Scenario
naming the unique spurious produced Citation targets and the missed
expected targets, each with its Citation-strength weight (strong 50 /
moderate 5 / weak 1), plus the precision/recall/F1 re-scored against those
keys. The numbers come from the harness's own ``coverage_scores`` — the
ledger never re-implements the arithmetic it reports — so a re-scored row
is exactly what the harness would score for the same production against
the same keys.

Re-scoring reads whatever expected keys the caller supplies, never the
expected dump an artifact stored: ground truth moves on (the operator's
record is edited and re-transcribed), while the artifact's production is
frozen — auditing an old run against today's keys is the ledger's whole
point. Spurious targets carry the answer's own rated Citation strength
(the only weight a produced target has; unrated citations name none);
missed targets carry the operator's rating under the max-rule, the same
derivation that weights the harness's recall.

Summary fidelity is excluded by design: the judge is LLM-only and per-pair
verdicts do not exist in old artifacts, so an offline ledger cannot re-score
it without lying about what it measured. Coverage only.

No LLM calls, ever: the module reads no settings, builds no provider
client, and works with no key configured — auditing a paid run must never
spend another.

Run from the repository root:

    python -m backend.src.eval_ledger --artifact logs/live-eval-report-2026-09-04-glm53_v7.json

or inside the stack:

    docker compose exec backend python -m backend.src.eval_ledger --artifact <path>
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .eval_harness import (
    LIVE_EVAL_SCENARIOS,
    STRENGTH_WEIGHTS,
    ExpectedFinding,
    ProducedFinding,
    coverage_scores,
    expected_target_strengths,
    expected_target_weights,
    format_provision_target,
    parse_provision,
)
from .models import (
    PROVISION_NUMBER_FIELDS,
    Citation,
    ProvisionTarget,
    Strength,
    max_rule_strengths,
)


@dataclass(frozen=True)
class MissedTarget:
    """One current expected target the artifact's production never cited:
    the operator's rating and its recall weight (strong 50 / moderate 5 /
    weak 1)."""

    target: ProvisionTarget
    strength: Strength
    weight: int


@dataclass(frozen=True)
class SpuriousTarget:
    """One produced Citation target no current expected key names: the
    answer's own Citation rating and its weight — ``None`` where the
    Summarizer shipped no rating."""

    target: ProvisionTarget
    strength: Optional[Strength]
    weight: Optional[int]


@dataclass(frozen=True)
class LedgerRow:
    """One artifact Scenario's accounting: the coverage re-scored against
    the supplied keys, plus the weight-aware spurious/missed lists — each
    sorted by source, kind, and number, so every read of an artifact
    produces the same rows."""

    case_id: str
    precision: float
    recall: float
    f1: float
    missed: List[MissedTarget]
    spurious: List[SpuriousTarget]


def _target_sort_key(entry):
    target = entry.target
    return (target.source_id, target.kind.value, target.number)


def _artifact_cases(artifact: object) -> List[dict]:
    """The artifact's case records, shape-checked just past what the ledger
    reads: extra fields (a future result shape's per-pair verdicts) ride
    along unread, missing ones refuse loudly — a malformed artifact must
    never turn into a quietly partial accounting."""
    if not isinstance(artifact, dict):
        raise ValueError("The parsed artifact is not a JSON object")
    cases = artifact.get("scenarios")
    if not isinstance(cases, list):
        raise ValueError("The parsed artifact is not a Live-eval run artifact: no 'scenarios' list")
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("A case record is not a JSON object")
        if not isinstance(case.get("id"), str):
            raise ValueError("A case record has no string id")
    return cases


def _rated_strength(case_id: str, raw: object, what: str) -> Strength:
    if not isinstance(raw, str):
        raise ValueError(f"Case {case_id!r} has a {what} that is not a strength string: {raw!r}")
    try:
        return Strength(raw)
    except ValueError as error:
        raise ValueError(f"Case {case_id!r} has an invalid {what} {raw!r}") from error


def _dump_target(case_id: str, raw: object) -> ProvisionTarget:
    """The structural target behind an audit dump's rendered form — the
    exact inverse of ``format_provision_target``."""
    if not isinstance(raw, str):
        raise ValueError(f"Case {case_id!r} has a produced citation without a target")
    source_id, _, label = raw.partition(" ")
    try:
        return parse_provision(source_id, label)
    except ValueError as error:
        raise ValueError(
            f"Case {case_id!r} has an unparseable produced citation target {raw!r}"
        ) from error


def _produced_citations(case_id: str, citations: object) -> List[Citation]:
    if not isinstance(citations, list):
        raise ValueError(f"Case {case_id!r} has a produced record whose citations are not a list")
    parsed: List[Citation] = []
    for entry in citations:
        if not isinstance(entry, dict):
            raise ValueError(f"Case {case_id!r} has a produced citation that is not a JSON object")
        rating = entry.get("strength")
        target = _dump_target(case_id, entry.get("target"))
        parsed.append(Citation.model_validate({
            "source_id": target.source_id,
            PROVISION_NUMBER_FIELDS[target.kind]: target.number,
            "strength": None if rating is None else _rated_strength(case_id, rating, "Citation strength rating"),
        }))
    return parsed


def _produced_findings(case_id: str, produced: object) -> List[ProducedFinding]:
    """The artifact's produced dump rebuilt into the harness's own shape, so
    the re-score rides ``coverage_scores`` unchanged. Only the fields the
    dump contract carries are accepted — a wrong-typed one refuses here as
    a malformed artifact, never crashes mid-print."""
    if not isinstance(produced, list):
        raise ValueError(f"Case {case_id!r} has a produced dump that is not a list")
    findings: List[ProducedFinding] = []
    for record in produced:
        if not isinstance(record, dict):
            raise ValueError(f"Case {case_id!r} has a produced record that is not a JSON object")
        statement = record.get("statement")
        if not isinstance(statement, str):
            raise ValueError(f"Case {case_id!r} has a produced record without a statement")
        findings.append(ProducedFinding(
            statement=statement,
            strength=_rated_strength(case_id, record.get("strength"), "Finding strength"),
            citations=_produced_citations(case_id, record.get("citations")),
        ))
    return findings


def ledger_rows(
    artifact: object,
    expected_keys: Dict[str, List[ExpectedFinding]],
) -> List[LedgerRow]:
    """The pure seam: a parsed run artifact plus the expected keys to score
    against → one ledger row per artifact Scenario, in artifact order.

    Coverage is re-scored against ``expected_keys`` — whatever is current —
    through the harness's ``coverage_scores``; the missed list is the
    current expected targets the production never cited, each with its
    max-rule weight; the spurious list is the produced targets no current
    key names, each with the answer's own rating. An artifact case the keys
    do not know refuses the run: the two records disagree, and silently
    dropping a case would read as a clean audit."""
    rows: List[LedgerRow] = []
    for case in _artifact_cases(artifact):
        case_id = case["id"]
        if case_id not in expected_keys:
            raise ValueError(
                f"Artifact case {case_id!r} has no current expected keys: the artifact "
                "and the supplied ground truth disagree on the case list, so it "
                "cannot be re-scored."
            )
        current = expected_keys[case_id]
        produced = _produced_findings(case_id, case.get("produced"))
        produced_targets = {citation.provision_target for finding in produced for citation in finding.citations}
        expected_strengths = expected_target_strengths(current)
        expected_weights = expected_target_weights(current)

        missed = sorted(
            (
                MissedTarget(target, expected_strengths[target], expected_weights[target])
                for target in expected_strengths.keys() - produced_targets
            ),
            key=_target_sort_key,
        )
        # The answer's own ratings, strongest per target where several
        # Citations cite one — the only weight a produced target carries.
        ratings = max_rule_strengths(
            (citation.provision_target, citation.strength)
            for finding in produced
            for citation in finding.citations
            if citation.strength is not None
        )
        spurious: List[SpuriousTarget] = []
        for target in produced_targets - expected_strengths.keys():
            strength = ratings.get(target)
            spurious.append(SpuriousTarget(
                target,
                strength,
                STRENGTH_WEIGHTS[strength] if strength is not None else None,
            ))
        spurious.sort(key=_target_sort_key)
        scores = coverage_scores(current, produced)
        rows.append(LedgerRow(
            case_id=case_id,
            precision=scores["precision"],
            recall=scores["recall"],
            f1=scores["f1"],
            missed=missed,
            spurious=spurious,
        ))
    return rows


def shipped_expected_keys() -> Dict[str, List[ExpectedFinding]]:
    """The currently shipped ground truth: the curated Live cases' expected
    Findings keyed by case id — whatever ``eval_harness`` ships now, never
    a stored copy."""
    return {scenario.id: scenario.expected for scenario in LIVE_EVAL_SCENARIOS}


def _fmt(score: float) -> str:
    """Three decimals, matching the runner's report style."""
    return f"{score:.3f}"


def print_ledger(
    artifact_path: Path,
    artifact: dict,
    rows: List[LedgerRow],
    expected_keys: Dict[str, List[ExpectedFinding]],
) -> None:
    """The operator-facing ledger: which artifact and whose model (results
    are model-sensitive by design, ADR-0009), then per case the re-scored
    numbers and both weight-aware lists, then the re-scored means. Spurious
    entries speak the answer's rating ('rated …', or 'unrated'), missed
    entries the operator's expectation ('expected …'). Shipped cases the
    artifact does not measure are named, so a partial audit never reads as
    a full one."""
    print(f"Eval ledger: {artifact_path}")
    model = artifact.get("llm_model")
    if isinstance(model, str):
        print(f"Model: {model}")
    print(f"Re-scored against the shipped ground truth: {len(rows)} case(s) in the artifact")
    for row in rows:
        print(
            f"  {row.case_id}: precision={_fmt(row.precision)} "
            f"recall={_fmt(row.recall)} F1={_fmt(row.f1)}"
        )
        print(f"    missed ({len(row.missed)}):")
        for missed_entry in row.missed:
            print(
                f"      {format_provision_target(missed_entry.target)} "
                f"(expected {missed_entry.strength.value}, weight {missed_entry.weight})"
            )
        print(f"    spurious ({len(row.spurious)}):")
        for spurious_entry in row.spurious:
            if spurious_entry.strength is None:
                print(f"      {format_provision_target(spurious_entry.target)} (unrated)")
            else:
                print(
                    f"      {format_provision_target(spurious_entry.target)} "
                    f"(rated {spurious_entry.strength.value}, weight {spurious_entry.weight})"
                )
    if rows:
        cases = len(rows)
        print(
            f"Mean re-scored precision: {_fmt(sum(row.precision for row in rows) / cases)} "
            f"recall: {_fmt(sum(row.recall for row in rows) / cases)} "
            f"F1: {_fmt(sum(row.f1 for row in rows) / cases)}"
        )
    unmeasured = sorted(set(expected_keys) - {row.case_id for row in rows})
    if unmeasured:
        print(f"Shipped cases this artifact does not measure: {', '.join(unmeasured)}")


def main(argv: list | None = None) -> int:
    """The CLI entry point: refuse with exit 1 when the artifact is missing,
    unreadable, or disagrees with the shipped ground truth; otherwise print
    the ledger and exit 0. ``argv`` defaults to the process arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Ledger a Live-eval run artifact: per-Scenario spurious/missed Citation "
            "accounting with weights, re-scored against the shipped ground truth. "
            "Coverage only — no LLM calls, ever."
        ),
    )
    parser.add_argument(
        "--artifact",
        "-a",
        type=Path,
        required=True,
        help="run artifact or checkpoint (JSON) to audit",
    )
    args = parser.parse_args(argv)

    try:
        artifact = json.loads(args.artifact.read_text())
        keys = shipped_expected_keys()
        rows = ledger_rows(artifact, keys)
    except (OSError, ValueError) as error:
        print(f"Eval ledger refused: {error}", file=sys.stderr)
        return 1
    print_ledger(args.artifact, artifact, rows, keys)
    return 0


if __name__ == "__main__":
    sys.exit(main())
