"""The eval ledger (spec #79, ticket #80): weight-aware spurious/missed
accounting over a Live-eval run artifact, re-scored offline.

The one new pure seam is ``ledger_rows`` — a parsed artifact plus the
expected keys to score against in, per-Scenario ledger rows out — tested
here with fixture artifacts whose stored keys deliberately differ from the
keys they are scored against. The CLI (``python -m
backend.src.eval_ledger``) wraps the seam; its tests monkeypatch the
shipped case list the same way the runner's tests do, so the pins stay
stable when ground truth is edited. The ledger performs no LLM calls: no
key, no settings, no provider — auditing a paid run must never spend
another.
"""

import json
from typing import Dict, List, Optional, Tuple

import pytest

from src.eval_harness import (
    LIVE_EVAL_SCENARIOS,
    EvalScenario,
    ExpectedFinding,
    parse_provision,
)
from src.eval_ledger import (
    MissedTarget,
    SpuriousTarget,
    ledger_rows,
    main,
    shipped_expected_keys,
)
from src.models import ProvisionTarget, Strength


def _expected(source_id: str, label: str, strength: Strength) -> ExpectedFinding:
    return ExpectedFinding(
        statement=f"expected: {source_id} {label}",
        citations=[{"source_id": source_id, "provision": label, "strength": strength}],
    )


def _produced_dump(*citations: Tuple[str, Optional[str]]) -> dict:
    """One produced Finding record as the artifact writes it: the statement,
    the Finding Strength, and each Citation's structural target with its
    rated Citation strength (``None`` where the Summarizer shipped none)."""
    return {
        "statement": "produced wording",
        "strength": "moderate",
        "citations": [
            {"target": target, "quote": None, "relevance": None, "strength": rating}
            for target, rating in citations
        ],
    }


def _artifact(cases: List[dict]) -> dict:
    """A report artifact as the Live eval CLI writes it."""
    return {
        "mode": "live",
        "generated_at": "2026-09-04T10:00:00+00:00",
        "llm_model": "test-model",
        "scenarios": cases,
    }


def _case(case_id: str, *produced: dict) -> dict:
    """One stored case record — its stored scores say a perfect run, so any
    ledger row that disagrees is the re-scoring talking, not the fixture."""
    return {
        "id": case_id,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "expected": [],
        "produced": list(produced),
        "summary_fidelity": None,
    }


def _target(source_id: str, label: str) -> ProvisionTarget:
    return parse_provision(source_id, label)


# --- The pure seam: artifact + expected keys → ledger rows -------------------


def test_rescores_against_the_current_keys_not_the_ones_the_artifact_stored():
    """The artifact's stored scores describe the ground truth it ran with;
    the ledger scores whatever keys are current. Here the artifact measured
    a perfect hit on Article 22 — but the current keys expect only Article
    5, so the row must report a zero re-score with Article 5 missed and
    Article 22 spurious."""
    artifact = _artifact([
        _case("case", _produced_dump(("gdpr Article 22", "strong"))),
    ])
    current = {"case": [_expected("gdpr", "Article 5", Strength.strong)]}

    (row,) = ledger_rows(artifact, current)

    assert (row.precision, row.recall, row.f1) == (0.0, 0.0, 0.0)
    assert row.missed == [MissedTarget(_target("gdpr", "Article 5"), Strength.strong, 50)]
    assert row.spurious == [SpuriousTarget(_target("gdpr", "Article 22"), Strength.strong, 50)]


def test_weight_aware_accounting_scores_through_the_harness_arithmetic():
    """The re-scored numbers come from the same coverage arithmetic the
    harness scores runs with: recall is the strength-weighted share of
    expected targets covered, precision counts every off-target produced
    Citation against itself. Hand-computed: one weak hit of a 50+1 total,
    one of two produced targets expected."""
    current = {
        "case": [
            _expected("gdpr", "Article 6", Strength.strong),
            _expected("ai-act", "Article 3", Strength.weak),
        ],
    }
    artifact = _artifact([
        _case("case", _produced_dump(("ai-act Article 3", "weak"), ("ai-act Article 99", None))),
    ])

    (row,) = ledger_rows(artifact, current)

    assert row.recall == pytest.approx(1 / 51)
    assert row.precision == pytest.approx(1 / 2)
    assert row.f1 == pytest.approx(2 * (1 / 51) * (1 / 2) / ((1 / 51) + (1 / 2)))


def test_missed_targets_carry_the_expected_weight():
    """A missed expected target names the operator's rating and its weight —
    strong 50 / moderate 5 / weak 1 — so a missed strong anchor is visibly
    costlier than a missed framing provision."""
    current = {
        "case": [
            _expected("gdpr", "Article 6", Strength.strong),
            _expected("gdpr", "Article 27", Strength.moderate),
            _expected("gdpr", "Article 3", Strength.weak),
        ],
    }
    artifact = _artifact([_case("case")])

    (row,) = ledger_rows(artifact, current)

    assert row.missed == [
        MissedTarget(_target("gdpr", "Article 3"), Strength.weak, 1),
        MissedTarget(_target("gdpr", "Article 6"), Strength.strong, 50),
        MissedTarget(_target("gdpr", "Article 27"), Strength.moderate, 5),
    ]
    assert row.spurious == []
    assert (row.precision, row.recall, row.f1) == (1.0, 0.0, 0.0)


def test_missed_target_weight_follows_the_expected_max_rule():
    """A target cited by several expected Findings carries its strongest
    rating's weight — the max-rule the harness weights recall by — never the
    sum or the last rating."""
    current = {
        "case": [
            _expected("gdpr", "Article 6", Strength.weak),
            _expected("gdpr", "Article 6(2)", Strength.strong),
            _expected("gdpr", "Article 27", Strength.moderate),
        ],
    }
    artifact = _artifact([_case("case", _produced_dump(("gdpr Article 27", "weak")))])

    (row,) = ledger_rows(artifact, current)

    assert row.missed == [MissedTarget(_target("gdpr", "Article 6"), Strength.strong, 50)]


def test_spurious_targets_are_unique_and_carry_their_produced_rating():
    """Set semantics: repeating one spurious target earns one row, and its
    weight is the answer's own Citation rating — the only weight a produced
    target has. A citation the Summarizer left unrated names no weight."""
    artifact = _artifact([
        _case(
            "case",
            _produced_dump(("dora Article 46", "moderate")),
            _produced_dump(("dora Article 46", None), ("gdpr Article 12", None)),
        ),
    ])
    current = {"case": [_expected("gdpr", "Article 5", Strength.weak)]}

    (row,) = ledger_rows(artifact, current)

    assert row.spurious == [
        SpuriousTarget(_target("dora", "Article 46"), Strength.moderate, 5),
        SpuriousTarget(_target("gdpr", "Article 12"), None, None),
    ]


def test_rows_follow_artifact_order_and_lists_sort_deterministically():
    """Rows come in the artifact's case order; within a row the missed and
    spurious lists sort by source, kind, and number — the same accounting
    every read of the artifact produces, stable across runs."""
    current = {
        "case-a": [_expected("gdpr", "Article 24", Strength.weak)],
        "case-b": [
            _expected("gdpr", "Article 55", Strength.weak),
            _expected("dora", "Article 2", Strength.moderate),
        ],
    }
    artifact = _artifact([
        _case("case-b", _produced_dump(("gdpr Article 58", "weak"), ("ai-act Article 99", None))),
        _case("case-a"),
    ])

    rows = ledger_rows(artifact, current)

    assert [row.case_id for row in rows] == ["case-b", "case-a"]
    assert [entry.target for entry in rows[0].missed] == [
        _target("dora", "Article 2"),
        _target("gdpr", "Article 55"),
    ]
    assert [entry.target for entry in rows[0].spurious] == [
        _target("ai-act", "Article 99"),
        _target("gdpr", "Article 58"),
    ]


def test_checkpoint_flavored_artifacts_ledger_the_same_way():
    """data/eval-runs/ checkpoint documents carry the same case records under
    run metadata instead of report metadata — the ledger reads both flavors,
    since a checkpoint is often the only record an aborted run left."""
    artifact = {
        "run_id": "glm53-v7-2026-09-04",
        "mode": "live",
        "started_at": "2026-09-04T10:00:00+00:00",
        "llm_model": "test-model",
        "scenario_hashes": {"case": "abc"},
        "scenarios": [_case("case", _produced_dump(("gdpr Article 33", "strong")))],
    }
    current = {"case": [_expected("gdpr", "Article 33", Strength.strong)]}

    (row,) = ledger_rows(artifact, current)

    assert (row.precision, row.recall, row.f1) == (1.0, 1.0, 1.0)
    assert row.missed == []
    assert row.spurious == []


def test_empty_current_keys_score_leakage_as_the_harness_does():
    """A case whose current ground truth expects nothing scores the locked
    empty rule: any production is spurious leakage scoring zero across the
    board."""
    artifact = _artifact([_case("case", _produced_dump(("gdpr Article 22", None)))])
    current: Dict[str, List[ExpectedFinding]] = {"case": []}

    (row,) = ledger_rows(artifact, current)

    assert (row.precision, row.recall, row.f1) == (0.0, 0.0, 0.0)
    assert row.missed == []
    assert row.spurious == [SpuriousTarget(_target("gdpr", "Article 22"), None, None)]


# --- Malformed input fails loudly, naming the case ---------------------------


@pytest.mark.parametrize("artifact", [
    [],
    "nope",
    {"mode": "live"},
    {"scenarios": "nope"},
    {"scenarios": ["not-a-dict"]},
    {"scenarios": [{"produced": []}]},
    {"scenarios": [{"id": "case"}]},
    {"scenarios": [{"id": "case", "produced": "nope"}]},
    {"scenarios": [{"id": "case", "produced": ["not-a-dict"]}]},
])
def test_malformed_artifacts_fail_loudly(artifact):
    """Keys for the fixture's case id, so each malformed record reaches the
    guard that targets it instead of stopping at the unknown-case refusal."""
    with pytest.raises(ValueError, match="artifact|[Cc]ase"):
        ledger_rows(artifact, {"case": []})


def test_a_produced_record_without_a_statement_or_strength_fails_loudly():
    with pytest.raises(ValueError, match="statement"):
        ledger_rows(_artifact([_case("case", {"citations": []})]), {"case": []})
    with pytest.raises(ValueError, match="strength"):
        ledger_rows(_artifact([_case("case", {"statement": "wording", "citations": []})]), {"case": []})


def test_an_unparseable_produced_target_fails_loudly():
    artifact = _artifact([_case("case", _produced_dump(("gdpr Preamble", None)))])
    with pytest.raises(ValueError, match="gdpr Preamble"):
        ledger_rows(artifact, {"case": []})


def test_an_invalid_produced_rating_fails_loudly():
    artifact = _artifact([_case("case", _produced_dump(("gdpr Article 22", "centrak")))])
    with pytest.raises(ValueError, match="centrak"):
        ledger_rows(artifact, {"case": []})


def test_an_artifact_case_without_current_keys_fails_loudly():
    """The ledger scores against the keys the caller supplies; an artifact
    case those keys no longer know is a mismatch to surface, never to skip —
    a silently dropped case would read as a clean audit."""
    artifact = _artifact([_case("renamed-case", _produced_dump(("gdpr Article 22", None)))])
    with pytest.raises(ValueError, match="renamed-case"):
        ledger_rows(artifact, {})


# --- The module entry point: python -m backend.src.eval_ledger ----------------


def _shipped(monkeypatch, *cases: EvalScenario) -> None:
    import src.eval_ledger as eval_ledger

    monkeypatch.setattr(eval_ledger, "LIVE_EVAL_SCENARIOS", list(cases))


def _shipped_case(case_id: str, *expected: ExpectedFinding) -> EvalScenario:
    return EvalScenario(id=case_id, scenario_id="some-scenario", question="What applies?", expected=list(expected))


def _write_artifact(tmp_path, artifact: dict):
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact))
    return path


def test_main_prints_rows_with_weights_and_the_re_scored_means(monkeypatch, tmp_path, capsys):
    """The CLI emits, per Scenario, the re-scored P/R/F1 and both lists with
    their weights: missed entries name the operator's expectation, spurious
    entries the answer's own rating — 'unrated' where the Summarizer shipped
    none."""
    _shipped(
        monkeypatch,
        _shipped_case(
            "case-a",
            _expected("gdpr", "Article 6", Strength.strong),
            _expected("ai-act", "Article 3", Strength.weak),
        ),
    )
    path = _write_artifact(tmp_path, _artifact([
        _case(
            "case-a",
            _produced_dump(("ai-act Article 3", "weak"), ("ai-act Article 99", None)),
            _produced_dump(("dora Article 46", "moderate")),
        ),
    ]))

    exit_code = main(["--artifact", str(path)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Eval ledger" in out
    assert "Model: test-model" in out
    assert "case-a: precision=0.333 recall=0.020 F1=0.037" in out
    assert "gdpr Article 6 (expected strong, weight 50)" in out
    assert "ai-act Article 99 (unrated)" in out
    assert "dora Article 46 (rated moderate, weight 5)" in out
    assert "Mean re-scored precision: 0.333 recall: 0.020 F1: 0.037" in out


def test_main_notes_shipped_cases_the_artifact_does_not_measure(monkeypatch, tmp_path, capsys):
    """An artifact covering fewer cases than the shipped list measures is
    normal (the list grows); the report must say so, or a partial audit
    reads as a full one."""
    _shipped(monkeypatch, _shipped_case("case-a"), _shipped_case("case-b"))
    path = _write_artifact(tmp_path, _artifact([_case("case-a")]))

    exit_code = main(["--artifact", str(path)])

    assert exit_code == 0
    assert "does not measure: case-b" in capsys.readouterr().out


def test_main_runs_without_any_provider_configuration(monkeypatch, tmp_path, capsys):
    """The ledger is offline by construction: no key, no settings, no
    provider client — auditing a paid run never spends another call."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _shipped(monkeypatch, _shipped_case("case-a", _expected("gdpr", "Article 33", Strength.strong)))
    path = _write_artifact(tmp_path, _artifact([_case("case-a")]))

    exit_code = main(["--artifact", str(path)])

    assert exit_code == 0
    assert "case-a" in capsys.readouterr().out


def test_main_refuses_a_missing_artifact_file(tmp_path, capsys):
    exit_code = main(["--artifact", str(tmp_path / "nope.json")])

    assert exit_code == 1
    assert "nope.json" in capsys.readouterr().err


def test_main_refuses_truncated_json(tmp_path, capsys):
    path = tmp_path / "artifact.json"
    path.write_text('{"scenarios": [')

    exit_code = main(["--artifact", str(path)])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "Eval ledger refused" in err


def test_main_refuses_an_artifact_case_without_shipped_keys(tmp_path, capsys):
    """A case id the shipped ground truth does not know refuses the run with
    the case named — the artifact and the record disagree."""
    path = _write_artifact(tmp_path, _artifact([_case("renamed-case")]))

    exit_code = main(["--artifact", str(path)])

    assert exit_code == 1
    assert "renamed-case" in capsys.readouterr().err


def test_shipped_expected_keys_read_the_live_case_list():
    """The CLI's keys are whatever the harness ships now: one entry per
    curated Live case, its expected Findings — never a stored copy."""
    keys = shipped_expected_keys()

    assert set(keys) == {case.id for case in LIVE_EVAL_SCENARIOS}
    for case in LIVE_EVAL_SCENARIOS:
        assert keys[case.id] == case.expected
