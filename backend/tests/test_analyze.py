import sys
from pathlib import Path

# Ensure backend.src is importable when running tests from repo root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def analyze(scenario_id, question, **scenario_extra):
    """POST /api/analyze with the given scenario/question and return the parsed response."""
    payload = {
        "scenario": {"id": scenario_id, **scenario_extra},
        "question": question,
    }
    resp = client.post("/api/analyze", json=payload)
    assert resp.status_code == 200
    return resp.json()


def test_analyze_spanish_fintech_demo():
    data = analyze(
        "spanish-fintech",
        "Would an automated loan denial violate data protection requirements?",
        description="Demo: Spanish fintech lending",
    )
    assert "answer" in data
    assert "trace" in data
    assert "detailed_trace" in data
    answer = data["answer"]
    assert "findings" in answer
    assert len(answer["findings"]) >= 1
    # Expect citations to include GDPR and AI Act when demo matched
    srcs = {c.get("source_id") for c in answer.get("citations", [])}
    assert "gdpr" in srcs
    assert "ai-act" in srcs
    assert "dora" in srcs
    for citation in answer.get("citations", []):
        targets = [citation.get(f) for f in ("article_number", "recital_number", "annex_number")]
        assert sum(t is not None for t in targets) == 1, "citation must target exactly one provision"
        assert citation.get("section")
        assert citation.get("provision")
    # Q3: recital and annex citations are present alongside article citations
    assert any(c.get("recital_number") is not None for c in answer["citations"]), "expected a recital citation"
    assert any(c.get("annex_number") is not None for c in answer["citations"]), "expected an annex citation"
    # Q4: each finding is tagged with an evidence strength
    for finding in answer.get("findings", []):
        assert finding.get("strength") in {"strong", "moderate", "weak"}
    trace_steps = data.get("detailed_trace", [])
    researcher_steps = [step for step in trace_steps if step.get("step") == "researcher"]
    assert researcher_steps, "Expected a researcher step in detailed_trace"
    assert "tool_calls" in researcher_steps[0]
    assert len(researcher_steps[0]["tool_calls"]) >= 1
    # Known-limitation line: corpus is English-only (surfaced as an action, not a Finding)
    actions_text = " ".join(answer.get("actions", [])).lower()
    assert "english-only" in actions_text


def test_response_shape_has_siblings_not_nested():
    """API returns answer, trace, detailed_trace as siblings; trace is NOT nested inside answer."""
    data = analyze("spanish-fintech", "What regulations apply?")

    # Top-level siblings
    assert set(data.keys()) == {"answer", "trace", "detailed_trace"}
    # Answer does NOT contain trace or detailed_trace
    assert "trace" not in data["answer"]
    assert "detailed_trace" not in data["answer"]


def test_weak_finding_present():
    """Demo answer includes the weak framing Finding (AI Act Article 3 definitions)."""
    data = analyze("spanish-fintech", "What regulations apply?")
    findings = data["answer"]["findings"]

    # Find the weak finding about definitions / Article 3
    weak_findings = [f for f in findings if f.get("strength") == "weak"]
    assert weak_findings, "Expected at least one weak Finding"
    # The definitions finding should mention AI system / provider / deployer / profiling
    def_finding = next((f for f in weak_findings if "ai system" in f["statement"].lower() or "provider" in f["statement"].lower() or "profiling" in f["statement"].lower()), None)
    assert def_finding, "Expected weak Finding about Article 3 definitions"
    assert def_finding["strength"] == "weak"
    # It should cite Article 3
    assert any(c.get("article_number") == 3 for c in def_finding.get("citations", [])), "Weak finding should cite Article 3"


def test_unsupported_claims_absent_from_answer_recorded_in_trace():
    """Unsupported claims are discarded from Answer but recorded in the trace."""
    data = analyze("spanish-fintech", "What regulations apply?")

    # Answer has no unsupported claims (no finding about special-category data, DORA applies to all, etc.)
    finding_statements = " ".join(f["statement"].lower() for f in data["answer"]["findings"])
    assert "special-category" not in finding_statements
    assert "dora applies to every" not in finding_statements
    assert "prohibits automated credit scoring" not in finding_statements

    # Trace records the discarded unsupported claims
    trace = data["trace"]
    assert "unsupported_claims_discarded" in trace
    discarded = trace["unsupported_claims_discarded"]
    assert isinstance(discarded, list)
    assert len(discarded) > 0
    # Check some known unsupported claims are listed
    assert any("special-category" in c.lower() for c in discarded)
    assert any("dora applies to every" in c.lower() for c in discarded)
    assert any("prohibits automated credit scoring" in c.lower() for c in discarded)


def test_non_canonical_scenario_gets_helpful_response_not_keyword_routed():
    """Non-canonical scenarios get a helpful response, NOT silently routed to demo via keywords."""
    # This scenario has Spanish + fintech + loan keywords but wrong id
    data = analyze(
        "other-scenario",
        "Does this loan scoring violate GDPR?",
        description="Spanish fintech loan application evaluation",
    )

    # Should NOT get the demo findings (which would have many findings with citations)
    assert len(data["answer"]["findings"]) == 0
    assert len(data["answer"]["citations"]) == 0

    # Should get helpful actions explaining how to invoke the demo
    actions = data["answer"]["actions"]
    assert len(actions) >= 1
    assert any("spanish-fintech" in a for a in actions)
    assert any("demo" in a.lower() for a in actions)

    # Trace should indicate noop
    trace = data["trace"]
    assert trace.get("workflow") == "noop"
    assert "spanish-fintech" in trace.get("summary", "").lower()


def test_exact_scenario_id_required():
    """Only exact scenario.id == 'spanish-fintech' triggers the demo."""
    # Test with similar but different id
    data = analyze("spanish-fintech-demo", "What regulations apply?")
    assert len(data["answer"]["findings"]) == 0

    # Test with exact match
    data = analyze("spanish-fintech", "What regulations apply?")
    assert len(data["answer"]["findings"]) >= 8  # At least the 9 demo findings


def test_corpus_english_only_limitation_surfaced():
    """The helpful response mentions corpus is English-only."""
    data = analyze("other", "¿Qué regulaciones aplican?")
    actions = " ".join(data["answer"]["actions"]).lower()
    assert "english" in actions


def test_exactly_one_target_citation_invariant():
    """Every citation targets exactly one of article/recital/annex."""
    data = analyze("spanish-fintech", "What regulations apply?")
    for citation in data["answer"]["citations"]:
        targets = [
            citation.get("article_number") is not None,
            citation.get("recital_number") is not None,
            citation.get("annex_number") is not None,
        ]
        assert sum(targets) == 1, f"Citation must target exactly one provision: {citation}"
