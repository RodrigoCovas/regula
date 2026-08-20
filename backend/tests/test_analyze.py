import sys
from pathlib import Path

# Ensure backend.src is importable when running tests from repo root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_analyze_spanish_fintech_demo():
    payload = {
        "scenario": {"id": "spanish-fintech", "description": "Demo: Spanish fintech lending"},
        "question": "Would an automated loan denial violate data protection requirements?"
    }
    resp = client.post("/api/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
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
    trace_steps = answer.get("detailed_trace", [])
    researcher_steps = [step for step in trace_steps if step.get("step") == "researcher"]
    assert researcher_steps, "Expected a researcher step in detailed_trace"
    assert "tool_calls" in researcher_steps[0]
    assert len(researcher_steps[0]["tool_calls"]) >= 1
