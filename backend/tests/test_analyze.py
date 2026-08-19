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
