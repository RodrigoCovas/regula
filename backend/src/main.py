"""
Regula — Regulatory Research & Compliance Assistant
Backend entry point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import json
import glob
from pathlib import Path
import logging

from .models import AnalyzeRequest, AnalyzeResponse, Answer, Finding, Citation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Regula",
    description="Regulatory research and compliance assistant",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load small curated corpus (metadata + articles) at startup
_CORPUS = {}
# Resolve repo root relative to backend/ package: go up two parents to repo root
_CORPUS_PATH = Path(__file__).resolve().parents[2] / "data" / "regulations"
if not _CORPUS_PATH.exists():
    logger.warning("Corpus path %s does not exist; retrieval will be limited", _CORPUS_PATH)
for path in glob.glob(str(_CORPUS_PATH / "*.json")):
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
            doc_id = doc.get("metadata", {}).get("id")
            if doc_id:
                _CORPUS[doc_id] = doc
    except (OSError, json.JSONDecodeError) as e:
        # Log parse/read failures so maintainers can fix data issues
        logger.warning("Skipping corpus file %s: %s", path, e)
        continue


def _find_article_text(doc: dict, number: int) -> Optional[str]:
    """Find first article with the given number in a doc JSON structure."""
    chapters = doc.get("chapters", [])
    for ch in chapters:
        for sec in ch.get("sections", []):
            for art in sec.get("articles", []):
                if art.get("number") == number:
                    # join paragraph texts for a short quote
                    paras = art.get("paragraphs", [])
                    texts = [p.get("text", "") for p in paras if p.get("text")]
                    return " ".join(texts)
    # fallback: search nested lists
    return None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(query: AnalyzeRequest):
    """Simple deterministic demo workflow for the Spanish fintech scenario.

    This endpoint accepts {scenario, question} and returns a single structured
    Answer with findings, citations, actions, and both compact and detailed traces.
    """
    scenario = query.scenario
    question = (query.question or "").lower()

    # Heuristic: treat requests mentioning loans/credit/fintech as the demo
    is_spanish_fintech = False
    if scenario and (scenario.id == "spanish-fintech" or (scenario.description and "fintech" in scenario.description.lower())):
        is_spanish_fintech = True
    if any(k in question for k in ["loan", "credit", "fintech", "lending"]):
        is_spanish_fintech = True

    findings: List[Finding] = []
    citations: List[Citation] = []
    retrieved_passages: List[dict] = []

    if is_spanish_fintech:
        # GDPR Article 22 (automated decision-making) — cite if present
        gdpr = _CORPUS.get("gdpr")
        gdpr_text = None
        if gdpr:
            gdpr_text = _find_article_text(gdpr, 22)
            if gdpr_text:
                retrieved_passages.append({"source_id": "gdpr", "article": 22, "text": gdpr_text[:1000]})
        gdpr_cit = Citation(
            source_id="gdpr",
            source_short_name=gdpr.get("metadata", {}).get("shortName") if gdpr else None,
            article_number=22,
            quote=(gdpr_text[:300] + "...") if gdpr_text else None,
        )
        citations.append(gdpr_cit)

        # AI Act Article 3 (definitions) as contextual citation
        ai = _CORPUS.get("ai-act")
        ai_text = None
        if ai:
            ai_text = _find_article_text(ai, 3)
            if ai_text:
                retrieved_passages.append({"source_id": "ai-act", "article": 3, "text": ai_text[:1000]})
        ai_cit = Citation(
            source_id="ai-act",
            source_short_name=ai.get("metadata", {}).get("shortName") if ai else None,
            article_number=3,
            quote=(ai_text[:300] + "...") if ai_text else None,
        )
        citations.append(ai_cit)

        findings.append(
            Finding(
                statement="Automated loan or credit decisions (profiling/automated decision-making) may trigger restrictions under GDPR Article 22 and require human oversight and rights for data subjects.",
                confidence="medium",
                citations=[gdpr_cit, ai_cit],
            )
        )

        actions = [
            "Recommend human-in-the-loop review for automated credit decisions.",
            "Document lawful basis and provide explanation to affected data subjects per GDPR.",
            "Validate model against fairness and discrimination metrics; keep logs for post-market monitoring.",
        ]

        trace = {
            "workflow": "planner -> researcher -> verifier",
            "summary": "Planner identified automated decision-making and personal data as research targets; Researcher retrieved GDPR Article 22 and AI Act definitions; Verifier ensured claims are supported and downgraded unsupported leaps."
        }

        detailed_trace = [
            {"step": "planner", "action": "identify topics: automated decision-making, profiling, data protection"},
            {"step": "researcher", "action": "retrieve GDPR Article 22 and AI Act Article 3", "retrieved": retrieved_passages},
            {"step": "verifier", "action": "create concise finding and recommended actions, avoid unsupported claims"},
        ]

    else:
        findings.append(
            Finding(
                statement="Could not identify a demo scenario match. Provide a Spanish fintech scenario or ask about loans/credit to get the demo answer.",
                confidence="low",
                citations=[],
            )
        )
        actions = ["Clarify scenario: use scenario.id 'spanish-fintech' or mention loans/credit in the question."]
        trace = {"workflow": "noop", "summary": "No demo match; no retrieval performed."}
        detailed_trace = []

    answer = Answer(
        findings=findings,
        actions=actions,
        citations=citations,
        trace=trace,
        detailed_trace=detailed_trace,
    )

    return AnalyzeResponse(answer=answer)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
