"""
Regula — Regulatory Research & Compliance Assistant
Backend entry point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict, List, Optional
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


def _find_article_context(doc: dict, number: int) -> Optional[Dict[str, Any]]:
    """Find article context for a given number in a doc JSON structure."""
    chapters = doc.get("chapters", [])
    for ch in chapters:
        chapter_title = ch.get("title")
        for sec in ch.get("sections", []):
            for art in sec.get("articles", []):
                if art.get("number") == number:
                    paras = art.get("paragraphs", [])
                    texts = [p.get("text", "") for p in paras if p.get("text")]
                    first_para = next((p.get("number") for p in paras if p.get("text")), None)
                    text = " ".join(texts).strip()
                    return {
                        "text": text or None,
                        "section": sec.get("title") or chapter_title,
                        "provision": f"paragraph {first_para}" if first_para is not None else f"article {number}",
                    }
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

    # Keep the canonical demo explicit instead of broad keyword matching.
    is_spanish_fintech = (
        bool(scenario and scenario.id == "spanish-fintech")
        or bool(
            scenario
            and scenario.description
            and "spanish" in scenario.description.lower()
            and "fintech" in scenario.description.lower()
        )
        or ("spanish" in question and any(k in question for k in ["loan", "credit", "fintech", "lending"]))
    )

    findings: List[Finding] = []
    citations: List[Citation] = []
    retrieved_passages: List[dict] = []
    tool_calls: List[dict] = []

    if is_spanish_fintech:
        target_sources = [("gdpr", 22), ("ai-act", 3)]
        for source_id, article_number in target_sources:
            doc = _CORPUS.get(source_id)
            article = _find_article_context(doc, article_number) if doc else None
            found = bool(article and article.get("text"))
            tool_calls.append(
                {
                    "tool": "corpus_lookup",
                    "input": {"source_id": source_id, "article_number": article_number},
                    "status": "found" if found else "not_found",
                }
            )
            if not found:
                continue

            text = article["text"]
            citation = Citation(
                source_id=source_id,
                source_short_name=doc.get("metadata", {}).get("shortName"),
                article_number=article_number,
                section=article.get("section"),
                provision=article.get("provision"),
                quote=(text[:300] + "...") if text else None,
            )
            citations.append(citation)
            retrieved_passages.append(
                {
                    "source_id": source_id,
                    "article": article_number,
                    "section": article.get("section"),
                    "provision": article.get("provision"),
                    "text": text[:1000],
                }
            )

        if citations:
            findings.append(
                Finding(
                    statement="Automated loan or credit decisions (profiling/automated decision-making) may trigger restrictions under GDPR Article 22 and require human oversight and rights for data subjects.",
                    confidence="medium",
                    citations=citations,
                )
            )
            actions = [
                "Recommend human-in-the-loop review for automated credit decisions.",
                "Document lawful basis and provide explanation to affected data subjects per GDPR.",
                "Validate model against fairness and discrimination metrics; keep logs for post-market monitoring.",
            ]
            trace = {
                "workflow": "planner -> researcher -> verifier",
                "summary": "Planner identified automated decision-making and personal data as research targets; Researcher retrieved available evidence; Verifier kept only evidence-backed claims."
            }
        else:
            findings.append(
                Finding(
                    statement="Insufficient evidence found in the fixed corpus to support a reliable answer for this demo request.",
                    confidence="low",
                    citations=[],
                )
            )
            actions = [
                "Expand the fixed corpus with relevant provisions before issuing conclusions.",
                "Re-run analysis once evidence for the requested scenario is available.",
            ]
            trace = {
                "workflow": "planner -> researcher -> verifier",
                "summary": "Planner identified target topics, but Researcher found insufficient evidence; Verifier refused unsupported claims."
            }

        detailed_trace = [
            {"step": "planner", "action": "identify topics: automated decision-making, profiling, data protection"},
            {
                "step": "researcher",
                "action": "retrieve GDPR Article 22 and AI Act Article 3 from fixed corpus",
                "retrieved": retrieved_passages,
                "tool_calls": tool_calls,
            },
            {"step": "verifier", "action": "include only evidence-backed claims and recommendations"},
        ]

    else:
        findings.append(
            Finding(
                statement="Could not identify a demo scenario match. Provide the canonical Spanish fintech scenario to get the demo answer.",
                confidence="low",
                citations=[],
            )
        )
        actions = ["Clarify scenario: use scenario.id 'spanish-fintech' or describe a Spanish fintech lending case."]
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
