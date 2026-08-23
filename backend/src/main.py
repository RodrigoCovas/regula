"""
Regula — Regulatory Research & Compliance Assistant
Backend entry point
"""

from contextlib import asynccontextmanager, closing
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Callable, Dict, List, Optional, TypedDict
import json
import glob
from pathlib import Path
import logging

from .config import ConfigurationError, Mode, load_settings
from .db import PgVectorStore, connect
from .models import AnalyzeRequest, AnalyzeResponse, Answer, Finding, Citation, Strength, Trace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fail fast before any request is served: a misconfigured environment aborts
# startup (uvicorn fails at import) instead of surfacing as a server error.
settings = load_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Re-load settings through the real startup lifecycle.

    Uvicorn normally fails earlier, at module import; this second validation
    is the behavioural seam that lets tests boot the app with an environment
    and observe misconfiguration as a startup failure.
    """
    global settings
    settings = load_settings()
    _log_startup_store_warning()
    yield


# The one documented ingestion command (backend/src/ingest.py docstring, README).
INGEST_COMMAND = "python -m backend.src.ingest"
INGEST_COMMAND_IN_STACK = "docker compose exec backend python -m backend.src.ingest"


def stored_chunk_count() -> int:
    """How many Chunks the pgvector store currently holds.

    Opens a short-lived connection per probe so a stale pooled connection can
    never wedge the guard and an ingestion that happens after boot is visible
    on the next request. Raises if the store cannot be reached; callers decide
    whether that is survivable (boot: yes; Live request: no).
    """
    with closing(connect(settings.database_url)) as connection:
        return PgVectorStore(connection).count_chunks()


def store_probe() -> Callable[[], int]:
    """Composition-root seam for the un-ingested guard.

    Injected as a zero-argument callable rather than a count: only the Live
    branch may call it, so Demo mode never touches the database no matter
    what state the store is in. Tests install deterministic fakes via
    ``app.dependency_overrides[store_probe]``.
    """
    return stored_chunk_count


def _log_startup_store_warning() -> None:
    """Warn once at boot when the vector store cannot serve Live mode.

    An empty or unreachable store is never a startup failure — the process
    boots in any mode so Demo mode keeps serving keyless stakeholders
    untouched. Exactly one warning, naming the ingest command when the store
    is empty.
    """
    try:
        count = stored_chunk_count()
    except Exception as error:  # noqa: BLE001 — any store failure must not abort boot
        logger.warning(
            "Could not check whether the Corpus is ingested (%s); "
            "Live mode requests will fail until the vector store is reachable.",
            error,
        )
        return
    if count == 0:
        logger.warning(
            "Vector store holds no Chunks yet: Live mode answers stay Not-available "
            "until the Corpus is ingested (%s).",
            INGEST_COMMAND,
        )


app = FastAPI(
    title="Regula",
    description="Regulatory research and compliance assistant",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load small curated corpus (metadata + articles + recitals + annexes) at startup
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


def _find_recital_context(doc: dict, number: int) -> Optional[Dict[str, Any]]:
    """Find recital context for a given number. Recital text is optional in the corpus."""
    for rec in doc.get("recitals", []):
        if rec.get("number") == number:
            return {
                "text": rec.get("text"),
                "section": "Recitals",
                "provision": f"Recital {number}",
            }
    return None


def _find_annex_context(doc: dict, number: int) -> Optional[Dict[str, Any]]:
    """Find annex context for a given official annex number (ids look like 'annex-3')."""
    for annex in doc.get("annexes", []):
        if annex.get("id") == f"annex-{number}":
            return {
                "text": annex.get("content"),
                "section": "Annexes",
                "provision": annex.get("title"),
            }
    return None


# Deterministic lookup map for the canonical Spanish fintech demo scenario.
# Each finding lists the provisions that support its claim, with evidence strength.
class LookupTarget(TypedDict):
    source_id: str
    kind: str
    number: int
    provision: str


def _resolve_target(doc: dict, target: LookupTarget) -> Optional[Dict[str, Any]]:
    """Resolve a lookup target to citation field name and document context.

    Articles and annexes are found only when they carry text. Recitals are found
    by number even when their text is absent from the corpus (quote stays None).
    """
    kind = target["kind"]
    number = target["number"]
    if kind == "article":
        ctx = _find_article_context(doc, number)
        if not ctx or not ctx.get("text"):
            return None
        return {"field": "article_number", "ctx": ctx}
    if kind == "recital":
        ctx = _find_recital_context(doc, number)
        if not ctx:
            return None
        return {"field": "recital_number", "ctx": ctx}
    if kind == "annex":
        ctx = _find_annex_context(doc, number)
        if not ctx or not ctx.get("text"):
            return None
        return {"field": "annex_number", "ctx": ctx}
    return None


class FindingDef(TypedDict):
    statement: str
    strength: Strength
    targets: List[LookupTarget]


DEMO_FINDING_DEFS: List[FindingDef] = [
    {
        "statement": "An AI system that evaluates the creditworthiness of natural persons or establishes their credit score is a high-risk AI system under the AI Act, so the full high-risk obligations apply.",
        "strength": Strength.strong,
        "targets": [
            {"source_id": "ai-act", "kind": "article", "number": 6, "provision": "Article 6(2)"},
            {"source_id": "ai-act", "kind": "annex", "number": 3, "provision": "Annex III point 5(b)"},
        ],
    },
    {
        "statement": "As deployer, the company must assign human oversight, keep automatically generated logs for at least six months, and inform applicants that they are subject to a high-risk AI system.",
        "strength": Strength.strong,
        "targets": [
            {"source_id": "ai-act", "kind": "article", "number": 26, "provision": "Article 26(2), (4), (6), (11)"},
        ],
    },
    {
        "statement": "Before first deployment, the deployer of the creditworthiness system must perform a Fundamental Rights Impact Assessment and notify its results to the market surveillance authority.",
        "strength": Strength.strong,
        "targets": [
            {"source_id": "ai-act", "kind": "article", "number": 27, "provision": "Article 27(1)-(4)"},
        ],
    },
    {
        "statement": "Applicants subject to a loan decision based on the system's output have a right to a clear and meaningful explanation of the role of the AI system in the decision.",
        "strength": Strength.strong,
        "targets": [
            {"source_id": "ai-act", "kind": "article", "number": 86, "provision": "Article 86(1)"},
        ],
    },
    {
        "statement": "GDPR restricts decisions based solely on automated processing, including profiling, that produce legal or similarly significant effects; loan scoring is such a decision, and even the contract-necessity exception still requires human intervention and contest rights.",
        "strength": Strength.strong,
        "targets": [
            {"source_id": "gdpr", "kind": "article", "number": 22, "provision": "Article 22(1), (2)(a), (3)"},
            {"source_id": "gdpr", "kind": "recital", "number": 71, "provision": "Recital 71"},
        ],
    },
    {
        "statement": "The credit-scoring processing requires a data protection impact assessment before it starts, because it is a systematic and extensive automated evaluation on which legally effective decisions are based.",
        "strength": Strength.strong,
        "targets": [
            {"source_id": "gdpr", "kind": "article", "number": 35, "provision": "Article 35(1), (3)(a)"},
        ],
    },
    {
        "statement": "DORA applies in full only if the company is itself a licensed financial entity; otherwise its main relevance is through ICT third-party risk where the AI is supplied to financial entities.",
        "strength": Strength.moderate,
        "targets": [
            {"source_id": "dora", "kind": "article", "number": 2, "provision": "Article 2(1)(a), (2)"},
        ],
    },
    {
        "statement": "DORA governs the digital operational resilience of financial entities, not the substance of credit decisions; credit scoring itself is regulated by the AI Act and GDPR, not DORA.",
        "strength": Strength.moderate,
        "targets": [
            {"source_id": "dora", "kind": "article", "number": 1, "provision": "Article 1(1)"},
        ],
    },
    {
        "statement": "The system qualifies as an AI system; the company will be a provider and/or deployer; credit scoring is a form of profiling as cross-referenced into the AI Act.",
        "strength": Strength.weak,
        "targets": [
            {"source_id": "ai-act", "kind": "article", "number": 3, "provision": "Article 3(1), (3), (4), (52)"},
        ],
    },
]

# Claims the planner proposes for the canonical scenario beyond the demo
# Findings. This is an honestly curated list of anticipated-but-unsupported
# Claims: none of them is stated by any produced Finding — by construction,
# every claim here is discarded as Unsupported. The verifier pass exists to
# make those rejections visible in the Execution trace and detailed trace,
# not to decide anything live.
UNSUPPORTED_CLAIM_CANDIDATES = [
    "Credit scoring data is special-category (sensitive) data.",
    "DORA applies to every fintech.",
    "DORA governs AI decisions / automated credit decisions.",
    "The AI Act prohibits automated credit scoring.",
    "AI Act Article 6(1)-style product high-risk obligations are currently applicable.",
    "Recital text for the AI Act and GDPR is available in the corpus.",
    "Spain-specific requirements are covered by the corpus.",
    "Whether the company is provider vs deployer can be determined from the corpus.",
]

ENGLISH_ONLY_LIMITATION = (
    "Known limitation: the corpus is English-only; questions in other languages are answered in English."
)


def _verify_claims(findings: List[Finding]) -> tuple[List[str], List[dict]]:
    """Trace the rejection of the curated anticipated-but-unsupported Claims.

    This is not a deciding pass: every claim in UNSUPPORTED_CLAIM_CANDIDATES
    matches no produced Finding by construction, so all are discarded as
    Unsupported claims with reason "no Evidence in the Corpus supports this
    claim". Returns the discarded statements and the per-claim decisions for
    the detailed trace.
    """
    supported = {f.statement for f in findings if f.citations}
    discarded: List[str] = []
    decisions: List[dict] = []
    for claim in UNSUPPORTED_CLAIM_CANDIDATES:
        if claim in supported:
            decisions.append({"claim": claim, "status": "kept"})
        else:
            discarded.append(claim)
            decisions.append({"claim": claim, "status": "rejected", "reason": "no Evidence in the Corpus supports this claim"})
    return discarded, decisions

DEMO_ACTIONS = [
    "Determine whether the company will be a provider or a deployer under the AI Act; the obligation set differs (AI Act Article 3(3)-(4)).",
    "Confirm whether the company is a licensed financial entity under DORA Article 2, which decides whether DORA applies in full.",
    "Run a GDPR data protection impact assessment (Article 35(3)(a)) and the AI Act Fundamental Rights Impact Assessment (Article 27) before first deployment.",
    "Design human oversight into the credit decision process (AI Act Article 26; GDPR Article 22(3)) so decisions are not solely automated.",
    "Provide applicants clear explanations of the system's role in decisions (AI Act Article 86; GDPR Articles 13(2)(f) and 15(1)(h)).",
    "This is a research prototype, not legal advice; confirm obligations with a qualified professional.",
]


def _lookup_tool_call(target: LookupTarget, status: str) -> dict:
    """One corpus_lookup tool-call record for the detailed trace."""
    return {
        "tool": "corpus_lookup",
        "input": {
            "source_id": target["source_id"],
            "kind": target["kind"],
            "number": target["number"],
        },
        "status": status,
    }


def _run_demo_workflow() -> Dict[str, Any]:
    """Run the deterministic demo workflow for the Spanish fintech scenario.

    Returns findings, citations, retrieved passages, and tool calls.
    """
    findings: List[Finding] = []
    citations: List[Citation] = []
    retrieved_passages: List[dict] = []
    tool_calls: List[dict] = []

    for finding_def in DEMO_FINDING_DEFS:
        finding_citations: List[Citation] = []
        for target in finding_def["targets"]:
            doc = _CORPUS.get(target["source_id"])
            if doc is None:
                tool_calls.append(_lookup_tool_call(target, "not_found"))
                continue
            resolved = _resolve_target(doc, target)
            tool_calls.append(_lookup_tool_call(target, "found" if resolved else "not_found"))
            if not resolved:
                continue

            ctx = resolved["ctx"]
            text = ctx.get("text")
            citation_kwargs = {
                "source_id": target["source_id"],
                "source_short_name": (doc.get("metadata") or {}).get("shortName"),
                "section": ctx.get("section"),
                "provision": target.get("provision") or ctx.get("provision"),
                "quote": (text[:300] + "...") if text else None,
            }
            citation_kwargs[resolved["field"]] = target["number"]
            citation = Citation(**citation_kwargs)
            finding_citations.append(citation)
            citations.append(citation)
            retrieved_passages.append(
                {
                    "source_id": target["source_id"],
                    "kind": target["kind"],
                    "number": target["number"],
                    "section": ctx.get("section"),
                    "provision": target.get("provision") or ctx.get("provision"),
                    "text": text[:1000] if text else None,
                }
            )

        if finding_citations:
            findings.append(
                Finding(
                    statement=finding_def["statement"],
                    strength=finding_def["strength"],
                    citations=finding_citations,
                )
            )

    return {
        "findings": findings,
        "citations": citations,
        "retrieved_passages": retrieved_passages,
        "tool_calls": tool_calls,
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


def _not_available_response(actions: List[str], summary: str) -> AnalyzeResponse:
    """The sibling shape every Not-available response shares.

    Never serves demo content: empty Findings and Citations, the
    "not-available" workflow marker, and the English-only Known limitation.
    """
    return AnalyzeResponse(
        answer=Answer(findings=[], citations=[], actions=actions),
        trace=Trace(workflow="not-available", summary=summary),
        detailed_trace=[],
        known_limitations=[ENGLISH_ONLY_LIMITATION],
    )


def _live_mode_not_available() -> AnalyzeResponse:
    """Not-available response for Live mode while its pipeline is unbuilt.

    Explains how to proceed instead of dead-ending.
    """
    return _not_available_response(
        actions=[
            "Live mode's research pipeline is not available yet, so no analysis can be served in this mode.",
            "Set REGULA_MODE=demo (the default) to analyze the canonical Spanish fintech scenario via scenario.id 'spanish-fintech'.",
        ],
        summary="Live mode selected but its pipeline is not built yet; no retrieval performed and no demo content served.",
    )


def _un_ingested_corpus_response() -> AnalyzeResponse:
    """Not-available response for Live mode while the vector store is empty.

    Names the exact ingest command and how to retry so recovery needs no
    documentation.
    """
    return _not_available_response(
        actions=[
            "The Corpus is not ingested yet, so Live mode has nothing to retrieve from.",
            f"Ingest it once with: {INGEST_COMMAND}",
            f"Inside the Docker stack, run: {INGEST_COMMAND_IN_STACK}",
            "Ingestion takes effect immediately — re-run your request afterwards; no restart is needed.",
            "Set REGULA_MODE=demo (the default) to analyze the canonical Spanish fintech scenario via scenario.id 'spanish-fintech'.",
        ],
        summary="Live mode selected but the vector store is empty; no retrieval performed.",
    )


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: AnalyzeRequest,
    probe: Callable[[], int] = Depends(store_probe),
):
    """Run the deterministic demo workflow for the Spanish fintech scenario.

    This endpoint accepts {scenario, question} and returns a structured
    response with answer, trace, detailed_trace, and known_limitations
    as siblings. Live mode first passes the un-ingested guard: an empty
    vector store yields a Not-available response naming the ingest command.
    """
    if settings.regula_mode == Mode.live:
        if probe() == 0:
            return _un_ingested_corpus_response()
        return _live_mode_not_available()

    scenario = request.scenario

    # Trigger the deterministic demo ONLY on exact scenario.id == "spanish-fintech"
    # No keyword-heuristic routing — it silently degrades which is forbidden
    is_spanish_fintech = bool(scenario and scenario.id == "spanish-fintech")

    if is_spanish_fintech:
        result = _run_demo_workflow()
        findings = result["findings"]
        citations = result["citations"]
        retrieved_passages = result["retrieved_passages"]
        tool_calls = result["tool_calls"]
        actions = DEMO_ACTIONS

        # Verifier pass: the curated anticipated-but-unsupported claims match
        # no produced Finding by construction; each rejection is recorded in
        # the Execution trace and detailed trace.
        discarded_claims, claim_decisions = _verify_claims(findings)
        trace = Trace(
            workflow="planner -> researcher -> verifier",
            summary="Planner identified automated credit decisions, profiling, and ICT risk as research targets; Researcher retrieved provisions from the AI Act, GDPR, and DORA; Verifier recorded each anticipated-but-unsupported claim as rejected — no Evidence in the Corpus supports it.",
            unsupported_claims_discarded=discarded_claims,
        )
        detailed_trace = [
            {"step": "planner", "action": "identify topics: automated credit decisions, high-risk AI, profiling, data protection, ICT risk"},
            {
                "step": "researcher",
                "action": "retrieve high-risk provisions from AI Act, GDPR, and DORA via corpus_lookup",
                "retrieved": retrieved_passages,
                "tool_calls": tool_calls,
            },
            {
                "step": "verifier",
                "action": "record the curated anticipated-but-unsupported claims as rejected (no Evidence in the Corpus supports this claim) and tag each finding with its evidence strength",
                "claim_decisions": claim_decisions,
            },
        ]
    else:
        # Helpful "not available" response — not a bare failure, not a guessed demo answer
        findings = []
        citations = []
        actions = [
            "The deterministic demo currently supports only one scenario: use scenario.id 'spanish-fintech' with a Spanish fintech lending question.",
            "This demo covers the EU AI Act (creditworthiness as high-risk), GDPR (automated decision-making), and DORA (financial entity scope).",
        ]
        trace = Trace(
            workflow="noop",
            summary="No demo match; no retrieval performed. Provide scenario.id 'spanish-fintech' to invoke the demo.",
        )
        detailed_trace = []

    answer = Answer(
        findings=findings,
        actions=actions,
        citations=citations,
    )

    return AnalyzeResponse(
        answer=answer,
        trace=trace,
        detailed_trace=detailed_trace,
        known_limitations=[ENGLISH_ONLY_LIMITATION],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
