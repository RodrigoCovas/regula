"""
Regula — Regulatory Research & Compliance Assistant
Backend entry point
"""

from contextlib import asynccontextmanager
from typing import Any, Dict, Iterator, List, Optional, TypedDict, Union
import logging
import time

from fastapi import Depends, FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware

from .availability import (
    ENGLISH_ONLY_LIMITATION,
    PROTOTYPE_LIMITATION,
    UNREACHABLE_STORE_ERRORS,
    log_startup_store_warning,
    store_unreachable,
    un_ingested_corpus_response,
    unreachable_llm_response,
    unreachable_store_response,
    vector_store_is_empty,
)
from .config import ConfigurationError, Mode, load_settings
from .corpus import load_documents
from .db import LazyStore, PgVectorStore, connect
from .embedder import OllamaEmbedder
from .llm import Llm, LlmUnreachableError, OpenRouterClient
from .live_workflow import LIVE_WORKFLOW_MARKER, SEEK_COUNSEL_ACTION, run_live_analysis
from .models import AnalyzeRequest, AnalyzeResponse, Answer, ClaimDecision, Finding, Citation, Strength, Trace, PROVISION_NUMBER_FIELDS, ProvisionKind, quote_snippet
from .progress import ProgressSnapshot, progress_registry, progress_sink, unknown_request_response
from .query_log import (
    STATUS_FAILURE,
    STATUS_SUCCESS,
    RequestObservation,
    aggregate_token_usage,
    append_query_record,
    build_query_record,
)
from .retrieval import Retriever, VectorRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fail fast before any request is served: a misconfigured environment aborts
# startup (uvicorn fails at import) instead of surfacing as a server error.
settings = load_settings()

# The one header through which a client tags a request with the progress id
# it will poll: POST /api/analyze with this header registers the run in the
# progress registry, and GET /api/progress/{request_id} reads it back.
REQUEST_ID_HEADER = "x-request-id"


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Re-load settings through the real startup lifecycle.

    Uvicorn normally fails earlier, at module import; this second validation
    is the behavioural seam that lets tests boot the app with an environment
    and observe misconfiguration as a startup failure.
    """
    global settings
    settings = load_settings()
    log_startup_store_warning(settings)
    yield


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

# The small curated corpus (metadata + articles + recitals + annexes), loaded
# once per process through the shared loader (corpus.py) — the Live path
# derives Citation short names from the same documents.
_CORPUS = load_documents()


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


# Where each provision kind's context lives in the corpus JSON structure.
_PROVISION_FINDERS = {
    ProvisionKind.article: _find_article_context,
    ProvisionKind.recital: _find_recital_context,
    ProvisionKind.annex: _find_annex_context,
}


def _resolve_target(doc: dict, target: LookupTarget) -> Optional[Dict[str, Any]]:
    """Resolve a lookup target to citation field name and document context.

    Articles and annexes are found only when they carry text. Recitals are found
    by number even when their text is absent from the corpus (quote stays None).
    """
    try:
        kind = ProvisionKind(target["kind"])
    except ValueError:
        return None
    ctx = _PROVISION_FINDERS[kind](doc, target["number"])
    if not ctx or (kind is not ProvisionKind.recital and not ctx.get("text")):
        return None
    return {"field": PROVISION_NUMBER_FIELDS[kind], "ctx": ctx}


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

def _verify_claims(findings: List[Finding]) -> tuple[List[str], List[ClaimDecision]]:
    """Trace the rejection of the curated anticipated-but-unsupported Claims.

    This is not a deciding pass: every claim in UNSUPPORTED_CLAIM_CANDIDATES
    matches no produced Finding by construction, so all are discarded as
    Unsupported claims with reason "no Evidence in the Corpus supports this
    claim". Returns the discarded statements and the per-claim decisions for
    the detailed trace.
    """
    supported = {f.statement for f in findings if f.citations}
    discarded: List[str] = []
    decisions: List[ClaimDecision] = []
    for claim in UNSUPPORTED_CLAIM_CANDIDATES:
        if claim in supported:
            decisions.append(ClaimDecision(claim=claim, status="kept"))
        else:
            discarded.append(claim)
            decisions.append(ClaimDecision(claim=claim, status="rejected", reason="no Evidence in the Corpus supports this claim"))
    return discarded, decisions

# Referral-voiced Actions (ADR-0004): each names something only a qualified
# professional can settle — a contingency the Corpus cannot determine, or
# verification of a cited provision against the company's actual situation.
# Never a directive compliance task, never a presumption that a Regulation
# applies. The standing seek-counsel hand-off closes the sheet; the
# research-prototype boundary lives in known_limitations, never among Actions.
DEMO_ACTIONS = [
    "Have a qualified professional determine whether the company is a provider or a deployer under the AI Act, since the obligation sets differ (AI Act Article 3(3)-(4)).",
    "Have a qualified professional confirm whether the company is a licensed financial entity under DORA Article 2, which decides whether DORA applies in full.",
    "Have a qualified professional assess whether a GDPR data protection impact assessment (Article 35(3)(a)) and an AI Act Fundamental Rights Impact Assessment (Article 27) are required before first deployment.",
    "Have a qualified professional verify that human oversight accompanies the credit decision process (AI Act Article 26; GDPR Article 22(3)), so decisions are not solely automated.",
    "Have a qualified professional confirm how applicants will be informed of the AI system's role in decisions (AI Act Article 86; GDPR Articles 13(2)(f) and 15(1)(h)).",
    SEEK_COUNSEL_ACTION,
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
                "quote": quote_snippet(text) if text else None,
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


# --- Composition root: the provider protocols of Live mode (spec #9) ---


def get_llm() -> Llm:
    """The structured-completion client: pinned ``upstage/solar-pro4`` via OpenRouter."""
    key = settings.openrouter_api_key.get_secret_value() if settings.openrouter_api_key else ""
    return OpenRouterClient(api_key=key)


def get_retriever() -> Iterator[Retriever]:
    """The Evidence source: local query embeddings over the pgvector store.

    The dependency is resolved on every request — Demo mode included — so
    the store connection stays lazy: it opens only when Live mode actually
    searches and closes when the request scope ends.
    """
    store = LazyStore(settings.database_url)
    try:
        yield VectorRetriever(store=store, embedder=OllamaEmbedder(base_url=settings.ollama_api_url))
    finally:
        store.close()


@app.get("/health")
async def health():
    return {"status": "ok"}


def _log_request(
    request: AnalyzeRequest,
    llm: Llm,
    observation: RequestObservation,
    *,
    status: str,
    workflow: Optional[str],
    latency_ms: float,
    error: Optional[str] = None,
) -> None:
    """Append one JSONL record for a finished request, success or failure.

    Best-effort: append_query_record never raises into the request path.
    """
    append_query_record(
        settings.query_log_path,
        build_query_record(
            mode=settings.regula_mode.value,
            scenario_id=request.scenario.id,
            workflow=workflow,
            status=status,
            latency_ms=latency_ms,
            retrieved_chunks=observation.retrieved_chunks,
            tokens=aggregate_token_usage(llm),
            error=error,
        ),
    )


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(
    request: AnalyzeRequest,
    request_id: Optional[str] = Header(default=None, alias=REQUEST_ID_HEADER),
    llm: Llm = Depends(get_llm),
    retriever: Retriever = Depends(get_retriever),
):
    """Answer a Scenario through Demo mode (canonical, deterministic) or
    Live mode (arbitrary scenarios via retrieval + LLM workflow).

    This endpoint accepts {scenario, question} and returns a structured
    response with answer, trace, detailed_trace, and known_limitations
    as siblings. Live mode first passes the availability gates: an empty
    vector store yields a Not-available response naming the ingest command,
    an unreachable store or LLM provider one naming the outage — never a
    server error for a condition the operator can fix.

    Every request appends one observability record to the query log —
    including failures, which carry an explicit failure status and the
    tokens spent before dying.

    A client may track a long Live-mode run: sending the X-Request-Id
    header registers the run in the progress registry once it reaches the
    Live workflow, and each phase transition (Planner → Researcher →
    Verifier → Proposer) is reported into it. GET /api/progress/{request_id}
    reads the current phase and transition history back. The body contract
    is unchanged — the header is the only progress channel.

    This handler is deliberately synchronous: Live-mode analysis blocks on
    LLM and retrieval calls, and FastAPI serves sync endpoints on a worker
    thread, leaving the event loop free to serve the polling progress
    endpoint mid-run.
    """
    started = time.perf_counter()
    observation = RequestObservation()
    status, workflow, error = STATUS_SUCCESS, None, None
    try:
        response = _dispatch_analyze(
            request,
            llm=llm,
            retriever=retriever,
            observation=observation,
            request_id=request_id,
        )
        workflow = response.trace.workflow
        return response
    except Exception as caught:
        status, error = STATUS_FAILURE, str(caught)
        raise
    finally:
        _log_request(
            request,
            llm,
            observation,
            status=status,
            workflow=workflow,
            latency_ms=(time.perf_counter() - started) * 1000,
            error=error,
        )


def _dispatch_analyze(
    request: AnalyzeRequest,
    *,
    llm: Llm,
    retriever: Retriever,
    observation: RequestObservation,
    request_id: Optional[str] = None,
) -> AnalyzeResponse:
    """Mode dispatch: the served answer for one request, Demo or Live."""
    if settings.regula_mode == Mode.live:
        try:
            empty = vector_store_is_empty(settings.database_url)
        except UNREACHABLE_STORE_ERRORS as error:
            # An outage is not the un-ingested case: it gets its own reply.
            # But only a genuine outage — a reachable store that raised
            # (timeout, canceled statement) still surfaces as a server error.
            if store_unreachable(error):
                return unreachable_store_response(error)
            raise
        if empty:
            return un_ingested_corpus_response()
        # Only a run that reaches the Live workflow reports progress: a
        # request served by a gate above (un-ingested Corpus, unreachable
        # store or LLM) never leaves a progress record behind.
        if request_id:
            progress_registry.register(request_id)
        progress = progress_sink(request_id, registry=progress_registry) if request_id else None
        try:
            response = run_live_analysis(
                request,
                llm=llm,
                retriever=retriever,
                observation=observation,
                progress=progress,
            )
            if request_id:
                progress_registry.complete(request_id, response)
            return response
        except LlmUnreachableError as error:
            # A genuine LLM outage is not a server error: it gets its own
            # Not-available reply, mirroring the unreachable-store case.
            # Any other LlmError — a reachable provider rejecting or
            # malforming an answer — still surfaces as a server error.
            not_available = unreachable_llm_response(error)
            if request_id:
                progress_registry.complete(request_id, not_available)
            return not_available
        except Exception as error:
            if request_id:
                progress_registry.fail(request_id, str(error))
            raise

    scenario = request.scenario

    # Trigger the deterministic demo ONLY on exact scenario.id == "spanish-fintech-startup-uses-9e165169"
    # No keyword-heuristic routing — it silently degrades which is forbidden
    is_spanish_fintech = bool(scenario and scenario.id == "spanish-fintech-startup-uses-9e165169")

    if is_spanish_fintech:
        result = _run_demo_workflow()
        findings = result["findings"]
        citations = result["citations"]
        retrieved_passages = result["retrieved_passages"]
        tool_calls = result["tool_calls"]
        observation.retrieved_chunks = len(retrieved_passages)
        actions = DEMO_ACTIONS

        # Verifier pass: the curated anticipated-but-unsupported claims match
        # no produced Finding by construction; each rejection is recorded in
        # the Execution trace and detailed trace.
        discarded_claims, claim_decisions = _verify_claims(findings)
        trace = Trace(
            workflow=LIVE_WORKFLOW_MARKER,
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
                "claim_decisions": [decision.model_dump() for decision in claim_decisions],
            },
        ]
    else:
        # Helpful "not available" response — not a bare failure, not a guessed demo answer
        findings = []
        citations = []
        actions = [
            "The deterministic demo currently supports only one scenario: use scenario.id 'spanish-fintech-startup-uses-9e165169' with the canonical AI credit scoring question.",
            "This demo covers the EU AI Act (creditworthiness as high-risk), GDPR (automated decision-making), and DORA (financial entity scope).",
        ]
        trace = Trace(
            workflow="noop",
            summary="No demo match; no retrieval performed. Provide scenario.id 'spanish-fintech-startup-uses-9e165169' to invoke the demo.",
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
        known_limitations=[ENGLISH_ONLY_LIMITATION, PROTOTYPE_LIMITATION],
    )


@app.get("/api/progress/{request_id}", response_model=Union[ProgressSnapshot, AnalyzeResponse])
async def progress(request_id: str):
    """Read one run's progress: the current workflow phase and the ordered
    transition history, as the Live workflow reports it into the progress
    registry.

    Once the run completes, the endpoint returns the final AnalyzeResponse
    instead of the progress snapshot — the frontend polls until it receives
    this response shape.

    An unknown request id — never submitted, or expired by the registry's
    TTL — gets the Not-available-shaped reply naming what happened.
    """
    completed = progress_registry.get_completed_response(request_id)
    if completed is not None:
        return completed
    snapshot = progress_registry.get(request_id)
    if snapshot is None:
        return unknown_request_response(request_id)
    return snapshot


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
