"""
Regula — Regulatory Research & Compliance Assistant
Backend entry point
"""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, TypedDict, Union
import logging
import threading
import time

from fastapi import Depends, FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware

from .availability import (
    ENGLISH_ONLY_LIMITATION,
    PROTOTYPE_LIMITATION,
    UNREACHABLE_STORE_ERRORS,
    log_startup_store_warning,
    missing_api_key_response,
    store_unreachable,
    un_ingested_corpus_response,
    unreachable_llm_response,
    unreachable_store_response,
    vector_store_is_empty,
)
from .config import ConfigurationError, chat_client, load_settings
from .corpus import load_documents
from .db import LazyStore, PgVectorStore, connect
from .embedder import OllamaEmbedder
from .llm import Llm, LlmUnreachableError
from .live_workflow import LIVE_WORKFLOW_MARKER, SEEK_COUNSEL_ACTION, run_live_analysis
from .models import AnalyzeRequest, AnalyzeResponse, Answer, ClaimDecision, Finding, Citation, GroundedSummary, Mode, ProvisionTarget, Readiness, Strength, Trace, PROVISION_NUMBER_FIELDS, ProvisionKind, answer_citations, quote_snippet
from .progress import ProgressSnapshot, progress_registry, progress_sink, unknown_request_response
from .query_log import (
    STATUS_FAILURE,
    STATUS_SUCCESS,
    RequestObservation,
    aggregate_token_usage,
    append_query_record,
    build_query_record,
)
from .readiness import live_readiness
from .retrieval import HybridRetriever, Retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fail fast before any request is served: a misconfigured environment aborts
# startup (uvicorn fails at import) instead of surfacing as a server error.
settings = load_settings()

# The one header through which a client tags a request with the progress id
# it will poll: POST /api/analyze with this header registers the run in the
# progress registry, and GET /api/progress/{request_id} reads it back.
REQUEST_ID_HEADER = "x-request-id"

_background_threads: set[threading.Thread] = set()
_background_threads_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Re-load settings through the real startup lifecycle.

    Uvicorn normally fails earlier, at module import; this second validation
    is the behavioural seam that lets tests boot the app with an environment
    and observe misconfiguration as a startup failure.

    On shutdown, joins every in-flight background analysis thread so the
    backend records its completed or failed terminal state before the
    process exits (spec #25, issue #41).
    """
    global settings
    settings = load_settings()
    log_startup_store_warning(settings)
    yield
    with _background_threads_lock:
        threads = list(_background_threads)
    for thread in threads:
        thread.join(timeout=30)


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


class LookupTarget(TypedDict):
    source_id: str
    kind: str
    number: int
    provision: str
    relevance: str
    # The operator's per-provision Citation-strength rating (ADR-0011), locked
    # alongside the relevance: the demo answer badges the rated centrality
    # exactly as the Summarizer rates it in Live mode — never derived from
    # Finding strength.
    strength: Strength


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


# The one exact-match demo trigger id (ADR-0005): the bank cloud outage
# case's Scenario id (issue #54). A derived id never equals it, so free-text
# Scenarios fall to the Not-available response — never keyword routing.
DEMO_SCENARIO_ID = "bank-cloud-outage"

# The locked demo content (issue #54): the bank cloud outage case, the best-
# performing Live eval Scenario. Each Finding def's statement follows the
# authored bank-case gold; every target's relevance summary and per-provision
# Strength rating are transcribed verbatim from the operator's citations
# record (data/regulations/citations.json, the bank entry) — the pinning test
# (test_live_eval.py) fails CI on any drift. The operator's ratings ride the
# Citations alone (ADR-0011); the per-Finding Strengths are curated content,
# judged from each Finding's own provisions (CONTEXT.md), presented
# strength-first like the Live Answer.


DEMO_FINDING_DEFS: List[FindingDef] = [
    {
        "statement": "The bank must run the outage through its ICT incident-management process, classify it by clients affected, duration, data losses, service criticality and economic impact and, once classified major, submit initial, intermediate and final reports within the harmonised time limits while promptly informing affected clients about the incident and mitigation measures.",
        "strength": Strength.strong,
        "targets": [
            {
                "source_id": "dora",
                "kind": "article",
                "number": 17,
                "provision": "Article 17",
                "relevance": "Article 17 obliges the bank to run the outage through its ICT incident-management process of detection, management, notification, escalation to senior management, and response procedures for timely restoration.",
                "strength": Strength.strong,
            },
            {
                "source_id": "dora",
                "kind": "article",
                "number": 18,
                "provision": "Article 18",
                "relevance": "Article 18(1) requires the bank to classify the outage by clients affected, duration, data losses, service criticality, and economic impact, which determines whether the major-incident reporting duty is triggered.",
                "strength": Strength.strong,
            },
            {
                "source_id": "dora",
                "kind": "article",
                "number": 19,
                "provision": "Article 19",
                "relevance": "Article 19 obliges the bank, once the outage is classified major, to submit initial, intermediate, and final reports to the competent authority and to promptly inform affected clients about the incident and mitigation measures.",
                "strength": Strength.strong,
            },
        ],
    },
    {
        "statement": "The bank must activate business continuity plans that prioritise resuming the critical online-banking function - plans that cover functions outsourced to the cloud provider - and restore from backups and restoration capabilities.",
        "strength": Strength.strong,
        "targets": [
            {
                "source_id": "dora",
                "kind": "article",
                "number": 11,
                "provision": "Article 11",
                "relevance": "Article 11 obliges the bank to activate ICT business continuity and response plans that prioritise resumption of the critical online-banking function, and Article 11(4) requires those plans to cover functions outsourced through ICT third-party providers like the cloud host.",
                "strength": Strength.moderate,
            },
            {
                "source_id": "dora",
                "kind": "article",
                "number": 12,
                "provision": "Article 12",
                "relevance": "Article 12's backup and restoration duties frame the recovery actions the bank invokes to bring online banking back within acceptable downtime.",
                "strength": Strength.weak,
            },
        ],
    },
    {
        "statement": "Client personal data must remain protected and available: the bank owes security measures and timely restoration of access after the technical incident under the GDPR's integrity and confidentiality baseline, with the recovery and notification processing resting on a lawful basis and the measures implemented and demonstrable.",
        "strength": Strength.strong,
        "targets": [
            {
                "source_id": "gdpr",
                "kind": "article",
                "number": 32,
                "provision": "Article 32",
                "relevance": "Article 32(1)(b)-(c) obliges the bank to ensure availability of its processing systems and timely restoration of access to client personal data after the technical incident, running in parallel with DORA's recovery duties.",
                "strength": Strength.moderate,
            },
            {
                "source_id": "gdpr",
                "kind": "article",
                "number": 5,
                "provision": "Article 5",
                "relevance": "Article 5(1)(f)'s integrity-and-confidentiality principle grounds the bank's ongoing duty to protect client personal data processed through the disrupted systems.",
                "strength": Strength.weak,
            },
            {
                "source_id": "gdpr",
                "kind": "article",
                "number": 6,
                "provision": "Article 6",
                "relevance": "Article 6 requires the recovery and notification processing of customer data to rest on a lawful basis, the lawfulness frame alongside the Article 5 principles for the breach response.",
                "strength": Strength.weak,
            },
            {
                "source_id": "gdpr",
                "kind": "article",
                "number": 24,
                "provision": "Article 24",
                "relevance": "Article 24 obliges the bank to implement and demonstrate measures that keep processing compliant, framing the recovery and accountability follow-up the outage response must include.",
                "strength": Strength.weak,
            },
        ],
    },
    {
        "statement": "In hosting and processing the bank's customer data on the disrupted systems, the cloud provider acts as a processor: the bank may engage it only under a contract carrying Article 28(3)'s guarantees, and the provider must assist the bank in meeting its security and breach-response obligations.",
        "strength": Strength.strong,
        "targets": [
            {
                "source_id": "gdpr",
                "kind": "article",
                "number": 28,
                "provision": "Article 28",
                "relevance": "Article 28 obliges the bank to engage the cloud host only as a processor under a contract carrying Article 28(3)'s guarantees, with Article 28(3)(f) requiring the provider's assistance with the breach and security duties, the GDPR counterpart to the DORA contract levers.",
                "strength": Strength.moderate,
            },
        ],
    },
    {
        "statement": "The bank must identify and document the processes dependent on the cloud provider and its interconnections, keeping those inventories updated - the identification step grounding the recovery and third-party analysis.",
        "strength": Strength.strong,
        "targets": [
            {
                "source_id": "dora",
                "kind": "article",
                "number": 8,
                "provision": "Article 8",
                "relevance": "Article 8 requires the bank to identify and document all processes dependent on the cloud provider and their interconnections, the classification grounding the criticality and recovery analysis for the disrupted functions.",
                "strength": Strength.weak,
            },
        ],
    },
    {
        "statement": "The bank's management body must approve, oversee and periodically review the ICT business continuity policy and the response and recovery plans the outage engages and the post-incident revisions produce.",
        "strength": Strength.strong,
        "targets": [
            {
                "source_id": "dora",
                "kind": "article",
                "number": 5,
                "provision": "Article 5",
                "relevance": "Article 5 places the management body over the outage response: it must approve, oversee, and periodically review the ICT business continuity policy and the response and recovery plans the incident engages.",
                "strength": Strength.weak,
            },
        ],
    },
    {
        "statement": "The bank must operate detection mechanisms that promptly surface anomalous activities and ICT-related incidents across its ICT systems.",
        "strength": Strength.strong,
        "targets": [
            {
                "source_id": "dora",
                "kind": "article",
                "number": 10,
                "provision": "Article 10",
                "relevance": "Article 10 obliges the bank to operate detection mechanisms that promptly surface anomalous activities and ICT-related incidents, the capability the outage's arrival tested.",
                "strength": Strength.weak,
            },
        ],
    },
    {
        "statement": "The recovery processing must appear in the bank's records of processing activities, covering purposes, categories of data subjects and personal data, recipients including the cloud provider, and the security measures.",
        "strength": Strength.strong,
        "targets": [
            {
                "source_id": "gdpr",
                "kind": "article",
                "number": 30,
                "provision": "Article 30",
                "relevance": "Article 30 requires the recovery processing to appear in the bank's records of activities, covering purposes, categories of data subjects and personal data, recipients including the cloud provider, and the security measures.",
                "strength": Strength.weak,
            },
        ],
    },
    {
        "statement": "As a credit institution the bank falls within DORA, and the cloud failure is an ICT-related incident affecting a critical function supplied by an ICT third-party service provider, with the response scaled to the bank's size and risk profile.",
        "strength": Strength.moderate,
        "targets": [
            {
                "source_id": "dora",
                "kind": "article",
                "number": 2,
                "provision": "Article 2",
                "relevance": "Article 2(1)(a) brings the bank within DORA as a credit institution, engaging the ICT incident-management and third-party regimes that govern the outage response.",
                "strength": Strength.moderate,
            },
            {
                "source_id": "dora",
                "kind": "article",
                "number": 3,
                "provision": "Article 3",
                "relevance": "Article 3's definitions of ICT-related incident, major incident, ICT third-party service provider, and critical or important function establish that the cloud failure is an ICT incident affecting a critical function supplied by an external provider.",
                "strength": Strength.moderate,
            },
            {
                "source_id": "dora",
                "kind": "article",
                "number": 4,
                "provision": "Article 4",
                "relevance": "Article 4 scales the bank's incident-response and third-party obligations to its size and risk profile, the proportionality lens applied across the analysis.",
                "strength": Strength.weak,
            },
        ],
    },
    {
        "statement": "The major incident triggers a review of the ICT risk-management framework and a post-incident review of the outage's causes and the response's effectiveness, with a communication strategy framing what clients and stakeholders are told.",
        "strength": Strength.moderate,
        "targets": [
            {
                "source_id": "dora",
                "kind": "article",
                "number": 6,
                "provision": "Article 6",
                "relevance": "Article 6(5) makes a major incident a mandatory trigger for reviewing the bank's ICT risk-management framework, alongside the recovery itself.",
                "strength": Strength.weak,
            },
            {
                "source_id": "dora",
                "kind": "article",
                "number": 13,
                "provision": "Article 13",
                "relevance": "Article 13(2) requires a post-incident review of the outage's causes and of the effectiveness of the bank's response once the incident has disrupted core activities.",
                "strength": Strength.weak,
            },
            {
                "source_id": "dora",
                "kind": "article",
                "number": 14,
                "provision": "Article 14",
                "relevance": "Article 14's communication-strategy duty frames how the bank informs clients and stakeholders about the outage, complementing the specific client-notification duty in Article 19(3).",
                "strength": Strength.weak,
            },
        ],
    },
    {
        "statement": "The bank stays fully responsible for DORA compliance despite the provider's failure and any supervisory feedback on its reports; its dependence on a single cloud provider requires concentration-risk analysis and substitutability considerations, and the contract must secure incident assistance, notification of material developments and tested contingency plans, with audit, termination and exit-strategy levers.",
        "strength": Strength.moderate,
        "targets": [
            {
                "source_id": "dora",
                "kind": "article",
                "number": 28,
                "provision": "Article 28",
                "relevance": "Article 28 keeps the bank fully responsible for DORA compliance despite the cloud provider's failure and supplies the audit, termination, and exit-strategy levers for the provider relationship going forward.",
                "strength": Strength.moderate,
            },
            {
                "source_id": "dora",
                "kind": "article",
                "number": 29,
                "provision": "Article 29",
                "relevance": "Article 29's concentration-risk duties apply because the bank depends on a single cloud provider for critical online banking, requiring substitutability analysis and consideration of alternative solutions.",
                "strength": Strength.weak,
            },
            {
                "source_id": "dora",
                "kind": "article",
                "number": 30,
                "provision": "Article 30",
                "relevance": "Article 30 requires the cloud contract to secure incident assistance, notification of material developments, and tested contingency plans from the provider, the provisions the bank invokes during and after the outage.",
                "strength": Strength.moderate,
            },
            {
                "source_id": "dora",
                "kind": "article",
                "number": 22,
                "provision": "Article 22",
                "relevance": "Article 22 provides that the bank remains fully responsible for the incident's handling and its consequences notwithstanding any supervisory feedback or guidance the competent authority gives on its notification and reports.",
                "strength": Strength.weak,
            },
        ],
    },
    {
        "statement": "If the outage investigation reveals a personal data breach, the 72-hour supervisory-authority notification and, where customers face high risk, the communication duty come into play alongside the DORA reporting, with the authority's corrective powers - from ordering communication to fines - behind that track.",
        "strength": Strength.moderate,
        "targets": [
            {
                "source_id": "gdpr",
                "kind": "article",
                "number": 4,
                "provision": "Article 4",
                "relevance": "Article 4(12) frames the assessment of whether the outage also involved a personal data breach, which turns on whether client data were destroyed, lost, altered, disclosed, or accessed.",
                "strength": Strength.weak,
            },
            {
                "source_id": "gdpr",
                "kind": "article",
                "number": 33,
                "provision": "Article 33",
                "relevance": "Article 33's 72-hour notification duty becomes live where the outage investigation reveals a personal data breach, an obligation the bank must keep in view alongside its DORA reporting.",
                "strength": Strength.weak,
            },
            {
                "source_id": "gdpr",
                "kind": "article",
                "number": 34,
                "provision": "Article 34",
                "relevance": "Article 34's communication duty attaches where the incident creates high risk to customers, complementing the DORA client notification for affected online-banking users.",
                "strength": Strength.weak,
            },
            {
                "source_id": "gdpr",
                "kind": "article",
                "number": 58,
                "provision": "Article 58",
                "relevance": "Article 58's corrective powers - ordering the breach's communication to data subjects, limiting or banning processing, and imposing administrative fines - hang over the conditional GDPR track the outage could open.",
                "strength": Strength.weak,
            },
        ],
    },
    {
        "statement": "As a credit institution the bank cannot use the simplified ICT risk-management framework and must comply with the general framework of Articles 5 to 15.",
        "strength": Strength.moderate,
        "targets": [
            {
                "source_id": "dora",
                "kind": "article",
                "number": 16,
                "provision": "Article 16",
                "relevance": "Article 16's simplified ICT risk-management framework is not open to a credit institution, so the bank must comply with the general framework of Articles 5 to 15, calibrating which obligations bite.",
                "strength": Strength.weak,
            },
        ],
    },
]


# Claims the planner proposes for the canonical scenario beyond the demo
# Findings. This is an honestly curated list of anticipated-but-unsupported
# Claims for the bank cloud outage case: none of them is stated by any
# produced Finding — by construction, every claim here is discarded as
# Unsupported. The verifier pass exists to make those rejections visible in
# the Execution trace and detailed trace, not to decide anything live.
UNSUPPORTED_CLAIM_CANDIDATES = [
    "The outage is automatically a personal data breach under the GDPR.",
    "DORA obliges the bank to compensate customers for losses caused by the outage.",
    "The DORA incident reports replace the GDPR's 72-hour breach notification.",
    "The bank may wait until the root cause is fully known before reporting the incident.",
    "The cloud provider, not the bank, is responsible for DORA compliance during the outage.",
    "The simplified ICT risk-management framework is available to the bank for this incident.",
    "Spain-specific supervisory requirements for incident reporting are covered by the corpus.",
    "The bank can exit the cloud contract without concentration-risk analysis or an exit strategy.",
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
    "Have a qualified professional determine whether the outage is a major ICT-related incident under DORA Article 18, since the initial, intermediate, and final reporting duties (Article 19) turn on that classification.",
    "Have a qualified professional assess whether the outage investigation reveals a personal data breach, which would start the GDPR's 72-hour notification (Article 33) and, where customers face high risk, the communication duty (Article 34) alongside the DORA reporting.",
    "Have a qualified professional verify that affected clients are promptly informed about the incident and the mitigation measures (DORA Article 19(3); GDPR Article 34).",
    "Have a qualified professional review the concentration-risk analysis for the bank's dependence on a single cloud provider, including substitutability and alternative solutions (DORA Article 29).",
    "Have a qualified professional confirm that the cloud contract secures incident assistance, notification of material developments, and tested contingency plans, and that the audit, termination, and exit-strategy levers remain usable (DORA Articles 28 and 30).",
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
    """Run the deterministic demo workflow for the bank cloud outage scenario.

    Returns findings, the Answer's citations (one per cited provision, with
    the locked Provision relevance and the locked Citation-strength rating —
    issue #47, ADR-0011), retrieved passages, and tool calls.
    """
    findings: List[Finding] = []
    # One carrier per cited provision: the locked relevance and rating ride
    # together, exactly as the Live gate's GroundedSummary does.
    summaries_by_target: Dict[ProvisionTarget, GroundedSummary] = {}
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
            summaries_by_target[citation.provision_target] = GroundedSummary(
                target=citation.provision_target,
                relevance=target["relevance"],
                strength=target["strength"],
            )
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
        "citations": answer_citations(findings, summaries_by_target),
        "retrieved_passages": retrieved_passages,
        "tool_calls": tool_calls,
    }


# --- Composition root: the provider protocols of Live mode (spec #9) ---


def get_llm() -> Llm:
    """The structured-completion client, pointed at the configured provider.

    ADR-0009: model, base URL, and key come from the environment with the
    Solar Pro 4 / OpenRouter defaults; deployments differ only by
    configuration.
    """
    return chat_client(settings)


def get_retriever() -> Iterator[Retriever]:
    """The Evidence source: hybrid lexical + vector retrieval over the
    pgvector store, fused by RRF inside the retriever (ADR-0012).

    The dependency is resolved on every request — Demo mode included — so
    the store connection stays lazy: it opens only when Live mode actually
    searches and closes when the request scope ends.
    """
    store = LazyStore(settings.database_url)
    try:
        yield HybridRetriever(
            store=store,
            embedder=OllamaEmbedder(
                base_url=settings.ollama_api_url, model=settings.embedding_model
            ),
        )
    finally:
        store.close()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/readiness", response_model=Readiness)
def readiness() -> Readiness:
    """Report Live-mode Readiness as three booleans (issue #45): the provider
    key configured, the embedding model available on Ollama, and a non-empty
    ingested Corpus — each probed from real state so tooling and the frontend
    can check the prerequisites without probing internals.

    A liveness probe this is not: /health keeps answering process-up only,
    while Readiness can be false on a perfectly healthy process that lacks a
    prerequisite. The endpoint always answers 200 — the booleans are the
    verdict — and an unreachable Ollama or store reads as not-ready, never a
    server error: an outage is a Readiness gap, not a crash.
    """
    return live_readiness(settings)


@dataclass
class RunContext:
    """Everything one analysis run carries: the request, the mode this run
    executes in (ADR-0008), and the request-scoped providers and observation
    the dispatch and the query log share."""

    request: AnalyzeRequest
    mode: Mode
    llm: Llm
    retriever: Retriever
    observation: RequestObservation


def _log_request(
    run: RunContext,
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
            mode=run.mode.value,
            scenario_id=run.request.scenario.id,
            workflow=workflow,
            status=status,
            latency_ms=latency_ms,
            retrieved_chunks=run.observation.retrieved_chunks,
            tokens=aggregate_token_usage(run.llm),
            error=error,
        ),
    )


def _resolve_mode(request: AnalyzeRequest) -> Mode:
    """The mode this run executes in (ADR-0008): the request's explicit
    choice, or the server default (REGULA_MODE, itself defaulting to Demo)
    when the request omits it."""
    return request.mode or settings.regula_mode


@app.post("/api/analyze", response_model=Union[AnalyzeResponse, ProgressSnapshot])
def analyze(
    request: AnalyzeRequest,
    request_id: Optional[str] = Header(default=None, alias=REQUEST_ID_HEADER),
    llm: Llm = Depends(get_llm),
    retriever: Retriever = Depends(get_retriever),
):
    """Answer a Scenario through Demo mode (canonical, deterministic) or
    Live mode (arbitrary scenarios via retrieval + LLM workflow).

    Mode is a per-run choice (ADR-0008), resolved once per run by
    ``_resolve_mode``.

    This endpoint accepts {scenario, question, mode} and returns a structured
    response with answer, trace, detailed_trace, and known_limitations
    as siblings. Live mode first passes the availability gates: a missing
    provider key or an empty vector store yields a Not-available response
    naming the fix, an unreachable store or LLM provider one naming the
    outage — never a server error for a condition the operator can fix.

    Every request appends one observability record to the query log —
    including failures, which carry an explicit failure status and the
    tokens spent before dying.

    A client may track a run: sending the X-Request-Id header registers the
    request in the progress registry — a Live run reports each phase
    transition (Planner → Researcher → Verifier → Proposer) into it as it
    works, and a Demo run records its (immediate) outcome — and
    GET /api/progress/{request_id} reads the result back.

    When the X-Request-Id header is present in Live mode, the analysis runs
    in a background thread and the POST returns promptly with a
    ProgressSnapshot — the UI-facing submission path does not depend on the
    original POST connection remaining open. A dropped client or proxy
    connection does not lose the run; progress polling still reaches the
    eventual Answer or a terminal failure. Without the header, the endpoint
    retains its synchronous contract for non-UI callers.
    """
    run = RunContext(
        request=request,
        mode=_resolve_mode(request),
        llm=llm,
        retriever=retriever,
        observation=RequestObservation(),
    )

    if run.mode == Mode.live and request_id:
        return _handle_live_ui_submission(run, request_id)

    started = time.perf_counter()
    status, workflow, error = STATUS_SUCCESS, None, None
    try:
        if request_id:
            # A UI-submitted Demo run polls for its answer like a Live one:
            # the decoupled client (#41) treats the progress endpoint as the
            # source of truth, so the synchronous dispatch registers its
            # outcome — the served response, or the failure — under the
            # submitted id. Polling then reaches the real answer, never the
            # unknown-request reply.
            progress_registry.register(request_id)
        response = _dispatch_analyze(run)
        workflow = response.trace.workflow
        if request_id:
            progress_registry.complete(request_id, response)
        return response
    except Exception as caught:
        status, error = STATUS_FAILURE, str(caught)
        if request_id:
            progress_registry.fail(request_id, str(caught))
        raise
    finally:
        _log_request(
            run,
            status=status,
            workflow=workflow,
            latency_ms=(time.perf_counter() - started) * 1000,
            error=error,
        )


def _check_live_availability() -> Optional[AnalyzeResponse]:
    """Run the Live-mode Readiness gates (missing provider key, un-ingested
    corpus, unreachable store). Returns None when the gates pass, or a
    Not-available AnalyzeResponse when they fail.

    Shared between the synchronous non-UI path and the background UI path
    so the gate logic lives in one place.
    """
    if not settings.openrouter_api_key:
        return missing_api_key_response()
    try:
        empty = vector_store_is_empty(settings.database_url)
    except UNREACHABLE_STORE_ERRORS as error:
        if store_unreachable(error):
            return unreachable_store_response(error)
        raise
    if empty:
        return un_ingested_corpus_response()
    return None


def _handle_live_ui_submission(run: RunContext, request_id: str) -> ProgressSnapshot:
    """UI-facing Live submission: runs the analysis in a background thread
    so the POST returns promptly with a ProgressSnapshot. The analysis
    continues independently of the original connection; progress polling
    reaches the eventual Answer or terminal failure.

    Availability gates and the full workflow run in the background thread:
    gate failures are recorded in the progress registry so polling
    retrieves them as a Not-available AnalyzeResponse.

    The run's resolved ``llm`` and ``retriever`` from the request context
    are used by the background thread — this keeps dependency overrides
    (used by tests) working and avoids creating duplicate connections for
    fakes.

    The thread is non-daemon and tracked in ``_background_threads``: the
    lifespan joins every in-flight thread on shutdown so the backend
    records its terminal state before the process exits.
    """
    started = time.perf_counter()
    progress_registry.register(request_id)

    def run_in_background() -> None:
        with _background_threads_lock:
            _background_threads.add(threading.current_thread())
        bg_status, bg_workflow, bg_error = STATUS_SUCCESS, None, None
        try:
            gate_response = _check_live_availability()
            if gate_response is not None:
                progress_registry.complete(request_id, gate_response)
                bg_workflow = gate_response.trace.workflow
                return
            progress = progress_sink(request_id, registry=progress_registry)
            response = run_live_analysis(
                run.request,
                llm=run.llm,
                retriever=run.retriever,
                observation=run.observation,
                progress=progress,
            )
            progress_registry.complete(request_id, response)
            bg_workflow = response.trace.workflow
        except LlmUnreachableError as error:
            not_available = unreachable_llm_response(error)
            progress_registry.complete(request_id, not_available)
            bg_workflow = not_available.trace.workflow
        except Exception as error:
            bg_status = STATUS_FAILURE
            bg_error = str(error)
            progress_registry.fail(request_id, str(error))
        finally:
            _log_request(
                run,
                status=bg_status,
                workflow=bg_workflow,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=bg_error,
            )
            with _background_threads_lock:
                _background_threads.discard(threading.current_thread())

    thread = threading.Thread(target=run_in_background)
    thread.start()

    snapshot = progress_registry.get(request_id)
    return snapshot if snapshot is not None else ProgressSnapshot(request_id=request_id)


def _dispatch_analyze(run: RunContext) -> AnalyzeResponse:
    """Mode dispatch: the served answer for one run, Demo or Live.

    Progress tracking for UI-facing Live submissions (with request_id) is
    handled by ``_handle_live_ui_submission``; this path serves non-UI
    callers (synchronous, no progress) and Demo mode.
    """
    if run.mode == Mode.live:
        gate_response = _check_live_availability()
        if gate_response is not None:
            return gate_response
        try:
            response = run_live_analysis(
                run.request,
                llm=run.llm,
                retriever=run.retriever,
                observation=run.observation,
            )
            return response
        except LlmUnreachableError as error:
            # A genuine LLM outage is not a server error: it gets its own
            # Not-available reply, mirroring the unreachable-store case.
            # Any other LlmError — a reachable provider rejecting or
            # malforming an answer — still surfaces as a server error.
            return unreachable_llm_response(error)

    scenario = run.request.scenario

    # Trigger the deterministic demo ONLY on exact scenario.id == DEMO_SCENARIO_ID
    # No keyword-heuristic routing — it silently degrades which is forbidden
    is_canonical = bool(scenario and scenario.id == DEMO_SCENARIO_ID)

    if is_canonical:
        result = _run_demo_workflow()
        findings = result["findings"]
        citations = result["citations"]
        retrieved_passages = result["retrieved_passages"]
        tool_calls = result["tool_calls"]
        run.observation.retrieved_chunks = len(retrieved_passages)
        actions = DEMO_ACTIONS

        # Verifier pass: the curated anticipated-but-unsupported claims match
        # no produced Finding by construction; each rejection is recorded in
        # the Execution trace and detailed trace.
        discarded_claims, claim_decisions = _verify_claims(findings)
        trace = Trace(
            workflow=LIVE_WORKFLOW_MARKER,
            summary="Planner identified ICT incident management and reporting, business continuity, third-party and concentration risk, and data protection as research targets; Researcher retrieved provisions from DORA and the GDPR; Verifier recorded each anticipated-but-unsupported claim as rejected — no Evidence in the Corpus supports it.",
            unsupported_claims_discarded=discarded_claims,
        )
        detailed_trace = [
            {"step": "planner", "action": "identify topics: ICT incident management and major-incident reporting, business continuity and recovery, ICT third-party and concentration risk, GDPR breach readiness"},
            {
                "step": "researcher",
                "action": "retrieve incident-management, continuity, third-party, and data-protection provisions from DORA and the GDPR via corpus_lookup",
                "retrieved": retrieved_passages,
                "tool_calls": tool_calls,
            },
            {
                "step": "verifier",
                "action": "record the curated anticipated-but-unsupported claims as rejected (no Evidence in the Corpus supports this claim) and tag each finding with its evidence strength",
                "claim_decisions": [decision.model_dump() for decision in claim_decisions],
            },
            {
                "step": "summarizer",
                "action": "serve the locked Provision relevance and rated Citation strength for each cited provision from the demo content",
                "summary_decisions": [],
            },
        ]
    else:
        # Helpful "not available" response — not a bare failure, not a guessed demo answer
        findings = []
        citations = []
        actions = [
            f"The deterministic demo currently supports one scenario: use scenario.id '{DEMO_SCENARIO_ID}' with the bank cloud outage question.",
            "In the web UI, use the 'Try the demo scenario' button to fill the form with the canonical scenario, then Analyze.",
            "This demo covers DORA (ICT incident management, third-party risk) and the GDPR (security of processing, breach notification) for a cloud outage at a bank.",
        ]
        trace = Trace(
            workflow="noop",
            summary=f"No demo match; no retrieval performed. Provide scenario.id '{DEMO_SCENARIO_ID}' to invoke the demo.",
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
