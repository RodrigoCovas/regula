# Regula — Regulatory Research & Compliance Assistant

A 7-day AI-native prototype for regulatory research and compliance analysis using agentic workflows, RAG, and structured knowledge extraction.

## Quick Start (stakeholders, local demo)

The deterministic demo runs entirely on your machine: no cloud account and no
API key are required. You only need Docker with Docker Compose.

1. **Clone the repository:**
```bash
git clone https://github.com/RodrigoCovas/regula.git
cd regula
```

2. **Start the stack:**
```bash
docker compose up -d --build
```
This builds both images and starts the backend on port 8000, the frontend on
port 3000, PostgreSQL with pgvector, and Ollama.

3. **Open the demo in your browser:**
Navigate to http://localhost:3000 to use the web interface. The form accepts a
scenario description and question; the Scenario id is derived automatically.
Click "Try the demo scenario" to run the canonical Spanish fintech example.

4. **Or send the canonical demo scenario via API:**
```bash
curl -s http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "scenario": {"id": "spanish-fintech-startup-uses-9e165169", "description": "A Spanish fintech startup that uses machine learning to assess creditworthiness for consumer loans. The platform automatically approves or denies applications based on applicant data including income, employment history, and spending patterns. The company operates only in Spain and plans to expand to other EU markets."},
    "question": "What regulations apply to our AI-based credit scoring platform?"
  }'
```
The response contains `answer`, `trace`, `detailed_trace`, and `known_limitations`
as siblings.
Only `scenario.id == "spanish-fintech-startup-uses-9e165169"` triggers the demo; any other id gets a
helpful not-available response.

5. **Check the service is healthy:**
```bash
curl http://localhost:8000/health
```

Before using Live mode, check its Readiness — the provider key, embedding
model, and ingested Corpus as three booleans:
```bash
curl http://localhost:8000/readiness
# {"api_key_set":true,"embedding_model_present":true,"corpus_ingested":false}
```

6. **Stop when done:**
```bash
docker compose down
```

### Modes

Mode is a per-run choice (ADR-0008): every analysis request carries `mode`
explicitly, and a request that omits it gets the server default
(`REGULA_MODE`, itself defaulting to `demo`). One backend serves both modes:

The web UI exposes the choice as a Demo/Live toggle that defaults to Demo on
every page load and never persists; Live submission stays blocked, with a
checklist of the missing prerequisites and their remediation commands, until
the readiness endpoint reports Live mode ready (issue #48).

- **Demo mode** (`mode: "demo"`) — keyless deterministic demo. Answers only
  `scenario.id == "spanish-fintech-startup-uses-9e165169"` from fixed content.
- **Live mode** (`mode: "live"`) — answers **arbitrary** scenarios through
  the real workflow: Planner → Researcher → Verifier → Proposer → Summarizer
  retrieve Chunks from the ingested pgvector Corpus and produce
  evidence-backed Findings with Strength badges, metadata-derived Citations,
  and Provision relevance. Live mode is executable only when
  the prerequisites hold: `OPENROUTER_API_KEY`, the embedding model, and an
  ingested Corpus. A missing provider key or an un-ingested / unreachable
  store yields a Not-available response naming the exact fix — never a boot
  refusal in the default mode, never a server error. The LLM provider is
  environment configuration (ADR-0009): Solar Pro 4 (`upstage/solar-pro4`)
  via OpenRouter by default, overridable with `LLM_MODEL`, `LLM_BASE_URL`,
  and the API key — see `backend/.env.example`.

### Request observability

Every `/api/analyze` request appends exactly one JSON line to
`logs/queries.jsonl` (the documented log convention; override the location
with `QUERY_LOG_PATH`). Each record carries the cost and behaviour basics —
no platform, no new infrastructure:

| Field | Meaning |
| --- | --- |
| `timestamp` | When the request finished (ISO 8601, UTC) |
| `mode` | `demo` or `live` |
| `scenario_id` | The Scenario's id |
| `workflow` | The served Execution trace's workflow marker; `null` when the request failed |
| `status` | `success`, or `failure` for a request that errored |
| `latency_ms` | Wall-clock request latency in milliseconds |
| `retrieved_chunks` | Chunks retrieved for the request (0 when none were) |
| `tokens` | `{prompt, completion, total}` summed across the request's LLM calls; `null` when no LLM ran |
| `error` | The failure cause, truncated; `null` on success |

Failures are recorded like any other request — never silently absent — so
cost stays observable precisely when things go wrong. A broken log
destination degrades to a warning; serving is never affected.

### Progress endpoint

Live-mode requests can take time as the workflow progresses through Planner,
Researcher, Verifier, Proposer, and Summarizer phases. The frontend polls
`GET /api/progress/{request_id}` to display which phase is currently running.
The backend records the active workflow step in an in-memory registry with
TTL cleanup; unknown request ids return a Not-available-shaped response.
Single-worker uvicorn makes in-memory state safe for this use case.

### Readiness endpoint

`GET /readiness` reports Live-mode Readiness (see Modes above) as three
independent booleans, each probed from real state on every call:

```json
{"api_key_set": true, "embedding_model_present": true, "corpus_ingested": true}
```

A missing prerequisite reads `false` individually — an unreachable Ollama or
vector store counts as not-ready rather than an error, so tooling can poll
the endpoint while the stack is coming up. `/health` stays the liveness
probe: process-up only, never a Readiness verdict.

### Running the tests locally (no Docker)

**Backend** (requires Python 3.13+):
```bash
pip install -r backend/requirements.txt
python -m pytest backend/tests/ -q
```

With type checking (dev):
```bash
pip install -r backend/requirements-dev.txt
mypy
```
Configuration lives in `pyproject.toml` (`[tool.mypy]`).

**Frontend** (requires Node.js 20+):
```bash
cd frontend
npm install
npm test
npm run typecheck
```

All checks run automatically on every push and pull request via GitHub Actions (`.github/workflows/ci.yml`).

### Developer setup (full stack)

The steps below are for development beyond the deterministic demo.

- No API keys are required for the deterministic demo; the backend makes no
  LLM calls. The full stack below is only needed for development beyond the demo.
- Start everything (PostgreSQL + pgvector, Ollama, backend, frontend):
```bash
docker compose up -d --build
```
- Pull the embedding model (required for Live mode):
```bash
docker compose exec ollama ollama pull hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:F16
```
- Frontend runs on http://localhost:3000 (development mode: `cd frontend && npm install && npm run dev`)
- Backend runs on http://localhost:8000

### Live mode setup (ingest the Corpus)

Live mode answers arbitrary Scenarios over the ingested Corpus. Setup is a
single documented command, run in this order — each step depends on the one
before it:

1. **Stack up** (PostgreSQL + pgvector, Ollama, backend, frontend):
```bash
docker compose up -d --build
```

2. **Pull the embedding model** (nomic-embed-text v1.5 via Hugging Face — the
   Ollama registry mirror is not reachable from every network):
```bash
docker compose exec ollama ollama pull hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:F16
```

3. **Run ingestion** (idempotent — re-running updates existing rows and
   inserts no duplicates):
```bash
docker compose exec backend python -m backend.src.ingest
```

Ingestion reads `data/regulations/*.json`, embeds every Chunk locally with the
pulled embedding model (`hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:F16` by
default; override with `--model`), and upserts it into PostgreSQL/pgvector
together with its provision metadata (Article XOR Recital XOR Annex). The
backend never ingests on startup; serving Live mode afterwards additionally
requires `OPENROUTER_API_KEY`.

### Running Live mode

After ingesting the Corpus:

1. **Configure the API key** in `backend/.env.local`:
```bash
echo "OPENROUTER_API_KEY=your-key-here" >> backend/.env.local
```

2. **Start the backend**:
```bash
python -m uvicorn backend.src.main:app --reload --reload-dir backend/src
```

3. **Start the frontend** (in another terminal):
```bash
cd frontend && npm run dev
```

4. **Run a Live analysis** — the request carries `mode` explicitly (ADR-0008):
```bash
curl -s http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"mode": "live", "scenario": {"description": "A Spanish fintech startup that uses machine learning to assess creditworthiness for consumer loans."}, "question": "What regulations apply to our AI-based credit scoring platform?"}'
```
The Live workflow (Planner → Researcher → Verifier → Proposer → Summarizer) answers using the ingested
Corpus and the configured LLM. Live mode requests typically take 60-90 seconds;
the frontend polls the progress endpoint and displays phase transitions as they
occur.

The API key is read from `backend/.env.local` automatically at startup. The LLM
provider is environment configuration (ADR-0009): it defaults to
`upstage/solar-pro4` via OpenRouter, and `LLM_MODEL`, `LLM_BASE_URL`,
`OPENROUTER_API_KEY`, and `EMBEDDING_MODEL` override each part — see
`backend/.env.example`.

### Using another LLM provider

The LLM client speaks the OpenAI-compatible chat-completions convention:
`LLM_BASE_URL` ends at the version root and the client appends
`/chat/completions` itself; auth is a plain `Authorization: Bearer` header
with no provider-specific extras. **OpenRouter is the tested path**; the
providers below are expected to work but untested.

| Provider | `LLM_BASE_URL` | `LLM_MODEL` example |
|---|---|---|
| OpenRouter (default, tested) | `https://openrouter.ai/api/v1` | `upstage/solar-pro4` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| Ollama (local) | `http://localhost:11434/v1` | `llama3.1` |
| vLLM / LM Studio (local) | `http://localhost:8000/v1` / `http://localhost:1234/v1` | the served model's name |

Provider caveats worth knowing before you switch:

- **The key variable's name is historical.** `OPENROUTER_API_KEY` holds
  whichever provider's key you are using; with the matching `LLM_BASE_URL`
  an OpenAI, Groq, or local key works the same way.
- **Local keyless servers still need a key value.** A Live-mode request
  without `OPENROUTER_API_KEY` configured answers Not-available (ADR-0008),
  and the client always sends the Bearer header — so set it to any non-empty
  placeholder (e.g. `ollama`); Ollama, vLLM, and LM Studio ignore its value.
- **Avoid reasoning models for now.** The client sends `max_tokens` (its
  locked single-pass budget); OpenAI's o-series and gpt-5 reasoning models
  reject that parameter in favour of `max_completion_tokens`. Standard chat
  models — GPT-4o, Llama, Solar Pro 4 — accept it. An unsupported model
  fails loudly with the provider's error, never silently.
- **The provider must speak the OpenAI chat-completions shape.** Anthropic's
  native API (`/v1/messages`) is a different protocol — use Claude through
  OpenRouter instead.
- **Embeddings are not covered by the provider switch.** `EMBEDDING_MODEL`
  always runs on local Ollama; a remote key cannot serve embeddings, and the
  pgvector column is fixed at nomic's 768 dimensions, so remote embedding
  APIs are out by construction. Changing the embedding model still requires
  re-ingest.
- **Results are model-sensitive (ADR-0009).** Evaluation numbers are
  comparable only within a model; whichever model produced the published
  metrics is quoted alongside them.

## Architecture

```
Frontend (Next.js + TypeScript + Tailwind) :3000
    ↓
FastAPI Backend (/api/analyze) :8000
    ↓
LangGraph Workflow:
  Planner → Researcher (with Retrieval) → Verifier → Proposer → Summarizer
    ↓
Retrieval Service Layer
    ↓
PostgreSQL + pgvector + Chunks Table
    ↓
Ollama Embeddings (nomic-embed-text)
```

## Project Structure

```
regula/
├── backend/
│   ├── src/
│   │   ├── main.py                 # FastAPI app and API routes
│   │   ├── config.py               # Configuration (server default mode, provider settings)
│   │   ├── models.py               # Pydantic schemas
│   │   ├── db.py                   # Database setup and connection
│   │   ├── retrieval.py            # Retrieval service layer
│   │   ├── embedder.py             # Embedding service (Ollama)
│   │   ├── chunking.py             # Document chunking logic
│   │   ├── corpus.py               # Corpus management
│   │   ├── ingest.py               # Document ingestion CLI
│   │   ├── llm.py                  # LLM client (OpenRouter)
│   │   ├── live_workflow.py        # Live-mode workflow (Planner → Researcher → Verifier → Proposer → Summarizer)
│   │   ├── availability.py         # Availability checks and Not-available responses
│   │   ├── progress.py             # In-memory progress registry for workflow phases
│   │   ├── query_log.py            # Query logging to queries.jsonl
│   │   ├── eval_harness.py         # Demo-mode evaluation harness
│   │   └── live_eval.py            # Live-mode evaluation CLI
│   ├── tests/                      # pytest test suite
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app/                        # Next.js App Router (layout.tsx, page.tsx)
│   ├── components/                 # React components (ScenarioForm, AnswerSurface, etc.)
│   ├── lib/                        # Client logic (analyze-client, progress, scenario-id)
│   ├── tests/                      # Node test runner test suite
│   ├── package.json
│   ├── tsconfig.json
│   └── Dockerfile
├── data/
│   └── regulations/                # Ingested documents (ai-act.json, gdpr.json, dora.json)
├── docs/
│   ├── adr/                        # Architecture Decision Records
│   └── agents/                     # Agent workflow documentation
├── research/                       # Research notes and decision trees
├── logs/
│   └── queries.jsonl               # Query logs (gitignored)
├── docker-compose.yml
├── CONTEXT.md                      # Domain model and glossary
├── AGENTS.md                       # Agent skills and checks
└── README.md
```

## 7-Day Roadmap

- **Day 1:** Setup + Data pipeline (embeddings queryable)
- **Day 2:** Retrieval service + Pydantic schemas
- **Day 3:** Planner + Researcher agents with LangGraph
- **Day 4:** Verifier + Evaluation framework (70% passing)
- **Day 5:** Debug & polish (80-90% passing)
- **Day 6:** Frontend + Observability
- **Day 7:** Testing + Deployment + Demo

## Key Technologies

- **LLM policy:** the deterministic MVP demo makes no LLM call and requires no keys or env vars. The stack below is for development beyond the demo only:
  - **LLM:** upstage/solar-pro4 (via OpenRouter) by default — `LLM_MODEL` / `LLM_BASE_URL` select any OpenAI-compatible provider
- **Embeddings:** Ollama + nomic-embed-text (default; `EMBEDDING_MODEL` overrides, re-ingest required)
- **Orchestration:** LangGraph
- **Database:** PostgreSQL + pgvector
- **Backend:** FastAPI + Pydantic
- **Frontend:** Next.js + TypeScript + Tailwind

## Evaluation

The eval harness (`backend/src/eval_harness.py`) scores the three
never-blended components of ADR-0010
(`docs/adr/0010-evaluation-scores-provisions-not-statements.md`), reported
separately per case and in aggregate:

- **Provision coverage** — a deterministic set F1 over Citation targets —
  provision kind + number, never label strings — with the expected side
  weighted by the operator's per-provision Strength ratings carried on its
  expected Citations (strong = 50, moderate = 5, weak = 1) and every
  off-target produced Citation counted against precision.
- **Summary fidelity** — a strict rubric judge (same configured provider,
  one batched call per case, schema-validated verdict) compares
  provision-aligned expected and produced Provision relevance; a
  contradiction between the two forces the floor score. Ground-truth
  relevance summaries are hand-authored per case, so the component reads
  `n/a` where labels are not yet written.
- **Strength agreement** — max-rule Citation strength compared on both
  sides over the provisions both sides cite: the expected half reads the
  operator's per-provision Strength ratings, the produced half the Answer's
  Finding Strengths. It is reported as its own
  number and read at small weight: it grades only the Strength labels the
  workflow assigned, while coverage and fidelity grade the substance.

Statement similarity gates no score; produced and expected statements ride
only in the per-case audit dump for human review.

The Demo-mode cases act as functional tripwires (see
`docs/adr/0001-eval-cases-are-functional-tripwires.md`): Demo production
derives its Citations from the same locked targets its ground truth
transcribes, so mean F1 is 1.0 by construction and any drop signals a
regression, not poor quality.

### Measuring Live answer quality (operator only)

Live-mode quality is measured by one operator command — it is never part of
CI and needs the Live prerequisites already in place (API key plus ingested
Corpus; see the setup section above):

```bash
python -m backend.src.live_eval --output eval-report.json
```

It runs the hand-authored ground-truth cases through `/api/analyze` in Live
mode, scores each Answer with the three components, and prints per-case
coverage, summary fidelity, and strength agreement plus the aggregate mean
of each. The judge runs on the same configured provider as the workflow; if
it cannot be reached — or a verdict fails its schema — the run aborts with
the fix instead of scoring silence. With `--output` it also writes a JSON
report: per-case component scores plus the produced-versus-expected dump
(statements, Strengths, Citation targets, relevance summaries) for the
human audit. Precision over provisions is pessimistic by construction — a
produced Citation outside the hand-authored expected set is not necessarily
wrong; review the spurious list before quoting numbers. Without a key or an
ingested Corpus it refuses with the fix instead of measuring garbage.

## Deployment

- **Permanently local:** Docker Compose only. Regula will not be deployed to any
  cloud host — including Render — by design;
  see `docs/adr/0002-regula-is-permanently-local-only.md`.

## Next Steps

1. [Day 1: Research & Data](./research/DAY_1_RESEARCH.md)
2. [Decision Tree](./research/DECISION_TREE.md)

## License

MIT

## Author

Built by Rodrigo with GitHub Copilot CLI
