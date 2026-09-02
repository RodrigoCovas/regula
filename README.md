# Regula — Regulatory Research & Compliance Assistant

Regula answers "what regulations apply to us, and what must we do about it?" for a described business situation. You describe your company, product, and jurisdiction as a **Scenario**, ask a regulatory question, and get an evidence-backed **Answer**: Findings with Strength badges, per-provision Citations with their answer-wide relevance, and Actions naming what only a qualified legal professional can settle. Regula never dispenses legal advice and never substitutes for professional judgment.

Under the hood it is an agentic retrieval pipeline (LangGraph) over a small, curated corpus of EU regulations — the AI Act, GDPR, and DORA — running permanently on your own machine: four containers started by one Docker Compose command. Two modes serve every request: a keyless deterministic **Demo** and a **Live** mode that runs the real workflow over the ingested Corpus with your own LLM provider key.

## Architecture

```
Browser (http://localhost:3000)
    │  /api/* and /readiness are proxied to the backend
    ▼
Frontend — Next.js + TypeScript + Tailwind            :3000
    ▼
Backend — FastAPI                                     :8000
    │  POST /api/analyze   GET /api/progress/{request_id}
    │  GET  /readiness     GET /health
    ▼
LangGraph workflow (Live mode):
    Planner → Researcher (with Retrieval) → Verifier → Proposer → Summarizer
    │  Demo mode instead serves the canonical Scenario from fixed content —
    │  no LLM calls, no retrieval
    ▼
Retrieval service layer
    ▼
PostgreSQL + pgvector (ingested Chunks)     Ollama (nomic-embed-text embeddings)
```

The four containers — frontend, backend, PostgreSQL with pgvector, and Ollama — come up together with `docker compose up -d --build`.

## Quickstart

Everything below runs from a fresh clone; Docker with Compose is the only tooling you need. Demo mode works immediately — no API key, no model pull, no ingest. Live mode adds three one-time prerequisites, marked below.

1. **Clone and configure:**
```bash
git clone https://github.com/RodrigoCovas/regula.git
cd regula
cp .env.example .env
```
Set `OPENROUTER_API_KEY` in `.env` to your provider key. Demo mode needs no key — leave it empty if you only want the demo. Compose reads this file automatically; it is the one configuration file a Docker deployment touches.

2. **Start the stack:**
```bash
docker compose up -d --build
```
The image build takes about a minute when warm (the first build on a machine also downloads base images). You end up with the web UI on http://localhost:3000, the API on port 8000, PostgreSQL with pgvector, and Ollama.

3. **Analyse your first scenario (Demo mode — works immediately):**
Open http://localhost:3000 and click **Try the demo scenario**. The answer arrives as Findings (each with a Strength badge), Citations with their provision relevance, and Actions, alongside the execution trace and known limitations. Via the API instead:

```bash
curl -s http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "demo",
    "scenario": {"id": "spanish-fintech-startup-uses-9e165169", "description": "A Spanish fintech startup that uses machine learning to assess creditworthiness for consumer loans. The platform automatically approves or denies applications based on applicant data including income, employment history, and spending patterns. The company operates only in Spain and plans to expand to other EU markets."},
    "question": "What regulations apply to our AI-based credit scoring platform?"
  }'
```
Demo mode serves only the canonical Spanish fintech Scenario; any other scenario id in Demo mode gets a Not-available response explaining the keyless demo.

4. **Check what is ready:**
```bash
curl http://localhost:8000/health
curl http://localhost:8000/readiness
# {"api_key_set":true,"embedding_model_present":false,"corpus_ingested":false}
```
`/health` is liveness only; `/readiness` reports the three Live-mode prerequisites as independently probed booleans. The example assumes the step-1 key is configured; a demo-only setup reads `api_key_set:false` — everything else below is unchanged. The other two booleans flip with the one-time prep in the next step.

5. **One-time Live prep — pull the embedding model, then ingest the Corpus:**
```bash
docker compose exec ollama ollama pull hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:F16
```
The pull takes ≈25 s. Then ingest (idempotent — re-running updates existing rows and inserts no duplicates):
```bash
docker compose exec backend python -m backend.src.ingest
```
Ingest embeds every Chunk of the three regulations locally with the pulled model and upserts it into PostgreSQL/pgvector with its provision metadata. Expect ≈6 min on CPU for the 764 Chunks — this is the slowest step in the guide, and it runs once. Ingest also reports any non-corpus JSON it skips (the eval ground truth `citations.json` lives beside the Corpus). Two things are normal on the way: before the first ingest, `docker compose logs backend` shows `WARNING: Could not check whether the Corpus is ingested (relation "chunks" does not exist)` on every boot — expected on a fresh stack, gone after ingesting. `/readiness` flips one boolean per step: `{true,false,false}` → `{true,true,false}` → `{true,true,true}`.

6. **Run a Live analysis.**
Switch the toggle to **Live** in the web UI and submit any scenario and question. The form hard-blocks Live until `/readiness` reads all-true, showing exactly which prerequisites are missing and the command for each (see [Modes & Readiness](#modes--readiness)). If you set the key in `.env` after starting the stack, run `docker compose up -d` once more so the backend container is recreated with it. A Live run answers through the real workflow over the ingested Corpus and the configured LLM; expect about 3 minutes per scenario, with the UI polling the progress endpoint and displaying each phase as it completes.

7. **Stop or reset:**
```bash
docker compose down        # stop; volumes (ingested Corpus, pulled model) survive
docker compose down -v     # full reset to fresh-clone state, including the Corpus and model
```

## Configuration

All configuration lives in the root `.env` (copied from `.env.example` in the Quickstart). Every variable is optional: deleting one or leaving it empty falls back to the backend's built-in default, except the key, whose emptiness means "not configured".

| Variable | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | *(empty)* | Provider key. Live mode needs it; Demo mode is keyless. The variable's name is historical — it holds whichever provider's key you use. |
| `LLM_MODEL` | `upstage/solar-pro4` | The chat model every completion asks for. Solar Pro 4 is the suggested default, not a requirement. |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible base URL ending at the version root — the client appends `/chat/completions` itself. |
| `EMBEDDING_MODEL` | `hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:F16` | Always served by the local Ollama. Changing it requires re-pull and re-ingest. |
| `REGULA_MODE` | `demo` | Server-side default mode for requests that omit `mode`. |

Provider notes worth knowing before you switch away from the default:

- **OpenRouter is the tested path.** Any OpenAI-compatible chat endpoint works the same way (`LLM_BASE_URL` + key in `OPENROUTER_API_KEY`): OpenAI, Groq, or a local server.
- **Local keyless servers still need a non-empty key value.** A Live-mode request without a configured key answers Not-available, and the client always sends the Bearer header — set `OPENROUTER_API_KEY=ollama` (any placeholder) for Ollama, vLLM, or LM Studio.
- **Avoid reasoning models.** The client sends `max_tokens` (the locked budget); OpenAI's o-series and gpt-5 reasoning models reject that parameter. Standard chat models — GPT-4o, Llama, Solar Pro 4 — accept it.
- **Embeddings are not covered by the provider switch.** They always run on local Ollama, and the pgvector column is fixed at nomic's 768 dimensions.
- **Results are model-sensitive** (ADR-0009): evaluation numbers are comparable only within the model that produced them — see [Evaluation](#evaluation).

Running the backend outside Docker (e.g. for tests) reads the same variables from `backend/.env.local` — see `backend/.env.example`.

## Modes & Readiness

Mode is a per-run choice (ADR-0008): every analysis request carries `mode` explicitly, and a request that omits it gets the server default (`REGULA_MODE`, itself defaulting to Demo). One running backend serves both modes. The web UI exposes the choice as a Demo/Live toggle that defaults to Demo on every page load and never persists.

- **Demo mode** (`mode: "demo"`) — keyless, deterministic, immediate. Serves only the canonical Spanish fintech Scenario (`spanish-fintech-startup-uses-9e165169`) from fixed content; derived scenario ids never trigger it (ADR-0005).
- **Live mode** (`mode: "live"`) — answers arbitrary scenarios through the real workflow: Planner → Researcher → Verifier → Proposer → Summarizer, retrieving Chunks from the ingested pgvector Corpus and producing evidence-backed Findings with Strength badges, metadata-derived Citations with Provision relevance, and Actions.

**Readiness** is the state that lets Live mode execute: a configured provider key, the embedding model available, and a non-empty ingested Corpus. `GET /readiness` probes each independently on every call and reports them as booleans — an unreachable Ollama or vector store reads `false`, never an error, so the endpoint is safe to poll while the stack comes up.

The checklist is how the UI enforces it. With Live selected and a prerequisite missing, the form hard-blocks submission and renders exactly the missing items — provider key (`OPENROUTER_API_KEY`), embedding model, ingested Corpus — each with a note and the verbatim command that fixes it, as text only. The UI never runs a pull, an ingest, or a key change for you; the three commands are the ones in the Quickstart's one-time prep.

The backend is the backstop: a Live request that races readiness, or a direct API caller, receives a Not-available response naming the missing prerequisite and its fix — never a server error, and never a boot refusal, since the stack always comes up ready to serve Demo mode.

## Evaluation

Answer quality is scored as three components, reported separately and **never blended** (ADR-0010):

- **Provision coverage** — a deterministic weighted set precision/recall/F1 over Citation targets (provision kind + number, never label strings). The expected side is weighted by the operator's per-provision Strength ratings (strong = 50, moderate = 5, weak = 1); every off-target produced Citation counts against precision.
- **Summary fidelity** — a strict rubric judge (the same configured provider, one batched call per case, schema-validated verdicts) compares provision-aligned Provision relevance against hand-authored ground-truth summaries; a contradiction between the two forces the floor score.
- **Strength agreement** — strength compared on the provisions both sides cite. It is reported as its own number and read at small weight: it grades only the Strength labels the workflow assigned, while coverage and fidelity grade the substance.

### Results

Run of 2026-08-30 on all 10 operator-authored ground-truth cases, produced by `upstage/solar-pro4` via OpenRouter (the configured default, recorded by the report artifact itself):

| Case | Coverage P | Coverage R | Coverage F1 | Summary fidelity | Strength agreement |
|---|---|---|---|---|---|
| live-retailer-breach | 0.286 | 0.240 | 0.261 | 1.000 | 1.000 |
| live-ai-recruitment-screening | 0.800 | 0.169 | 0.279 | 1.000 | 0.250 |
| live-bank-cloud-outage | 0.167 | 0.050 | 0.077 | 1.000 | 1.000 |
| live-employee-productivity-monitoring | 0.625 | 0.203 | 0.306 | 0.700 | 0.800 |
| live-telecom-chatbot | 0.500 | 0.065 | 0.115 | 1.000 | 1.000 |
| live-fintech-loan-recommendations | 1.000 | 0.209 | 0.346 | 0.750 | 0.875 |
| live-insurance-health-pricing | 0.833 | 0.197 | 0.319 | 1.000 | 1.000 |
| live-ransomware-investment-firm | 0.571 | 0.190 | 0.285 | 0.500 | 0.500 |
| live-genai-customer-service | 0.182 | 0.111 | 0.138 | 0.250 | 1.000 |
| live-ai-trading-cloud-attack | 0.400 | 0.058 | 0.101 | 1.000 | 0.000 |
| **Aggregate mean** | **0.536** | **0.149** | **0.223** | **0.820** | **0.742** |

The run meets the project's "works" bar on all four numbers: coverage mean precision ≥ 0.50 (measured 0.536), coverage mean F1 ≥ 0.20 (measured 0.223), summary fidelity mean ≥ 0.75 (measured 0.820), and strength agreement mean ≥ 0.50 (measured 0.742, the deliberately lenient small-weight component).

Reading the numbers:

- **Precision is pessimistic by construction.** A produced Citation outside the hand-authored expected set is not necessarily wrong. Before quoting these numbers, the produced-versus-expected spurious list was audited case by case: of 19 spurious targets, 4 were genuine misreads (documented as Summarizer/retrieval evidence for follow-up work), 6 were off-target retrieval drift, and 9 were acceptable supplements the ground truth simply does not list.
- **Recall is budget-bound by design.** The locked single-pass Planner budget (ADR-0003) researches a fixed number of targets per answer against expected sets spanning 11–45 provisions, so recall reads low by construction; precision is the quality signal inside that ceiling.
- **The Summarizer shipped on every case** — 118/118 produced Citations carried Provision relevance, so summary fidelity was measured on all 10 cases.
- **Numbers are model-sensitive** (ADR-0009): they are comparable only within the model that produced them. Switching models re-measures; it does not inherit this claim.

### Reproduce

With the stack running and the Live prerequisites in place (provider key, embedding model, ingested Corpus — see the Quickstart's one-time prep):

```bash
docker compose exec backend python -m backend.src.live_eval --output logs/live-eval-report.json
```

The CLI runs the ground-truth cases through `/api/analyze` in Live mode, prints per-case coverage, summary fidelity, and strength agreement plus the aggregate mean of each, and writes a JSON report artifact to `./logs` on the host — per-case scores, the produced-versus-expected dump for the human audit, and the `llm_model` that produced the numbers. A transient provider glitch aborts the run with the failing call named; the remedy is to re-run. Without a key or an ingested Corpus it refuses with the fix instead of measuring garbage.

Demo-mode cases are functional tripwires (ADR-0001), scored by the same harness: Demo production derives its Citations from the same locked targets its ground truth transcribes, so mean F1 is 1.0 by construction — any drop signals a regression, not poor quality.

## Development

**Tests and type checking** (no Docker required):

- Backend (Python 3.13+):
```bash
pip install -r backend/requirements.txt
python -m pytest backend/tests/ -q
```
Type checking (dev dependencies):
```bash
pip install -r backend/requirements-dev.txt
mypy
```
Configuration lives in `pyproject.toml` (`[tool.mypy]`).

- Frontend (Node.js 20+):
```bash
cd frontend
npm install
npm test
npm run typecheck
```

All four checks run on every push and pull request via GitHub Actions (`.github/workflows/ci.yml`).

**Backend dev server** (host-run, for backend development — the Docker stack already serves everything else):
```bash
python -m uvicorn backend.src.main:app --reload --reload-dir backend/src
```
The reload watch is scoped to `backend/src`, so frontend work never restarts the backend and a long Live run keeps its request id and progress history.

**Observability:** every `/api/analyze` request appends exactly one JSON line to `logs/queries.jsonl` (the `./logs` bind mount on the host; override the location with `QUERY_LOG_PATH`) — including failures, which carry an explicit failure status and the tokens spent before dying. Fields:

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

**Progress:** a Live submission that carries an `X-Request-Id` header runs in the background and reports each workflow phase to `GET /api/progress/{request_id}`, which returns the eventual Answer too — the UI polls it, and a dropped client connection never loses a run. The backend serves as a single worker process, which makes the in-memory registry safe.

**Project structure:**

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
│   │   ├── ingest.py               # Ingestion CLI
│   │   ├── llm.py                  # LLM client (OpenAI-compatible)
│   │   ├── availability.py         # Availability checks and Not-available responses
│   │   ├── readiness.py            # Live-mode readiness probes
│   │   ├── progress.py             # In-memory progress registry for workflow phases
│   │   ├── query_log.py            # Query logging to queries.jsonl
│   │   ├── live_workflow.py        # Live-mode workflow (Planner → Researcher → Verifier → Proposer → Summarizer)
│   │   ├── eval_harness.py         # Provision-coverage eval harness and Demo tripwires
│   │   ├── eval_judge.py           # Strict rubric judge (summary fidelity, strength agreement)
│   │   └── live_eval.py            # Live-mode evaluation CLI
│   ├── tests/                      # pytest test suite
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app/                        # Next.js App Router (layout.tsx, page.tsx)
│   ├── components/                 # React components (ScenarioForm, AnswerSurface, ModeToggle, ReadinessChecklist, …)
│   ├── lib/                        # Client logic (analyze-client, progress, readiness, scenario-id)
│   ├── tests/                      # Node test runner test suite
│   ├── package.json
│   ├── tsconfig.json
│   └── Dockerfile
├── data/
│   └── regulations/                # The Corpus (ai-act.json, gdpr.json, dora.json) and the eval ground truth (citations.json)
├── docs/
│   ├── adr/                        # Architecture Decision Records
│   └── agents/                     # Agent workflow documentation
├── research/                       # Research notes and decision trees
├── logs/                           # Query logs and eval report artifacts (gitignored)
├── docker-compose.yml
├── CONTEXT.md                      # Domain model and glossary
├── AGENTS.md                       # Agent skills and checks
└── README.md
```

## Deployment

**Permanently local:** Docker Compose only. Regula is not deployed to any cloud host by design; see `docs/adr/0002-regula-is-permanently-local-only.md`.

## License

MIT
