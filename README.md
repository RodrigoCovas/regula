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

2. **Start the backend:**
```bash
docker compose up -d --build backend
```
This builds the FastAPI image and starts it on port 8000 (PostgreSQL starts as
a compose dependency but is not used by the demo). Ollama and pgvector are not
required by the demo.

3. **Send the canonical demo scenario:**
```bash
curl -s http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "scenario": {"id": "spanish-fintech", "description": "Spanish fintech lending"},
    "question": "What regulations apply?"
  }'
```
The response contains `answer`, `trace`, `detailed_trace`, and `known_limitations`
as siblings.
Only `scenario.id == "spanish-fintech"` triggers the demo; any other id gets a
helpful not-available response.

4. **Check the service is healthy:**
```bash
curl http://localhost:8000/health
```

5. **Stop when done:**
```bash
docker compose down
```

### Modes

One environment variable selects the mode before any request is served; nothing
ever silently degrades between paths:

- `REGULA_MODE=demo` (default) — keyless deterministic demo. Answers only
  `scenario.id == "spanish-fintech"` from fixed content.
- `REGULA_MODE=live` — requires `OPENROUTER_API_KEY`. Starting Live mode
  without the key refuses to boot with an error naming the missing variable.
  The Live research pipeline is not built yet, so Live-mode requests receive
  a Not-available response (never demo content).

### Running the tests locally (no Docker)

Requires Python 3.13+:
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

### Developer setup (full stack)

The steps below are for development beyond the deterministic demo.

- No API keys are required for the deterministic demo; the backend makes no
  LLM calls. The full stack below is only needed for development beyond the demo.
- Start everything (PostgreSQL + pgvector, Ollama, backend):
```bash
docker compose up -d
docker compose exec ollama ollama pull hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:F16
```

### Live mode setup (ingest the Corpus)

Live mode answers arbitrary Scenarios over the ingested Corpus. Setup is a
single documented command, run in this order — each step depends on the one
before it:

1. **Stack up** (PostgreSQL + pgvector and Ollama):
```bash
docker compose up -d postgres ollama
```

2. **Pull the embedding model** (nomic-embed-text v1.5 via Hugging Face — the
   Ollama registry mirror is not reachable from every network):
```bash
docker compose exec ollama ollama pull hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:F16
```

3. **Run ingestion** (idempotent — re-running updates existing rows and
   inserts no duplicates):
```bash
python -m backend.src.ingest
```
Run it from the repository root with the Python dependencies installed
(`pip install -r backend/requirements.txt`), or inside the stack:
```bash
docker compose exec backend python -m backend.src.ingest
```

Ingestion reads `data/regulations/*.json`, embeds every Chunk locally with the
pulled embedding model (`hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:F16` by
default; override with `--model`), and upserts it into PostgreSQL/pgvector
together with its provision metadata (Article XOR Recital XOR Annex). The
backend never ingests on startup; serving Live mode afterwards additionally
requires `REGULA_MODE=live` plus `OPENROUTER_API_KEY`.

Frontend runs on http://localhost:3000 (`cd frontend && npm install && npm run dev`)
Backend runs on http://localhost:8000

## Architecture

```
Frontend (Next.js + TypeScript + Tailwind)
    ↓
FastAPI Backend (/api/analyze)
    ↓
LangGraph Workflow:
  Planner → Researcher (with Retrieval Tool) → Verifier → Answer
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
│   │   ├── main.py                 # FastAPI app
│   │   ├── config.py               # Configuration
│   │   ├── db.py                   # Database setup
│   │   ├── models.py               # Pydantic schemas
│   │   ├── retriever.py            # Retrieval service
│   │   ├── agents/
│   │   │   ├── planner.py
│   │   │   ├── researcher.py
│   │   │   └── verifier.py
│   │   ├── workflow.py             # LangGraph orchestration
│   │   └── ingest.py               # Document ingestion
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   ├── components/
│   │   └── styles/
│   ├── package.json
│   └── tsconfig.json
├── data/
│   └── regulations/                # Ingested documents
├── logs/
│   └── queries.jsonl               # Query logs
├── docker-compose.yml
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
  - **LLM:** DeepSeek V4 Flash (via OpenRouter)
- **Embeddings:** Ollama + nomic-embed-text
- **Orchestration:** LangGraph
- **Database:** PostgreSQL + pgvector
- **Backend:** FastAPI + Pydantic
- **Frontend:** Next.js + TypeScript + Tailwind

## Evaluation

The deterministic demo ships with a strength-weighted eval harness
(`backend/src/eval_harness.py`): 7 curated cases scored by weighted F1 over
Findings, where Strength weights are strong = 3, moderate = 2, weak = 1.
Matched Findings mis-tagged on Strength keep `weight × (1 − distance/2)` of
their credit, and spurious Findings subtract their produced weight.

Until the LLM/RAG pipeline lands, these cases act as functional tripwires
(see `docs/adr/0001-eval-cases-are-functional-tripwires.md`): mean F1 is 1.0
by construction, so any drop signals a regression, not poor quality.

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
