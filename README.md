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

### Running the tests locally (no Docker)

Requires Python 3.11+:
```bash
pip install -r backend/requirements.txt
pytest backend/tests/
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
docker exec regula-ollama-1 ollama pull nomic-embed-text
```

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

- **LLM:** DeepSeek V4 Flash (via OpenRouter)
- **Embeddings:** Ollama + nomic-embed-text
- **Orchestration:** LangGraph
- **Database:** PostgreSQL + pgvector
- **Backend:** FastAPI + Pydantic
- **Frontend:** Next.js + TypeScript + Tailwind

## Evaluation

20 curated test cases (10 simple, 10 nuanced) with metrics:
- Retrieval correctness (30%)
- Citation fidelity (40%)
- Answer correctness (20%)
- Workflow success (10%)

## Deployment

- **Local:** Docker Compose (fully reproducible)
- **Live Demo:** Render Free Tier

## Next Steps

1. [Day 1: Setup & Data Pipeline](./docs/DAY_1.md)
2. [Day 2: Retrieval & Schemas](./docs/DAY_2.md)
3. [Decision Tree](./DECISION_TREE.md)

## License

MIT

## Author

Built by Rodrigo with GitHub Copilot CLI
