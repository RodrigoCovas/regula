# Regula — Regulatory Research & Compliance Assistant

A 7-day AI-native prototype for regulatory research and compliance analysis using agentic workflows, RAG, and structured knowledge extraction.

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ (for local development)
- Node.js 18+ (for frontend development)
- OpenRouter API key (for LLM access via DeepSeek)

### Setup

1. **Clone and initialize:**
```bash
cd /home/rodrigo/Work/regula
cp backend/.env.example backend/.env
# Edit backend/.env and add your OPENROUTER_API_KEY
```

2. **Start services:**
```bash
docker-compose up -d
```

This will start:
- PostgreSQL + pgvector (database at port 5432)
- Ollama embedding service (port 11434)
- FastAPI backend (port 8000)

3. **Initialize Ollama embeddings:**
```bash
docker exec regula-ollama ollama pull nomic-embed-text
```

4. **Ingest regulatory documents** (Day 1 task):
```bash
python backend/src/ingest.py
```

5. **Run backend tests:**
```bash
pytest backend/tests/
```

6. **Start frontend** (local dev, separate terminal):
```bash
cd frontend
npm install
npm run dev
```

Frontend runs on http://localhost:3000
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
