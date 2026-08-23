# Regula — Complete Decision Tree

All decisions have been locked via the grilling process. See full breakdown in session workspace at `/home/rodrigo/.copilot/session-state/35513d9c-b5d0-4674-bbb5-27523852e920/files/REGULA_ROADMAP.md`

## Summary of Key Decisions

### Tier 1: Foundation
- **LLM:** nvidia/nemotron-3-ultra-550b-a55b:free (OpenRouter) — $0 for demo
- **Embeddings:** Ollama + nomic-embed-text (local, free)
- **Development:** Local docker-compose only (permanently local — see `docs/adr/0002-regula-is-permanently-local-only.md`)
- **Focus:** 80-90% backend, 10-20% frontend

### Tier 2: Data & Storage
- **Documents:** lexplorer JSON (GDPR, AI Act, DORA) — 2.5 MB, structured
- **Chunking:** Hybrid (preserve articles, ~400 tokens)
- **Vector Store:** PostgreSQL + pgvector
- **Citations:** Structured JSON + rendered

### Tier 3: Agent Architecture
- **Orchestration:** LangGraph (simple state graph)
- **Workflow:** START → Planner → Researcher → Verifier → END
- **Schemas:** Pydantic (all agents)
- **Retrieval:** Service layer + tool with LLM tool_choice

### Tier 4: Quality & Observability
- **Persistence:** Stateless + JSON logs
- **Token Budget:** Single-pass, ~8 chunks, 2K reasoning
- **Evaluation:** 20 cases (10 simple, 10 nuanced)
- **Scoring:** Retrieval 30%, Citation 40%, Answer 20%, Workflow 10%

### Tier 5: Deployment
- **Frontend:** TypeScript + Next.js + Tailwind (~300 lines)
- **Deployment:** Permanently local-only, docker-compose; cloud (incl. Render) rejected — see `docs/adr/0002-regula-is-permanently-local-only.md`
