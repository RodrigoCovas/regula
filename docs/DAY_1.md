# Day 1: Setup & Data Pipeline

**Goal:** Local dev environment running, documents ingested, embeddings computed.
**Estimated time:** 7-8 hours
**Success criteria:** Docker-compose running, 500+ chunks embedded and queryable

---

## Task 1.1: Project Setup (1h)

**Status:** ✅ DONE (via scaffolding)

Project structure created with:
- Backend: FastAPI + Pydantic + LangGraph stubs
- Frontend: Next.js + TypeScript starter
- Docker Compose: PostgreSQL + pgvector + Ollama + FastAPI
- Git repo initialized

**Next:** Verify local setup works

---

## Task 1.2: Data Ingestion Pipeline (3h)

**Goal:** Parse EU regulations from JSON, chunk intelligently, add metadata.

### Step 1: Download regulatory documents

```bash
cd /home/rodrigo/Work/regula/data/regulations
curl -O https://raw.githubusercontent.com/alonalkalay-bolt/lexplorer/main/packages/data/gdpr.json
curl -O https://raw.githubusercontent.com/alonalkalay-bolt/lexplorer/main/packages/data/ai-act.json
curl -O https://raw.githubusercontent.com/alonalkalay-bolt/lexplorer/main/packages/data/dora.json

# Verify downloads
ls -lh *.json
```

### Step 2: Implement ingestion logic

Create `backend/src/ingest.py`:
- Load JSON files
- Parse articles + recitals from each regulation
- Extract text and metadata
- Implement hybrid chunking (preserve article boundaries, split long articles to ~400 tokens)
- Store chunks in PostgreSQL with metadata

**Key schema:**
```python
Chunk {
  id: UUID
  document: str  # "GDPR", "EU AI Act", "DORA"
  article_id: str  # "3", "10a"
  section_name: str  # "Recitals", "Title II"
  text: str  # actual content (~400 tokens)
  full_article: str  # original uncut article
  chunk_index: int  # chunk number within article
  created_at: datetime
}
```

### Step 3: Test with manual inspection

After ingestion, verify:
```bash
# Count chunks
SELECT COUNT(*) FROM chunks;
# Expected: ~500-700 chunks across 3 regulations

# Sample a chunk
SELECT document, article_id, section_name, text FROM chunks LIMIT 1;

# Verify metadata
SELECT DISTINCT document FROM chunks;
# Expected: GDPR, EU AI Act, DORA
```

---

## Task 1.3: Embeddings & Vector Storage (3h)

**Goal:** Compute embeddings for all chunks, store in pgvector, test retrieval.

### Step 1: Set up local Ollama

```bash
# Start Ollama (if using Docker)
docker-compose up ollama

# Or local installation: https://ollama.ai
ollama serve

# In another terminal, pull model
ollama pull nomic-embed-text
```

Verify:
```bash
curl http://localhost:11434/api/tags
# Should list "nomic-embed-text"
```

### Step 2: Test Ollama embedding

```bash
curl http://localhost:11434/api/embed -d '{
  "model": "nomic-embed-text",
  "input": "Test embedding"
}'
```

---

## Day 1 Deliverables

✅ Project scaffolding complete
✅ docker-compose.yml configured and running
✅ PostgreSQL + pgvector up
✅ Ollama with nomic-embed-text ready
✅ GDPR, AI Act, DORA documents downloaded (lexplorer JSON)
✅ Ingestion pipeline: JSON → chunks → PostgreSQL
✅ Hybrid chunking working (articles preserved, long articles split)
✅ Embeddings computed and stored in pgvector
✅ Vector search test passing
✅ ~500-700 chunks queryable by similarity

---

## Verification Checklist

- [ ] `docker-compose up -d` starts all services
- [ ] PostgreSQL accessible at `localhost:5432`
- [ ] Ollama embedder accessible at `localhost:11434`
- [ ] FastAPI runs at `localhost:8000` (health check: `curl http://localhost:8000/health`)
- [ ] Regulatory documents in `data/regulations/*.json`
- [ ] `python backend/src/ingest.py` completes without errors
- [ ] `SELECT COUNT(*) FROM chunks;` returns 500+
- [ ] Sample retrieval returns relevant passages

---

## Next: Day 2

Once Day 1 is complete, proceed to Day 2:
- Build Retriever service layer (search method)
- Define Pydantic schemas
- Create test harness for schemas
- Implement structured logging
