# Provider configuration is environment-driven

The LLM was deliberately pinned to OpenRouter's `upstage/solar-pro4` with no env override (a repin from the free nemotron tier, recorded as an amendment on spec #9) — the pin bought eval reproducibility at the cost of flexibility. Interviewers and operators now bring their own key and model, so `LLM_MODEL`, `LLM_BASE_URL`, and the API key are environment configuration, using the OpenAI-compatible base-URL convention (`/chat/completions` appended to the base). Solar Pro 4 stays the default and the README's suggested model; the pin's history (free-tier rate limits, upstream errors) remains visible in the query log.

Update 2026-08-29 (#43): the embedding model joins the same treatment — `EMBEDDING_MODEL` (default nomic) is read by the backend's retriever and the ingest CLI, so both ends of the vector path move together. Changing it requires re-ingest (documented in `.env.example`); detecting ingest/query mismatch is deferred. The backend loads `backend/.env.local` at settings load, making the documented configuration home real for every entry point.

Update 2026-09-11: the configuration home moves to the root `.env` — the same file Compose reads — so a Docker deployment and a host-run backend (dev server, ingest CLI, eval commands) share one configuration file. `backend/.env.local` and `backend/.env.example` are retired; `backend/.env.example`'s documentation of `DATABASE_URL` and `OLLAMA_API_URL` moved into the root `.env.example` with host-run-only notes.

## Consequences

- Eval results are model-sensitive: the README quotes the model that produced them, and numbers across models are not directly comparable.
- OpenRouter is the tested path; other OpenAI-compatible endpoints are expected to work but untested.
