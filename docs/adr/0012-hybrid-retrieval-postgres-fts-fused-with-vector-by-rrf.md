# Hybrid retrieval: Postgres FTS fused with vector search by RRF

Vector-only retrieval missed lexically distinctive, duty-bearing provisions — the 2026-08-31 Live run scored recall below 0.40 in six of ten scenarios (e.g. bank-cloud-outage retrieved DORA broadly but missed Articles 18–20, 12, 13, 14), and the Evidence pool ceiling (18 chunks) capped recall against expected sets of 15–20 provisions. Retrieval is now hybrid: native Postgres full-text search (generated `tsvector` column on `chunks.content`, GIN index, `websearch_to_tsquery('english', …)`, added idempotently in `ensure_schema` — no migration framework exists) fused with the existing vector search by Reciprocal Rank Fusion (k = 60), which needs no score calibration between cosine and ts_rank. Per Research target the vector leg keeps its 0.55 similarity floor and takes top-12, the lexical leg takes top-8, and the plan-derived Evidence pool fills by the existing fair-share rule with per-target depth 12, 12 seats per target, and 1–6 Planner targets. The Planner is grounded in the Corpus by the deterministic Corpus inventory (titled Articles and Annexes rendered from Chunk metadata) instead of parametric knowledge alone. The Insufficient-evidence path now triggers only when **both** legs return nothing: lexical-only evidence may fill the pool.

## Considered options

- **BM25 extensions (paradedb/pg_search)** — rejected: a new image dependency for marginal gain at this corpus size; native FTS keeps the stack at one Postgres image.
- **Weighted score fusion** — rejected: cosine similarity and ts_rank are not comparable; rank-based RRF avoids calibration entirely.
- **Lexical results gated behind at least one vector hit above the floor** — rejected: it caps lexical recall for exactly the queries the embedder cannot see, which is the hole being fixed.
- **Ingest-time LLM keyword extraction to ground the Planner** — rejected: near-zero information gain on a corpus of three regulations any model knows parametrically, at the cost of a new ingest failure mode and re-ingest coupling; the deterministic Corpus inventory dominates it (zero LLM calls, complete provision-level table of contents, auto-scaling to custom corpora).

## Consequences

- A junk question can now lexically match boilerplate into the Evidence pool, weakening the junk-only guarantee the vector floor once provided alone; the second line of defense is claim-level strictness in the Researcher and Verifier prompts (scenario-relevance, not abstract restatement).
- The generated-column `ALTER` rewrites the chunks table once on first start after upgrade; at this corpus size the cost is negligible.
- Coverage weights remain expected-side only: fusion changes what recall can find, never what precision counts.
