# Regula — Deterministic Demo Prototype: Locked Decisions & Pending Work

## Problem Statement

Regula is a regulatory research and compliance assistant that answers questions from a small legal corpus with evidence-backed Findings, Citations, and Actions. A grilling session (2026-08-20) locked the full design tree for the MVP's deterministic demo slice: the answer structure, the evidence-strength model, the evaluation scheme, the lookup-selection rule, and the deployment venue. Several of these locked decisions are **not yet reflected in the code** — the API still nests the trace inside the Answer, the demo carries no weak Findings, routing silently degrades via keyword heuristics, and there is no evaluation harness, corpus provenance, or stakeholder setup guide.

## Solution

The deterministic demo prototype is brought in line with the locked decisions so it works as intended: an API that returns `answer`/`trace`/`detailed_trace` as siblings, a demo answer that keeps weak Findings and discards Unsupported claims (recording them in the trace), routing that triggers only on the canonical scenario id and responds helpfully otherwise, a strength-weighted evaluation harness of curated cases, corpus-level provenance, and step-by-step local setup instructions.

## User Stories

1. As a stakeholder running the demo locally, I want to follow step-by-step setup instructions, so that I can start the app without external dependencies or a cloud account.
2. As a stakeholder, I want to submit the canonical Spanish fintech scenario, so that I get a complete, evidence-backed Answer.
3. As a stakeholder, I want the API to return the answer and the trace as separate siblings, so that the conclusions are not polluted by workflow machinery.
4. As a stakeholder, I want each Finding to carry a Strength badge, so that I can judge how directly the cited provisions support it.
5. As a stakeholder, I want weak framing Findings (e.g. the definitions Article) to stay visible in the answer, so that I see the vocabulary the stronger Findings rely on.
6. As a stakeholder, I want claims the corpus cannot support to be absent from the answer, so that I am never shown an unsupported conclusion.
7. As a stakeholder, I want rejected Unsupported claims to be recorded in the Execution trace, so that I can audit what was considered and refused.
8. As a stakeholder, I want a helpful response when I ask a non-canonical question, so that I learn the demo's scope and exactly how to invoke the supported scenario instead of hitting a dead end.
9. As a stakeholder, I want to know the corpus is English-only, so that I am not surprised when a Spanish question is answered in English.
10. As a maintainer, I want an evaluation harness of 6–8 curated cases, so that I can verify the deterministic prototype behaves as intended.
11. As a maintainer, I want findings scored by Strength-weighted F1, so that getting strong Findings right dominates the score.
12. As a maintainer, I want mis-tagged Findings penalized by the distance on the Strength scale, so that a weak↔strong reversal costs the whole Finding while an off-by-one badge costs half.
13. As a maintainer, I want spurious Findings to subtract credit by the Strength they were produced with, so that a fabricated strong claim hurts more than a fabricated weak one.
14. As a maintainer, I want the keyword-heuristic routing removed, so that a non-canonical Scenario never silently receives the demo Answer.
15. As a maintainer, I want corpus provenance recorded once at the corpus level, so that I know the source, extraction date, tool, and known gaps.
16. As a maintainer, I want the demo to run without vector retrieval, so that the deterministic slice is self-contained and free to operate.

## Implementation Decisions

### API response shape (Q2)
- The API returns three **siblings**: `answer`, `trace`, `detailed_trace`.
- `Answer` holds only Findings, Citations, and Actions — no trace fields inside it.
- Apply: `backend/src/models.py` currently nests `trace`/`detailed_trace` inside `Answer`; move them out to the top-level response shape. The `AnalyzeResponse` schema is the contract.

### Evidence strength model (Q4, Q5, Q5b)
- `Finding` carries a `Strength` enum (`strong` / `moderate` / `weak`), surfaced as a user-facing badge.
- **Weak Findings are kept** in the answer as user-facing conclusions (framing evidence — e.g. the definitions Article). The demo's `DEMO_FINDING_DEFS` currently carries no weak Finding; reinstate the definitions finding (see `research/DEMO_SCENARIO_FINDINGS.md` #8) as a weak, framing-only Finding.
- **Unsupported claims** (no Evidence in the Corpus either way) are **always discarded** — never in the Answer.
- When an Unsupported claim is discarded, the **Execution trace records the rejection** (no explicit note in the answer).

### Lookup selection (Q9)
- Trigger the deterministic demo **only** on exact `scenario.id == "spanish-fintech"`.
- Remove the keyword-heuristic routing in `backend/src/main.py` (description/question keyword matching) — it silently routes non-canonical Scenarios to the demo, which the "never silently degrade" rule forbids.
- Non-canonical requests return a **helpful "not available" response**: a clear message stating the demo currently supports only the canonical scenario and how to invoke it (`scenario.id == "spanish-fintech"`), rather than a bare failure or a guessed demo Answer.
- The deterministic `corpus_lookup` resolves Article/Recital/Annex targets (already applied); recitals are cited by number even when their text is absent from the corpus.

### Evaluation harness (Q6, Q6c)
- Build a greenfield eval harness: **6–8 curated cases**, each a Scenario + Regulatory-question pair with a ground-truth Answer (expected Findings, each with expected Strength and expected Citations).
- **Scoring:**
  - Strength weights: `strong = 3`, `moderate = 2`, `weak = 1`.
  - Weighted recall: `Σ weight(matched expected findings) / Σ weight(all expected findings)`.
  - Weighted precision accounts for spurious Findings: a produced Finding not in ground truth subtracts `weight(its produced strength)`.
  - Composite per case: **weighted F1** (harmonic mean of weighted precision and weighted recall).
  - Mis-tag penalty: a matched Finding retains `weight × (1 − distance/2)` of its credit, where distance is the step count on the ordered Strength scale (`weak < moderate < strong`). Distance 0 = full; distance 1 = half; distance 2 (weak↔strong) = zero.
- The retrieval and workflow score dimensions are deferred until pgvector lands.

### Corpus provenance & language (Q8)
- Keep the lexplorer JSON for the MVP.
- Record provenance **once at the corpus level**: create `PROVENANCE.md` beside `data/regulations/` covering source URLs, extraction date, extraction tool, and known gaps (notably: AI Act and GDPR recital text absent, DORA carries full recital text; AI Act annex numbering shifted by one in the JSON).
- Demo answers are always in English; surface "corpus is English-only" as a known-limitation line in the answer/trace, not as a Finding.

### Local deployment (Q7a, Q7b)
- Use `nvidia/nemotron-3-ultra-550b-a55b:free` only (no env-var override to a paid model).
- **Local-only deploy**: no cloud venue. Provide step-by-step setup instructions (docker-compose) for stakeholders. The full stack (Ollama + pgvector) is not required by the demo; the deployment-host decision is deferred to a future ADR.

## Testing Decisions

- **What makes a good test:** test external behavior through the `/api/analyze` seam, never implementation details. Assert on the response contract (shape, presence/absence, values), not on internal functions.
- **Seam:** the existing `POST /api/analyze` endpoint via `TestClient` in `backend/tests/test_analyze.py` — the highest seam available, reused rather than adding new ones. The eval harness is a separate offline module, not an HTTP test.
- **Modules tested:**
  - `backend/tests/test_analyze.py` (extended): asserts sibling response shape (`answer`/`trace`/`detailed_trace`, no trace nested inside `answer`); a weak Finding present with a strength badge; Unsupported claims absent from the Answer but present in the trace; non-canonical requests receive the helpful not-available response and are NOT routed to the demo via keywords; the existing exactly-one-target citation invariant still holds.
  - Eval harness tests: feed known curated cases and assert the computed score, including a spurious-Finding penalty and the distance-scaled mis-tag penalty.
- **Prior art:** the existing `test_analyze.py` already asserts the exactly-one-target citation rule, recital + annex citations, DORA in sources, and a Strength on every Finding — extend in the same style.

## Out of Scope

- Embeddings + pgvector vector retrieval (deferred until the retrieval interface is built).
- LLM-driven Planner/Researcher/Verifier agents (the demo is fully deterministic today).
- Frontend work.
- Any cloud deployment / Render hosting.
- Restoring recital text for the AI Act or GDPR.
- Spain-specific national requirements (outside the corpus entirely).

## Further Notes

- All decisions here were locked via grilling on 2026-08-20; see `research/SESSION_NOTES_2026-08-20.md` for the full design tree, `research/DEMO_SCENARIO_FINDINGS.md` for the evidence-backed findings, and `CONTEXT.md` for the domain glossary (`Strength`, `Unsupported claim`, `Citation`, etc.).
- The demo is only the start: later the system will support other questions; this work only ensures the deterministic prototype works as intended.
- The future deployment-host decision is flagged as a future ADR (hard to reverse, surprising without context, a real trade-off).