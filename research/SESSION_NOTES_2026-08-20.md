# Regula — Grilling Session Notes (2026-08-20)

Design-tree decisions locked during the grilling session. Working language follows `CONTEXT.md` (Scenario, Regulatory question, Corpus, Regulation, Evidence, Citation, Finding, Answer, Action, Execution trace, Claim, Planner, Researcher, Verifier, Recital, Annex, Definition, Strength).

Companion files: `DECISION_TREE.md` (original locked decisions), `DEMO_SCENARIO_FINDINGS.md` (evidence research), `REGULA_PROJECT_BRIEF.md`, `DAY_1_RESEARCH.md`.

---

## Decisions locked this session

### Q1 — Lookup method
- The canonical demo uses a **deterministic lookup** (`corpus_lookup`: resolve a provision target directly, no embeddings).
- **Embeddings + pgvector** are the next lookup method, added later behind the same retrieval interface so the two can be swapped.

### Q2 — API response shape
- The API returns three **siblings**: `answer`, `trace`, `detailed_trace`.
- `Answer` holds only Findings, Citations, and Actions (no trace fields inside it). `models.py` still needs this change applied (currently nests trace).

### Q3 — Citation targets
- A Citation targets **exactly one** of three provision types: **Article**, **Recital**, or **Annex** (mutually exclusive, enforced by a Pydantic validator).
- Structured fields: `article_number`, `recital_number`, `annex_number` (all `Optional[int]`), plus `provision` free-text for sub-points (e.g. "point 5(b)") and optional `quote`.
- **Recital text is absent** for the AI Act and GDPR in the corpus (numbers + per-article references only; DORA has full recital text). A recital is cited by number; `quote` is simply absent — no availability marker needed.
- **Definitions are not a fourth type.** They are legally Article 3 (DORA's Article 3 text is extracted into `definitions[]`; `articleMap["3"]` is empty). Handle at the lookup layer, not in the schema.
- Annexes exist only in the AI Act corpus (`annex-1` … `annex-13`; official numbering matches the id).

### Q4 — Evidence strength transparency
- Each Finding carries a `strength` enum: `strong` (directly supported by the cited provisions) / `moderate` (derived or contingent on facts) / `weak` (framing only).
- Renamed `Finding.confidence` → `Finding.strength` (strength describes how directly Evidence supports a Claim; confidence implied probability).
- **Reversed this session:** `weak` findings are kept in the answer as user-facing conclusions — a weak Finding carries framing-only Evidence and stays; a statement with no Evidence at all is an Unsupported claim and is always discarded.
- Demo evidence upgraded to the researched set (see `DEMO_SCENARIO_FINDINGS.md`): AI Act 6(2)+Annex III pt 5(b), 26, 27, 86; GDPR 22+Recital 71, 35(3)(a); DORA 2, 1. The old demo cited AI Act Art 3 (definitions) to support an obligations claim — rejected as weak.

### Q5 — Insufficient-evidence rule
- **Unsupported claims and weak claims are different things.** A Claim with no Evidence in the corpus (nothing for or against it) is an **Unsupported claim**: always discarded, never in the answer. A **weak Finding** carries framing-only Evidence (e.g. a definitions Article) and **is kept** in the answer, badged weak.
- Definitions of strong/moderate/weak Findings written into `CONTEXT.md` (`Strength`): strong = provisions directly and explicitly support the claim; moderate = derived from provisions read together or contingent on facts the corpus can't settle; weak = framing only.

### Q5b — Disclosure of discarded claims
- No explicit note in the Answer. The **Execution trace records that the claim was rejected** as unsupported.
- Trace gains a record for each Unsupported claim considered and rejected.

### Q6 — Evaluation scope & strength-weighted scoring
- **6–8 curated cases** derived from the demo scenario (each a Scenario + Regulatory-question pair). The 20-case weighted plan in `DECISION_TREE.md` is discarded for the demo slice.
- Retrieval and workflow score dimensions deferred until pgvector lands; the MVP scores Findings and Citations only.
- **Strength-weighted scoring** (greenfield — no eval scaffolding exists yet):
  - Weights: `strong = 3`, `moderate = 2`, `weak = 1`.
  - Weighted recall: `Σ weight(matched expected findings) / Σ weight(all expected findings)`.
  - Spurious-claim penalty: a produced Finding not in ground truth subtracts `weight(its produced strength)` — a fabricated `strong` claim costs more than a fabricated `weak` one.
  - Composite per case: **weighted F1** (harmonic mean of weighted precision + weighted recall).

### Q6c — Distance-scaled mis-tag penalty
- A mis-tagged Finding retains `weight × (1 − distance/2)` of its credit, where distance is the step count on the ordered Strength scale (`weak < moderate < strong`):
  - distance 0 (exact): full weight.
  - distance 1 (off-by-one, e.g. weak tagged moderate): half weight.
  - distance 2 (e.g. weak tagged strong): zero weight — as bad as a wrong finding.

### Q7a — LLM model strategy
- Use **`nvidia/nemotron-3-ultra-550b-a55b:free` only** — accept the rate limits and the volatility. No env-var override to a paid model for the MVP.
- Rationale: the demo path makes no LLM calls today, so cost stays at $0.

### Q7b — Deployment venue
- **Local-only deploy, no cloud venue.** Run the demo via docker-compose locally, with step-by-step setup instructions for stakeholders. Rationale: a cloud demo with only one case scenario would disappoint stakeholders; local setup lets them run and poke at it themselves. No Render, no free-tier tuning, nothing throwaway.
- **Fact (verified 2026-08-20):** Render free tier has no free Postgres (managed Postgres is paid) and the free web tier (0.1 CPU / 512MB) cannot run Ollama — so "full stack on Render free tier" was never viable. The deployment-host decision (when the vector path lands) is deferred and flagged as a future ADR.
- Also verified: `nvidia/nemotron-3-ultra-550b-a55b:free` on OpenRouter is genuinely $0 but **rate-limited and volatile** (free tiers can rotate to paid without notice). The demo path makes **no LLM call today**, so the MVP demo is free regardless.

### Q8 — Corpus provenance & language
- Keep the lexplorer JSON for the MVP; record provenance **once at the corpus level** (a `PROVENANCE.md` beside `data/regulations/` — source URL, extraction date, tool, known gaps like the AI Act/GDPR recital-text absence). Not per-Citation: a Citation's job is legal pinpointing, not sourcing history.
- Demo answers always in **English** (corpus is English). A Spanish question is answered in English with the limitation surfaced as a known-limitation line in the answer/trace, not as a finding. Language drift is a known limitation.

### Q9 — Lookup selection
- Rule: **canonical demo scenario id → deterministic `corpus_lookup`; everything else → vector retrieval; never silently degrade.**
- The current keyword-heuristic routing in `main.py` (description containing "spanish"+"fintech", or question containing "spanish" + loan/credit/…) is **removed** — it silently routes non-canonical scenarios to the demo, which the rule forbids.
- Non-canonical requests get a **helpful "not available" response** (option 3): a clear message listing the one scenario the demo supports and how to invoke it (`scenario.id == "spanish-fintech"`), rather than a bare failure — so a local stakeholder run doesn't dead-end into confusion.
- The demo is only the start: later the system will support other questions; this session only ensures the deterministic prototype works as intended.

---

## Frontier
Empty — every branch of the design tree has been visited and locked.

## Glossary changes (CONTEXT.md)
- `Citation`: now "reference to one provision — an Article, a Recital, or an Annex".
- Added terms: `Recital`, `Annex`, `Definition`, `Strength` (with three-level definitions), `Unsupported claim`.
- `Finding`: now "tagged with a Strength"; weak findings kept as user-facing conclusions.
- `Claim`: clarified as becoming a Finding when supported by Evidence, or an Unsupported claim when not.

## Code changes applied
- `backend/src/models.py`: `Citation` gains `recital_number` + `annex_number` + exactly-one validator; `Finding.confidence` → `Finding.strength` (enum).
- `backend/src/main.py`: demo findings rewritten to the researched set with strengths; `corpus_lookup` resolves article/recital/annex targets; recitals found by number even without text.
- `backend/tests/test_analyze.py`: asserts exactly-one-target, recital + annex citations present, DORA in sources, strength on every finding. Test passes.

## Pending code changes (from locked decisions, not yet applied)
- Q2: `Answer` must stop nesting `trace`/`detailed_trace` — API returns `answer`/`trace`/`detailed_trace` as siblings.
- Q5/Q5b: the demo's `DEMO_FINDING_DEFS` carries no `weak` finding today; "weak findings stay in the answer" means reinstating a weak framing finding (e.g. the definitions finding from `DEMO_SCENARIO_FINDINGS.md` #8) and recording rejected Unsupported claims in the Execution trace.
- Q9: remove keyword-heuristic routing from `main.py`; trigger only on exact `scenario.id == "spanish-fintech"`; non-canonical requests get the helpful "not available" message.
- Q6: build the eval harness (6–8 curated cases, strength-weighted F1 with distance-scaled mis-tag penalty) — greenfield, nothing exists yet.
- Q8: create `PROVENANCE.md` beside `data/regulations/`.
- Q7b: step-by-step local setup instructions (docker-compose) for stakeholders.

## Session artifacts
- `research/DEMO_SCENARIO_FINDINGS.md` — evidence-backed findings for the demo scenario, every citation verified against `data/regulations/`. Highlights: creditworthiness AI is high-risk (Art 6(2) + Annex III pt 5(b)); GDPR 22/35 apply; DORA covers resilience, not credit decisions; claims the corpus does NOT support (e.g. "DORA applies to every fintech", "credit data is special-category data").

---

## Post-grilling wrap-up (same day)

### Spec & tickets published
- `research/SPEC_DETERMINISTIC_DEMO.md` written from the locked decisions, then published as **GitHub issue #1** (parent spec, label `ready-for-agent`, kept open).
- Broken into **six tracer-bullet tickets**, published as GitHub issues #2–#7 (see `docs/agents/issue-tracker.md`), each with **native blocking edges** (`issues/<n>/dependencies/blocked_by`, keyed by database id — see tracker doc):
  - **#2 API response contract & demo answer content** (Q2, Q5, Q5b) — no blockers.
  - **#3 Strength-weighted evaluation scorer** (Q6, Q6c) — no blockers.
  - **#4 Corpus provenance record** (Q8) — no blockers.
  - **#5 Local setup instructions** (Q7a, Q7b) — no blockers.
  - **#6 Lookup routing & helpful not-available response** (Q9) — blocked by #2.
  - **#7 Eval harness: curated cases & runner** (Q6) — blocked by #2, #3, #6.
- **Frontier (can start now):** #2, #3, #4, #5.
- Old local tickets under `.scratch/regula/issues/` (from the pre-grilling spec) deleted.

### Notes for next session
- `gh` auth works via the system keyring (`RodrigoCovas`); if it ever fails, run `gh auth login`.
- **Bash gotcha:** backtick-quoted inline code in a `gh issue create/--body` string is consumed by bash command substitution — use plain quotes in issue bodies, or escape backticks. Issue #2's first criterion was mangled this way and manually repaired.
- The `ready-for-agent` label exists on the repo (created this session).
- Domain model (`CONTEXT.md`) now carries the three-level `Strength` definitions, `Unsupported claim`, and the weak-Finding-kept rule — read it before implementing tickets #2/#3/#7.
- All of the above is committed in `173a656` ("Plan stage and specs"); only the `.scratch/regula/issues/` deletions remain uncommitted.