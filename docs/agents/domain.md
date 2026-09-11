# Domain Docs

How agents should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root — the domain model and glossary: the workflow roles (Planner, Researcher, Verifier, Proposer, Summarizer), the answer vocabulary (Finding, Citation, Strength, Execution trace, …), and the mode/readiness terms (Demo mode, Live mode, Readiness, Not-available response).
- **`docs/adr/`** — numbered, kebab-case Architecture Decision Records. Read the ADRs that touch the area you're about to work in.

Both exist in this repo. If a doc you expect is missing, proceed silently — don't flag its absence or suggest creating one upfront. New glossary terms and durable decisions get recorded in these files when they are actually resolved during work, not speculatively.

## File structure

This is a single-context repo: one root `CONTEXT.md`, system-wide ADRs under `docs/adr/`.

```
/
├── CONTEXT.md                       # Domain model and glossary
├── docs/adr/                        # Numbered, kebab-case Architecture Decision Records
├── backend/         # FastAPI app + the agent workflow (LangGraph)
├── frontend/        # Next.js UI
├── data/            # The Corpus (data/regulations/) and its provenance record
└── research/        # Project brief and decision tree (process documentation)
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids — where an entry lists the alternatives to avoid, they bind.

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing language the project doesn't use (reconsider) or there's a real gap (propose the term to the maintainer before coining one).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0008 (mode is a per-run choice) — but worth reopening because…_
