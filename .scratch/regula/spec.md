Status: ready-for-agent

## Problem Statement

Regula needs a first working vertical slice that can answer a realistic regulatory question from a fixed curated corpus, with traceable citations, a compact execution trace, and a clear refusal to invent unsupported claims. The current repo has scaffolding and research, but not a complete end-to-end product flow.

## Solution

Build a small MVP that accepts a Scenario and Question, runs the Planner → Researcher → Verifier workflow, retrieves evidence from the fixed corpus, and returns a single Answer composed of findings, citations, and actions. The first demo should target the Spanish fintech / AI loan-assessment scenario and expose both a compact execution trace and a more detailed step/tool trace.

## User Stories

1. As a user, I want to submit a Scenario and Question, so that I can ask Regula about a realistic regulatory situation.
2. As a user, I want Regula to treat the Spanish fintech / AI loan-assessment case as the canonical demo, so that the first working slice feels concrete and credible.
3. As a user, I want Regula to return one Answer instead of several disconnected outputs, so that I can read the result in one place.
4. As a user, I want the Answer to include findings, citations, and actions, so that I can understand both the conclusion and the next steps.
5. As a user, I want citations to point to a document plus article/section/provision, so that I can trace each claim back to source material.
6. As a user, I want exact quotes to be optional, so that citations stay readable while still supporting traceability when needed.
7. As a user, I want Regula to prefer insufficient evidence over guessing, so that I can trust it not to overstate what the corpus supports.
8. As a user, I want to see a compact execution trace by default, so that I can understand how the answer was produced.
9. As a user, I want a more detailed execution trace that lists steps, retrieved passages, and tool calls, so that I can inspect the workflow without exposing private reasoning.
10. As a user, I want the Planner to identify what should be researched, so that the research stays focused.
11. As a user, I want the Researcher to gather evidence from the corpus, so that the answer is grounded in source material.
12. As a user, I want the Verifier to reject unsupported claims, so that the final Answer stays faithful to the evidence.
13. As a user, I want the corpus to be fixed and curated for the MVP, so that the system remains understandable and testable.
14. As a user, I want the corpus to include the EU AI Act, GDPR, and DORA, so that Regula can cover the intended regulatory domain.
15. As a user, I want Regula to keep the MVP small and local-first, so that I can run and inspect it during development.
16. As a user, I want the backend to expose a single API/analyze seam, so that the whole workflow can be tested through the user-facing contract.
17. As a user, I want the frontend to show the final Answer, citations, actions, and execution trace, so that the experience is useful as a demo.
18. As a user, I want the system to be reproducible locally, so that I can run the full slice without hidden setup steps.
19. As a user, I want basic evaluation cases for retrieval, citation fidelity, answer faithfulness, and workflow success, so that I can tell whether the MVP is working.
20. As a user, I want the product to stay clearly framed as a research prototype, so that it does not present itself as legal advice.

## Implementation Decisions

- Build a single end-to-end workflow with Planner, Researcher, and Verifier only.
- Treat the Scenario and Question as the input to the workflow.
- Return one Answer that contains findings, citations, and actions.
- Use a fixed curated corpus for the MVP rather than user-managed documents.
- Use the EU AI Act, GDPR, and DORA as the corpus seed.
- Keep citations at document + article/section/provision granularity, with optional supporting quotes.
- Prefer insufficient evidence over best-effort guessing when the corpus does not support a claim.
- Expose a compact execution trace by default, plus a more detailed trace of steps, retrieved passages, and tool calls.
- Use a single highest seam at the API/analyze boundary for the first slice; retrieval is exercised through that flow.
- Keep the backend architecture layered but simple: presentation, orchestration, retrieval interface, persistence, and document handling stay separate.
- Use deterministic application logic for validation, formatting, and workflow boundaries.
- Use structured outputs for agent results so the verifier and UI can consume them consistently.
- Keep the frontend intentionally small and focused on the analysis experience rather than a broad dashboard.
- Record the MVP as a local development workflow first, with local reproducibility as a core requirement.
- Make evaluation cases small and curated, centered on the intended demo scenario and adjacent regulatory questions.

## Testing Decisions

- Test external behavior through the API/analyze contract rather than internal implementation details.
- Verify that successful analyses return a single Answer with findings, citations, actions, and an execution trace.
- Verify that unsupported claims are downgraded or omitted rather than silently invented.
- Verify that citations include the expected source granularity and remain traceable to the corpus.
- Verify that the Spanish fintech / AI loan-assessment scenario produces the expected workflow shape and evidence profile.
- Verify that the detailed execution trace contains steps, retrieved passages, and tool calls.
- Verify the retrieval and citation behavior with a small curated evaluation set.
- Prior art: the repo already uses a single FastAPI entrypoint and a simple `/api/analyze` surface, so the tests should grow from that seam rather than introducing a new one.

## Out of Scope

- Full regulatory knowledge graphs.
- User-managed corpus ingestion.
- Multi-user authentication and account management.
- Distributed inference or fine-tuning.
- Autonomous agent swarms or a complex workflow engine.
- Neo4j or other graph-database modeling for the MVP.
- Large-scale corpus ingestion beyond the fixed starter set.
- Legal advice positioning or product claims beyond research-prototype scope.

## Further Notes

- This spec aligns the product vocabulary with `CONTEXT.md`.
- The initial demo should feel complete even if the internal implementation stays deliberately small.
- The next step after this spec is ticketing the work into ordered implementation slices.
