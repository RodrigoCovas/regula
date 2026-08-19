# Regula — Project Brief

## Project

Build **Regula**, a small agentic regulatory research and compliance assistant.

The system lets a user describe a company/product scenario and ask a regulatory question, for example:

> "We are building an AI system that evaluates loan applications in Spain. What EU regulations could apply and what should we investigate before deployment?"

The system should:

1. Understand the request.
2. Plan the research.
3. Retrieve relevant evidence from regulatory documents.
4. Produce evidence-backed findings.
5. Verify the findings against the retrieved evidence.
6. Return an answer with citations.
7. Produce a short list of actionable next steps.

This is a **7-day prototype**, not a production recreation of Reversa or HappyRobot.

---

## Goals

### Primary goal

Build a technically credible backend that demonstrates:

- unstructured document ingestion,
- RAG/retrieval,
- agent orchestration,
- tool use,
- structured outputs,
- verification,
- evaluation,
- basic observability,
- clean backend architecture.

### Secondary goal

Use the project to learn the practical architecture and tooling behind:

- vector retrieval,
- embeddings,
- RAG pipelines,
- stateful agent orchestration,
- tool calling,
- agent verification,
- evaluation of agentic systems.

The architecture should remain simple enough that the developer can understand every important component.

---

## Target-role relevance

The project is intended to demonstrate capabilities relevant to:

### Reversa — Founding Engineer

Relevant themes:

- regulatory/unstructured data,
- retrieval systems,
- LLM agents,
- AI-native applications,
- structured knowledge extracted from documents,
- strong backend/data engineering.

### HappyRobot

Relevant themes:

- agent/tool orchestration,
- reliable AI workflows,
- ML/data pipelines,
- evaluation,
- observability,
- production-minded backend engineering.

The project should not attempt to reproduce proprietary systems.

---

## MVP scope

Implement one strong end-to-end workflow:

```text
User
  ↓
Planner
  ↓
Researcher
  ↓
Verifier
  ↓
Final answer + citations + actions
```

### Agents

Use only three logical agents:

**Planner**
- interprets the request,
- determines what needs to be researched.

**Researcher**
- uses retrieval tools,
- gathers relevant evidence,
- produces structured findings.

**Verifier**
- checks findings against the retrieved evidence,
- identifies unsupported or contradictory claims,
- approves or requests correction.

Do not add agents simply to increase apparent complexity.

---

## Regulatory corpus

Start with a small corpus containing:

- EU AI Act
- GDPR
- DORA

The corpus must remain small enough to ingest, inspect, test, and understand during the project.

The first demo scenario can involve a Spanish fintech using an AI system for loan assessment.

The application is a research prototype and must not present itself as professional legal advice.

---

## Core RAG pipeline

The initial retrieval architecture should support:

```text
Source documents
      ↓
Parsing
      ↓
Chunking
      ↓
Metadata
      ↓
Embeddings
      ↓
Vector store
      ↓
Retriever
      ↓
Agent context
      ↓
Cited answer
```

Citations are a core product requirement.

Important claims should be traceable to a source document and relevant article/section.

The system should prefer expressing insufficient evidence over inventing an answer.

Start with vector retrieval. Hybrid lexical/vector retrieval is optional.

---

## Initial architecture

Use simple components with clear boundaries.

Conceptually:

```text
Frontend
   ↓
FastAPI
   ↓
Application / Orchestration
   ↓
Agent workflow
   ↓
Retrieval interface
   ↓
PostgreSQL + pgvector
   ↓
Documents / embeddings / metadata
```

Keep retrieval, agent orchestration, domain/application logic, persistence, and presentation separated.

### Candidate stack

- Python
- FastAPI
- PostgreSQL
- pgvector
- LangGraph
- Next.js
- Tailwind

These are starting choices, not immutable requirements. Replace a component only when there is a concrete reason.

Prefer one database and a small number of dependencies.

---

## Agent workflow

The initial state graph should be approximately:

```text
START
  ↓
Planner
  ↓
Researcher
  ↓
Verifier
  ↓
END
```

The Researcher should access retrieval through an explicit tool/interface.

Agent outputs should use structured schemas.

Prefer deterministic application logic for:

- validation,
- persistence,
- formatting,
- workflow boundaries,
- evaluation,
- safety checks.

Use LLM reasoning where it adds genuine value.

---

## Evaluation

Create a small curated evaluation set, initially around 20 questions.

Each case should define the expected evidence/concepts needed to answer it.

Evaluate at least:

- retrieval correctness,
- citation correctness,
- answer correctness/faithfulness,
- workflow success.

The evaluation system should be simple and reproducible.

Do not build a large benchmark infrastructure.

---

## Observability

The application should expose a simple execution trace, without exposing private chain-of-thought.

Example:

```text
Planner       ✓
Researcher    ✓
Retrieved 8 passages
Verifier      ✓
Completed in 8.4s
```

Track basic:

- latency,
- tool calls,
- success/failure,
- optionally token usage/cost.

A heavyweight observability platform is optional.

---

## Frontend

The frontend should be visually polished but intentionally small.

Primary experience:

1. Enter a company/product scenario and question.
2. Start analysis.
3. Show final answer.
4. Show source citations.
5. Show recommended actions.
6. Show a compact execution trace.

Avoid building a full SaaS dashboard.

The backend is the main focus.

---

## Explicit non-goals

Do not build during the MVP:

- full regulatory knowledge graph,
- Neo4j architecture,
- autonomous agent swarms,
- complex workflow engine,
- multi-user/authentication infrastructure,
- Kubernetes,
- distributed inference,
- fine-tuning,
- voice interface,
- large-scale corpus ingestion,
- sophisticated memory,
- many external integrations.

These may be future extensions.

---

## 7-day constraint

The system must be achievable within approximately **7 intense working days**, including time spent learning unfamiliar RAG/agent tooling.

Priorities:

1. working vertical slice,
2. understandable architecture,
3. strong retrieval/evidence handling,
4. reliable agent flow,
5. basic evaluation,
6. polished demo.

When scope conflicts with the deadline, reduce features rather than adding architectural complexity.

---

## Desired user experience

Example:

**Input**

> We are a Spanish fintech using an AI system to evaluate loan applications. What EU regulations could apply and what should we investigate before deployment?

**Output**

- relevant regulations,
- concise evidence-backed explanation,
- citations to specific provisions,
- uncertainty/missing information where applicable,
- several recommended investigation/remediation actions.

The system should make clear which statements are directly supported by sources and which are conclusions derived from them.

---

## Future direction

The architecture should leave room for future work such as:

```text
MVP
 ↓
Hybrid retrieval
 ↓
Regulatory relationship graph
 ↓
Cross-jurisdiction reasoning
 ↓
Regulatory change detection
 ↓
Workflow/action execution
 ↓
Larger-scale ingestion
```

None of these are required for the MVP.

---

## Success criteria

The finished prototype should allow a user to:

- submit a realistic regulatory question,
- receive a useful answer,
- inspect source citations,
- see recommended actions,
- see a basic execution trace.

The codebase should have:

- clear module boundaries,
- testable retrieval,
- structured agent state,
- automated evaluation cases,
- basic error handling,
- local reproducibility,
- deployable packaging.

The developer should be able to explain the major architecture and trade-offs.

---

## Initial open decisions

These should be resolved during project design rather than silently assumed:

- exact document sources,
- document parsing approach,
- chunking strategy,
- embedding model,
- LLM/model provider,
- exact agent state schema,
- citation representation,
- retrieval interface,
- evaluation methodology,
- deployment target,
- observability approach,
- exact frontend interaction model.

---

## Reference material

Target roles:

- https://reversa.ai/careers/founding-engineer
- https://jobs.ashbyhq.com/happyrobot.ai/43d9bd48-7701-4719-affd-ecf92adfc37a

Engineering workflow reference:

- https://github.com/mattpocock/skills
