# Regula

Regula is a regulatory research and compliance assistant for answering questions from a small legal corpus with evidence-backed findings and citations. It helps users find the Regulations relevant to their Scenario so qualified legal professionals can act on them; it never dispenses legal advice and never substitutes for professional judgment.

## Frontend Runtime

**Node.js**: 20 (LTS) — pinned in `frontend/.nvmrc`

**Next.js**: 16.3.3 — pinned in `frontend/package.json`

**React**: 18.3.1 — pinned in `frontend/package.json`

## Frontend Dependency Security

**Audit status**: 0 vulnerabilities (verified `npm audit` clean)

**Install script policy**: Two packages have approved install scripts in `frontend/package.json`:
- `esbuild@0.28.2` — native binary installer for platform-specific build tool; required for TypeScript transpilation
- `unrs-resolver@1.12.2` — native resolver for Next.js module resolution; required for build

Both are build-time dependencies with well-maintained postinstall scripts that download platform-specific binaries. No runtime code execution risk.

**ESLint deprecation**: `eslint@9.39.5` carries an npm deprecation notice, but this is a dev-only tool. The deprecation does not produce application-actionable warnings — it does not affect the running server, production build, or end users. ESLint 10.x requires breaking changes to plugin APIs that `eslint-config-next` does not yet support.

## Language

**Scenario**:
The user's described company, product, and jurisdiction — the context for a regulatory question. The same scenario can have multiple questions.
_Avoid_: Case, request, prompt

**Regulatory question**:
A question asking what rules, obligations, or risks may apply in a scenario. It leans toward applicability (which Regulations govern) or obligations (what must be done); surfacing obligation depth is the workflow's job regardless of how the user phrases the question.

**Research target**:
One regulation-scoped line of inquiry the Planner derives from the Regulatory question; a question decomposes into one or more targets. The Evidence pool is sized so every target keeps representation.  
_Avoid_: Source, query, sub-question

**Applicability target**:
The one Research target the Planner aims at a Regulation the Scenario plausibly implicates but does not confirm; it retrieves that Regulation's perimeter provision so an exclusion can cite it.
_Avoid_: Scope check, applicability question

**Engagement-threshold target**:
The one Research target the Planner reserves for a Regulation the Scenario confirms; it retrieves the provision that decides whether the regime engages at all — a breach notion, a profiling notion, a financial-entity perimeter, or the principles constraining the contested processing — so the Answer cites the threshold instead of presupposing it.
_Avoid_: Engagement check, threshold question

**Perimeter provision**:
The provision of a Regulation that decides who the Regulation covers — typically its scope Article — and the Citation an exclusion Finding rests on.
_Avoid_: Scope provision, applicability clause

**Corpus**:
The fixed curated set of source documents Regula reasons over in the MVP.  
_Avoid_: Knowledge base, dataset

**Regulation**:
A governing legal instrument such as the EU AI Act, GDPR, or DORA.

**Evidence**:
Relevant source text that supports or contradicts a finding.

**Evidence pool**:
The bounded set of Chunks chosen from retrieval to support drafting — the only Evidence the Researcher and Verifier reason over. Its size derives from the plan: every Research target claims `SEATS_PER_TARGET` **seats**, so the pool grows with the number of targets and no target is starved. It surfaces in the Execution trace as the retrieved passages.  
_Avoid_: Chunk budget, result set, context window

**Corpus inventory**:
The Corpus's own table of contents — one entry per titled provision (Articles and Annexes), rendered from Chunk metadata and shown to the Planner. It grounds target formulation in the ingested Corpus's actual vocabulary instead of the Planner's prior knowledge.
_Avoid_: Keyword index, document list, catalogue

**Citation**:
A reference to one provision of a source document — an Article, a Recital, or an Annex — supporting a Claim; exact quotes are optional.
_Avoid_: Reference, source link

**Citation strength**:
How central a cited provision is to the overall Answer, rated at three levels: strong when the provision directly imposes or decides the obligations the Answer turns on, moderate when it is a supporting duty or factor the Answer relies on, weak when it is definitional or framing. Rated per provision for the whole Answer, calibrated against the whole cited set — strong is reserved for the few provisions the Answer would collapse without — and drawing only on the content of the Findings citing it. Two Findings citing the same provision can carry different Finding strengths; the provision carries one Citation strength. Distinct from a Finding's Strength, which judges how directly Evidence supports one Finding.
_Avoid_: Confidence, derived strength, average strength, Finding strength

**Provision relevance**:
The answer-wide statement of why one cited provision matters to the overall Answer, aggregating across the Findings that cite it. It draws only on the content of those Findings — never on material outside the Answer.
_Avoid_: Citation summary, importance score

**Recital**:
A numbered explanatory paragraph at the start of a Regulation that states the intent behind the rules.
_Avoid_: Preamble, whereas-clause, rationale

**Annex**:
A numbered appendix to a Regulation that contains lists or technical detail referenced by its Articles.
_Avoid_: Appendix, schedule

**Definition**:
A term and its official meaning set out in a Regulation, typically in its definitions Article.
_Avoid_: Glossary, dictionary

**Chunk**:
A retrievable passage of a Regulation — typically a whole Article or Recital — carrying the provision numbers needed to build Citations.
_Avoid_: Passage, snippet, segment

**Finding**:
A conclusion drawn from Evidence about what likely applies or matters in the Scenario, tagged with a Strength.

**Strength**:
How directly the cited Evidence supports a Finding, at three levels:
- **strong**: the cited provisions directly and explicitly support the claim — they name the exact situation, use case, or obligation (e.g. Annex III point 5(b) names creditworthiness evaluation as high-risk).
- **moderate**: the claim is derived from the cited provisions read together, or is contingent on facts the Scenario text leaves open (e.g. a financial-entity licence status the Scenario does not state).
- **weak**: the cited provisions are framing only — they supply vocabulary or context (e.g. a definitions Article) without establishing an obligation on their own.
Every Finding carries Evidence and stays in the Answer, badged with its Strength. A statement with no Evidence is not a Finding and never appears in the Answer.
_Avoid_: Confidence, probability

  **Finding strength**:
  The Strength of one Finding: how directly that Finding's own Evidence supports that Finding's claim. Judged per Finding, on the provisions that Finding cites.
  _Avoid_: Citation strength, average strength

**Exclusion Finding**:
A Finding that a Regulation does not reach the Scenario, kept when the Evidence pool holds that Regulation's perimeter provision.
_Avoid_: Negative finding, out-of-scope finding

**Unsupported claim**:
A statement with no Evidence in the Corpus either way — no provision can be cited for or against it; one that merely restates provisions without bearing on the question asked for the Scenario; or one whose contingency the Scenario text settles out of scope. Unsupported claims are always discarded and never appear in the Answer. Distinct from a weak Finding, which carries framing-only Evidence.
_Avoid_: Unverified claim, speculation

**Insufficient evidence**:
The situation where retrieval returns nothing for any Research target — neither the vector nor the lexical search finds a single Chunk. Surfaced as a Known limitation together with suggested Actions; it never produces Findings. Lexical-only evidence fills the Evidence pool instead of triggering it.
_Avoid_: No results, empty answer

**Answer**:
The user-facing conclusions: the Findings, the Citations with their Provision relevance, and the Actions. The API returns the Answer, the Execution trace, the detailed trace, and the Known limitations as siblings — workflow machinery never nests inside the Answer.

**Action**:
A pointer that aids legal professionals acting on an Answer: it names something only they can settle — a fact the Scenario text leaves open, or verification of the cited provisions against the company's actual situation. An Action never presumes a Regulation applies to the Scenario and never substitutes for professional judgment.
_Avoid_: Legal advice, recommendation, compliance task

**Action proposal**:
A candidate Action the Proposer emits for the grounding gate; only proposals whose Citations resolve to a kept moderate- or strong Finding become Actions, and rejected proposals stay visible in the detailed trace.

**Known limitation**:
A stated boundary of the current system, surfaced alongside every response but never among its Findings or Actions — e.g. the Corpus is English-only, or Regula is a research prototype whose Answers require verification by a qualified professional.
_Avoid_: Caveat, disclaimer

**Execution trace**:
A compact, user-visible summary of the workflow steps, retrieved passages, and tool calls used to produce the answer.  
_Avoid_: Chain-of-thought

**Claim**:
A statement the answer makes about what a regulation says or implies. A Claim becomes a Finding when supported by Evidence that bears on the question asked for the Scenario; a Claim with no Evidence, one that merely restates provisions without bearing on that question, or one whose contingency the Scenario text settles out of scope is an Unsupported claim and is discarded.

**Demo mode**:
The keyless path that serves the canonical Spanish fintech Scenario from fixed content — no LLM calls, no vector retrieval. Mode is an explicit per-run choice, never inferred from the Scenario's text or id.
_Avoid_: Offline mode, mock mode

**Live mode**:
The path where the workflow runs for real over the ingested Corpus, answering arbitrary Scenarios via hybrid lexical and vector retrieval. Executable only when Readiness holds.
_Avoid_: Production mode, online mode

**Readiness**:
The state that lets a run execute in Live mode: a configured provider key, the embedding model available, and a non-empty ingested Corpus. Missing readiness yields the Not-available response, never a degraded Answer.
_Avoid_: Health, setup status

**Not-available response**:
A user-facing reply explaining why a requested capability cannot be served — an unknown Scenario in Demo mode, or missing Readiness in Live mode (missing provider key, un-ingested Corpus, unavailable embedding model, unreachable store or LLM provider) — and how to proceed.
_Avoid_: Error message, fallback, dead end

## Workflow

**Planner**:
Interprets the scenario and decides what should be researched.

**Researcher**:
Collects evidence from the corpus and turns it into structured findings.

**Verifier**:
Checks findings against the evidence and rejects unsupported claims.

**Proposer**:
Turns kept Findings into referral Actions for legal professionals; every Action it emits must be anchored in a Finding's Evidence, and it never advises on its own authority.

**Summarizer**:
Turns the kept, cited Findings into Provision relevance — one grounded statement per cited provision — and rates each cited provision's Citation strength; both draw only on the content of the Findings citing it.
