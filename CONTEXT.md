# Regula

Regula is a regulatory research and compliance assistant for answering questions from a small legal corpus with evidence-backed findings and citations.

## Language

**Scenario**:
The user’s described company, product, jurisdiction, and question.  
_Avoid_: Case, request, prompt

**Regulatory question**:
A question asking what rules, obligations, or risks may apply in a scenario.

**Corpus**:
The fixed curated set of source documents Regula reasons over in the MVP.  
_Avoid_: Knowledge base, dataset

**Regulation**:
A governing legal instrument such as the EU AI Act, GDPR, or DORA.

**Evidence**:
Relevant source text that supports or contradicts a finding.

**Citation**:
A reference to one provision of a source document — an Article, a Recital, or an Annex — supporting a Claim; exact quotes are optional.
_Avoid_: Reference, source link

**Recital**:
A numbered explanatory paragraph at the start of a Regulation that states the intent behind the rules.
_Avoid_: Preamble, whereas-clause, rationale

**Annex**:
A numbered appendix to a Regulation that contains lists or technical detail referenced by its Articles.
_Avoid_: Appendix, schedule

**Definition**:
A term and its official meaning set out in a Regulation, typically in its definitions Article.
_Avoid_: Glossary, dictionary

**Finding**:
A conclusion drawn from Evidence about what likely applies or matters in the Scenario, tagged with a Strength.

**Strength**:
How directly the cited Evidence supports a Finding, at three levels:
- **strong**: the cited provisions directly and explicitly support the claim — they name the exact situation, use case, or obligation (e.g. Annex III point 5(b) names creditworthiness evaluation as high-risk).
- **moderate**: the claim is derived from the cited provisions read together, or is contingent on facts the Corpus cannot settle (e.g. whether the company is a licensed financial entity under DORA).
- **weak**: the cited provisions are framing only — they supply vocabulary or context (e.g. a definitions Article) without establishing an obligation on their own.
Every Finding carries Evidence and stays in the Answer, badged with its Strength. A statement with no Evidence is not a Finding and never appears in the Answer.
_Avoid_: Confidence, probability

**Unsupported claim**:
A statement with no Evidence in the Corpus either way — no provision can be cited for or against it. Unsupported claims are always discarded and never appear in the Answer. Distinct from a weak Finding, which carries framing-only Evidence.
_Avoid_: Unverified claim, speculation

**Answer**:
The user-facing conclusions: the Findings, Citations, and Actions. The API returns the Answer, the Execution trace, and the detailed trace as siblings — workflow machinery never nests inside the Answer.

**Action**:
A concrete next step the user should investigate or complete after reading the answer.

**Execution trace**:
A compact, user-visible summary of the workflow steps, retrieved passages, and tool calls used to produce the answer.  
_Avoid_: Chain-of-thought

**Claim**:
A statement the answer makes about what a regulation says or implies. A Claim becomes a Finding when supported by Evidence; a Claim with no Evidence is an Unsupported claim and is discarded.

## Workflow

**Planner**:
Interprets the scenario and decides what should be researched.

**Researcher**:
Collects evidence from the corpus and turns it into structured findings.

**Verifier**:
Checks findings against the evidence and rejects unsupported claims.
