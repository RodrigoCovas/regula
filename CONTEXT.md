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
A reference that points back to the source document and specific provision or section supporting a claim; exact quotes are optional.

**Finding**:
A conclusion drawn from evidence about what likely applies or matters in the scenario.

**Answer**:
The single user-facing response that combines findings, citations, and actions.

**Action**:
A concrete next step the user should investigate or complete after reading the answer.

**Execution trace**:
A compact, user-visible summary of the workflow steps, retrieved passages, and tool calls used to produce the answer.  
_Avoid_: Chain-of-thought

**Claim**:
A statement the answer makes about what a regulation says or implies.

## Workflow

**Planner**:
Interprets the scenario and decides what should be researched.

**Researcher**:
Collects evidence from the corpus and turns it into structured findings.

**Verifier**:
Checks findings against the evidence and rejects unsupported claims.
