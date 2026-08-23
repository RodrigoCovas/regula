# Evaluation cases are functional tripwires until the LLM/RAG pipeline

The strength-weighted scorer (`backend/src/eval_harness.py`) is real, but while the demo is deterministic and ground truth is transcribed from the same locked content, mean F1 is 1.0 by construction: the curated cases pin contracts (routing leakage, strength tagging, demo-content drift) rather than measure answer quality. We decided to leave these simple cases as-is for the MVP, and to introduce quality evaluation only with the LLM/RAG pipeline, in a fixed order: author curated ground truth (the manual work on #7) → replace exact statement-string matching with semantic matching → add citation-fidelity scoring → only then grow the case suite into sophisticated quality cases.

## Consequences

- Exact statement-string matching and unscored citations are deliberate MVP simplifications, not oversights — do not "fix" them before the pipeline exists; a paraphrase-tolerant matcher has nothing real to judge while output is verbatim by construction.
- Any drop below F1 = 1.0 today is the intended signal: it means a regression (reworded Finding, retagged Strength, or routing leak onto non-canonical ids), not poor quality.
