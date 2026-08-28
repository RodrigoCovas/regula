# Evaluation scores provisions, not statements

The 2026-08-28 Live eval returned precision 0.000 on all 11 cases while every run succeeded and produced Findings: the semantic statement matcher (token-set cosine, threshold 0.43) paired almost nothing — it was calibrated on a single case with a 0.01 margin below the measured paraphrase floor, and it cannot bridge hand-authored, provision-dense ground truth and the Live LLM's verbose findings. Statement similarity therefore leaves the scoring path entirely. The eval reports three separate components, never blended: (1) provision coverage — deterministic set F1 over citation targets, strength-weighted on the expected side; (2) summary fidelity — a strict rubric judge comparing provision-aligned Provision relevance (same role, same obligation direction; contradiction forces the floor score); (3) strength agreement under the max-rule Citation strength, at small weight. Produced-vs-expected statements survive only as diagnostic material for the audit. This retires the semantic-matcher scoring path recorded in ADR-0001's updates.

## Consequences

- Ground truth grows: each case's cited provisions need a hand-authored relevance summary (operator work, #7 precedent — the coding agent never writes them).
- Precision over provisions is pessimistic by construction: a produced citation outside the hand-authored expected set is not necessarily wrong; the audit reviews the spurious list before numbers are quoted.
- The eval gains a second LLM (the judge, same configured provider) and with it run-to-run variance; the strict rubric and schema-validated verdicts bound it.
