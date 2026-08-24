# Answer depth outranks latency, and the Evidence pool scales with Research targets

Issue #23 showed the Live workflow answering the canonical Scenario correctly at regime level while missing every operational deployer obligation: the fixed 8-Chunk Evidence pool filled before AI Act Articles 26/27/86 could win seats, and a provision that never reaches the pool can never become a Finding. We decided answer depth is part of the product's contract — users are never asked to phrase the optimal question — and since Regula is permanently local-only (ADR-0002), extra prompt tokens and slower runs buy nothing but better answers, so depth outranks latency wherever they conflict. Rather than raising the constant, the Evidence pool derives from the plan (`pool = SEATS_PER_TARGET × len(plan.targets)`): seat demand scales with the Planner's decomposition, breadth arriving as more Research targets and depth as finer-grained ones, so sharpening queries automatically widens the pool — while a flat budget would break again the moment the corpus grows past three Regulations.

## Consequences

- Implementation waits for scoring to exist (#12/#13); until then this is a recorded pre-commitment on #23, per ADR-0001's no-tuning-before-evals rule.
- Per-target retrieval depth and pool size are separate concerns from here on; one constant playing both roles is how #23 happened.
- Dilution is the accepted risk of larger pools — weaker tail material reaching drafting — and the eval harness's spurious-Finding subtraction is its designed guard; do not remove it when scores dip.
