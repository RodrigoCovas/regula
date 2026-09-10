# The Planner budget rises to fifty Research targets

The declared 1-8 budget (ADR-0015) still caps recall: expected provision sets span 11–45 while eight targets' seats bound what retrieval can surface, and the operator judges the constraint the main live-performance bottleneck. The declared budget rises to 50 (`MAX_RESEARCH_TARGETS`), keeping ADR-0015's structure untouched — reserved engagement-threshold targets stay first, the regime-coverage rule stays, and a provider response over the limit is corrected once by positional truncation, with the Evidence pool still deriving from the accepted plan (ADR-0003), so a full plan seats `SEATS_PER_TARGET × 50`. The operator chose the open-ended ceiling deliberately: plans may grow to the demand the corpus actually imposes rather than a guessed number.

## Consequences

- The pinned declared-budget phrase, the truncation-correction test, and the pool-bound test re-pin to 50 (and 51 over-limit) so budget and rubric drift keep failing the suite; nothing in the retrieval, Proposer, Summarizer, scoring, or Demo-mode paths moves.
- The Evidence pool scales with the plan as before (ADR-0003), so no target is starved at the raised ceiling; a full plan's pool is 600 Chunks. Because seats and budget multiply, per-target seats (`SEATS_PER_TARGET`) are the intended trim lever if agent prompts strain context or latency — pool evidence from the 2026-09-09 run shows per-target retrieval supply averaging ~9–11 unique Chunks at the current leg depths, so seat shares can fall without starving targets.
- Lowering the budget takes a new ADR, not a code change.

Parent of the raise: ADR-0015.