# Baselines are agent-produced answers scored by the shared eval harness

The Baseline comparison measures what the workflow's retrieval-and-verification machinery adds over answering without it. Baseline answers are produced outside the system — one structured completion per case, generated in the OpenCode agent harness from locked prompt templates and stored as JSON under `logs/baseline-runs/<variant>/` — and are scored by exactly the harness a pipeline run goes through: the ADR-0010 components over the same ground truth, with the same fidelity judge. The loader maps each baseline file into the response shape the harness consumes and validates it strictly: a record that does not match the stored template shape refuses the run, and nothing is normalized inside the pipeline — malformed files are fixed by hand before the comparison runs. Two domain rules are the recorded exceptions: duplicate summaries for the same structural target resolve first-wins (the pipeline's own duplicate gate — a repeat is rejected, the first stands), and a citation whose provision label does not parse is dropped and recorded rather than voiding the run (mirroring the pipeline's handling of labels that match no Chunk).

## Considered options

- **Toolless re-runs first** — rejected for the first comparison. The current runs were produced by agents with tools such as WebFetch available, which is a threat to validity: a Parametric baseline with web access is not strictly parametric. The runs are kept and scored anyway, the threat is recorded with the artifact, and toolless runs (an agent with every tool denied, or raw API calls) remain the planned follow-up before any external claim.
- **Token accounting** — dropped. The comparison reports performance metrics only (provision coverage, summary fidelity): the baselines moved from scripted API calls to agent runs, where per-call token usage is not attributable to the answer alone.
- **Scoring the baselines with separate logic** — rejected. Reusing `evaluate_scenarios` (same coverage scorer, same judge) is the whole point: the answering path becomes the only differing variable.

## Consequences

- The fidelity judge is eval overhead — one batched call per case per variant — and is never counted as baseline cost.
- Precision counts every off-target produced citation against the baseline, so a baseline that cites invented provisions scores its true quality rather than zero.
- The Baseline directory names carry the canonical spellings (`parametric`, `full-corpus`); the misspelled `parametrized/` was renamed before any artifact referenced it.
