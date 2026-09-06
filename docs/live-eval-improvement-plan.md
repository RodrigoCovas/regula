# Live-eval improvement plan

Reference guide for the fix backlog distilled from the analysis of run
`glm53-v7-2026-09-04` (Live eval, model `z-ai/glm-5.3-flash`, artifact
`logs/live-eval-report-2026-09-04-glm53_v7.json`). The artifact holds 10
cases; the analysis projection below is for the 8 the operator quoted
(both extra cases share the same pathology, so the plan covers them too).

## How the metrics work (read first)

`coverage_scores` (`backend/src/eval_harness.py`) is set arithmetic over
Citation targets — structural provisions like `gdpr Article 4`, never
statements:

- **Recall** = strength-weighted share of expected targets covered
  (strong = 50, moderate = 5, weak = 1). A missed weak target is nearly
  free; a missed strong target costs 50.
- **Precision** = fraction of unique produced targets hitting any expected
  target. Unweighted: one spurious weak citation costs as much as a
  spurious core one.
- **Summary fidelity** = the rubric judge over provision-aligned relevance
  summaries (`eval_judge.py`); a cited provision the Summarizer left bare
  floors that pair to 0; judge scores are {0, 0.5, 1}.

Consequence: the highest-leverage changes are subtractive (stop citing
what the scenario's facts do not engage); weak-strength recall misses are
cheap to leave, strong/moderate misses are not.

Run baseline: mean P 0.742, mean R 0.908, mean F1 0.797, mean fidelity
0.960. Full-stack ceiling if everything below lands perfectly: F1 ≈ 0.995;
realistic target 0.90–0.94.

Before any change: regenerate the per-case spurious/missed accounting from
a run artifact (compare produced citation targets against expected labels
with the weight map) so every fix is argued from the same ledger, not from
reading answers.

---

## Fix 1 — Engagement/contingency gate (both directions)

**Problem.** Limbs are produced whose conditions the scenario facts settle
the other way (retailer: conditional DORA limbs; ransomware: AI Act Art 73
with no AI system in the scenario; telecom: high-risk limbs for a
transparency-tier chatbot), while genuinely open conditions get dropped
(fintech: the whole conditional DORA block). The policy is currently
inverted in both directions.

**Two parts — both required.**

1. *Deterministic gate in application code* (`live_workflow.py`, in/after
   `_decide_claims`, recorded per claim in the detailed trace like every
   other rejection). Keep the repo's rule: enforced by application code,
   never trusted to the LLM.
   - Per regime R cited by a kept finding, compute the engagement state
     from the kept set: R is **open** if some kept finding cites R's
     perimeter/threshold provision (`gdpr` Art 2/3/4, `ai-act` Art 2/6,
     `dora` Art 2) as applying or as an open question; **closed** if a
     finding cites it as not reaching the scenario.
   - A duty-area finding scoped only to R is dropped (rejected, with
     reason) unless R is open. When R is closed, the perimeter citation
     must instead surface as the exclusion Finding — that is the expected
     form (retailer expected cites dora Art 2 for exactly this).
   - Key design point: this single rule fixes retailer + ransomware +
     telecom *and* recovers fintech's conditional DORA block, because the
     fintech facts leave DORA status open. Do not hard-code cases.
2. *Verifier prompt change* (`_VERIFIER_SYSTEM`): add a worked example for
   within-regime classification next to the existing financial-entity one:
   the scenario's described use case either sits in an Annex III category
   or it does not — develop the high-risk regime only when it does; a
   "could be high-risk if it did X" limb is unsupported when the scenario
   never describes X. Also state the inverse: a contingency the text
   genuinely leaves open (entity licensing status) stays a moderate
   conditional finding.

**Watch out.** The gate depends on anchors being cited (Fix 2) — without
them it over-deletes legitimate duty claims. Land with Fix 2 or after it.
Fintech is the regression tripwire: its conditional DORA block must
survive; write a test that pins this.

**Expected impact.** +0.05 mean F1 (largest single principled lever).
Fixes the three worst cases (telecom P 0.361, retailer P 0.538,
ransomware P 0.737).

---

## Fix 2 — Planner reserved anchors actually land

**Problem.** The Planner already reserves "engagement-threshold targets"
(definitions, perimeters, principles) but they fall out of the answer:
gdpr Art 5 missed in employee-productivity (weight 50 — 68% of that
case's entire recall gap), dora Art 2 missed in bank (moderate, weight 5),
gdpr Art 4 missed in 4 cases, ai-act Art 2/3 missed in 3.

**Changes.**

- `_RESEARCHER_SYSTEM`: instruct that when the evidence pool holds a
  regime's engagement/definitional provision (gdpr Art 2/3/4, gdpr Art 5
  principles, ai-act Art 2/3, dora Art 2) it must draft the
  applicability/engagement claim citing it — engagement claims are never
  optional filler.
- `_PLANNER_SYSTEM`: the reserved-targets-first rule exists; verify it
  survives truncation and that retrieval returns those chunks (check the
  inventory vocabulary the Planner lifts keywords from: "material scope",
  "territorial scope", "principles relating to processing"). If
  definitions are crowded out of the pool, revisit fair-share seats for
  reserved targets.
- Optional deterministic backstop: if a reserved target's provision is
  never cited by any kept finding, record it in the Execution trace (and
  optionally one corrective re-prompt, mirroring the plan-truncation
  correction pattern).

**Expected impact.** +0.036 mean F1 (and it nudges precision up ~+0.03,
because the added targets are expected).

---

## Fix 3 — AI-Act markers/auxiliary block

**Problem.** Expected in every high-risk AI scenario, produced zero times
across the run: AI literacy (ai-act Art 4 — expected in all 5 AI
scenarios), provider-side duties (Art 9/10/11/12/15/16), conformity
assessment + EU-database registration (Art 43/49), post-market monitoring
(72/73), role-flip (Art 25), timing (113), fines framing (99).

**Changes.**

- `_PLANNER_SYSTEM`: one new duty-area rule — per confirmed high-risk AI
  regime, one target for the provider-side obligations and compliance
  markers (Art 9–16, 43, 49) and one for post-market monitoring and
  serious-incident reporting (72/73); plus AI literacy (Art 4) and the
  application date (113) as standing targets.
- `_RESEARCHER_SYSTEM`: when the company is a deployer of a
  provider-supplied high-risk system, draft the vendor-verification
  claims ("verify the provider's conformity assessment, CE marking,
  registration") — that is how the ground truth words these findings.

**Expected impact.** +0.022 mean F1. All weak except Art 10 in
fintech/insurance (moderate) — cheap, systematic.

---

## Fix 4 — GDPR completeness block

**Problem.** Mid-weight duties hit in some cases, dropped in others:
gdpr Art 15 (access right) missed in 4 cases, Art 30 (records) in 3,
Art 25/32 (DPbDD + security) in employee and fintech (moderate each),
Art 24 (accountability) in ransomware, Art 55/56/77 (which authority,
complaints) in retailer, Art 44 (transfers) in telecom, ai-act Art 86
(explanation right) missed outright in insurance.

**Changes.**

- `_PLANNER_SYSTEM`: per confirmed GDPR engagement, standing duty-area
  targets for (a) information & access rights (13/14/15/12), (b) records
  (30), (c) DPbDD + security (25/32), (d) accountability (24); for breach
  scenarios, notification-authority detail (55/56) and complaint
  exposure (77); transfer analysis (44) when inference/storage may sit
  outside the EEA.
- These findings inherit the Fix 1 gate: a duty-area finding survives
  only while its regime is open, so the added breadth cannot reintroduce
  the conditional-limb noise.

**Expected impact.** +0.032 mean F1.

---

## Fix 5 — DORA reporting machinery

**Problem.** Template/channel/feedback provisions (dora Art 20/21/22),
post-incident review (13), communication strategy (14), the
simplified-framework check (16) and proportionality (4) are expected in
bank and ransomware (and 16 in fintech) and missed everywhere.

**Changes.** One `_PLANNER_SYSTEM` duty-area rule for confirmed DORA:
reporting machinery (20/21/22), post-incident review and communication
strategy (13/14), the size/status check (16), proportionality (4).

**Expected impact.** +0.013 mean F1 (all weak). Do it because it is one
rule, not for the number.

---

## Fix 6 — Citation discipline (the blunt extension)

**Problem.** 47 unique off-target citations across the run, of which the
gate removes ~20 (conditions not raised) and the remainder are
non-decisive: cross-reference embellishments (dora 46), auxiliary
elaborations (employee gdpr 38/39, insurance gdpr 7/12, bank dora
5/8/24/31/42, ransomware dora 8/10/24/46), generic plumbing (telecom's
16-item GDPR catalogue).

**Changes.**

- `_VERIFIER_SYSTEM` / `_RESEARCHER_SYSTEM`: cite and keep only the
  provisions that *decide* a claim; framing-only references ride only
  when the claim is about the framing (definitions, perimeter,
  principles).
- Option B if prompts under-deliver: a deterministic per-finding
  citation cap/dedup in application code — never drop the last
  moderate/strong anchor of a finding.

**Hard dependency.** Do the curation pass (Fix 7) first: several of the
removals this fix targets are only "spurious" under case-by-case
curated judgment (telecom's gdpr Art 2/3 territorial citations, dora
Art 5 in bank vs fintech, gdpr Art 7 in/out). Tuning against
inconsistent ground truth means tuning twice.

**Expected impact.** +0.09 mean F1 *on top of* Fix 1 (combined +0.15);
the single biggest number in the backlog, and the one that changes
user-visible answer shape — deserves its own A/B'd iteration.

---

## Fix 7 — Summarizer coverage + fidelity observability

**Problem.** Insurance's DPO finding shipped citations for gdpr Art 37/39
with `relevance: null, strength: null` — partial Summarizer coverage —
flooring 2 fidelity pairs deterministically (that is the whole 0.867).
Telecom and ransomware each had one judged pair score 0.5 whose identity
is invisible: per-pair verdicts and Known-limitation text are not
persisted.

**Changes.**

- `live_workflow.py` summarizer node: on partial coverage (some cited
  provisions left bare), one corrective re-prompt listing the uncovered
  refs, mirroring the plan-truncation correction pattern; keep the
  degrade-never-fail behaviour as the fallback.
- Artifact/dump (`live_eval.py`, `eval_harness.py`):
  `EvalScenarioResult` gains the per-pair fidelity verdicts (ref,
  same_role, same_direction, contradiction, score) and the Known
  limitation text; the checkpoint shape (`_RESULT_KEYS`) and its tests
  change accordingly.
- Root cause (already established from the run artifact — the query log is
  telemetry-only, there is nothing to dig for there): the insurance case's
  gdpr Art 37 (×2) and Art 39 refs were emitted, resolved, and quoted, and
  left unrated — a Summarizer coverage-check failure, not unknown or
  duplicated refs. The retry targets the coverage check.

**Expected impact.** Fidelity only: mean 0.960 → ~0.99. Zero coverage
impact; do it any time, it is independent.

---

## Fix 8 — Curation pass (ground truth consistency)

**Problem.** The relevance standard varies case to case, which caps
attainable precision and makes prompt tuning whack-a-mole:

- gdpr Art 2/3 (scope) expected in recruitment/employee, absent from
  telecom — where the produced answer cites them and gets penalized.
- dora Art 5 (management body): expected weak in fintech/ransomware,
  spurious in bank.
- gdpr Art 7 (consent conditions): expected in recruitment/fintech,
  spurious in insurance.
- gdpr Art 27 (Art 3(2) representative limb): a correct elaboration
  penalized in recruitment.

**Changes.**

- Ground truth is `data/regulations/citations.json` (operator record)
  transcribed verbatim into `eval_harness.py`. Edit both, or the pinning
  test
  (`test_live_eval.py::test_shipped_ground_truth_transcribes_the_operators_citations_file`)
  fails loudly. Record the resolved rubric in `data/PROVENANCE.md`
  (the authoring conventions section).
- Editing any case changes its definition hash: existing checkpoints go
  stale and the run refuses — start a new `--run-id` per the ADR-0013
  rules.
- Note: the fintech conditional DORA block (16 weak citations) is *not*
  a problem — producing it raises that case's precision (14/15 → 30/31)
  and recall; it is pipeline-fixable via Fix 1's "open conditions" path.

**Expected impact.** Removes curation-induced noise so Fix 6's tuning
sticks; no pipeline delta by itself.

---

## Measurement protocol

- Run: `python -m backend.src.live_eval --run-id <id> --output <path>`
  (or inside the stack). Checkpoints refuse on model change; cases are
  hashed — editing a case or ground truth invalidates old checkpoints.
- The artifact now has 10 cases (live-genai-customer-service,
  live-ai-trading-cloud-attack were added after the quoted run): do not
  compare means across runs without noting the case-list change.
- Checks before committing: `python -m pytest backend/tests/ -q` and
  `mypy` (config in `pyproject.toml`). The workflow's deterministic
  pieces (gate, anchor backstop, summarizer retry) get scripted-fake
  tests in `backend/tests/test_live_pipeline.py`; the artifact changes
  touch `test_live_eval.py` / `test_eval_harness.py`.
- Prompt-only fixes are measurable only in Live mode — keep
  per-fix `--run-id`s even when bundling, and re-derive the spurious/
  missed ledger after each run so regressions name their targets.

## Impact summary (modeled on glm53-v7, perfect execution)

| Fix | ΔmeanP | ΔmeanR | ΔmeanF1 | Notes |
| --- | ------ | ------ | ------- | ----- |
| 1 gate (both directions) | +0.06 | +0.005 | +0.05 | includes fintech DORA recovery |
| 2 anchors | +0.03 | +0.036 | +0.036 | employee gdpr 5 [50] |
| 3 AI-Act markers | +0.02 | +0.018 | +0.022 | |
| 4 GDPR completeness | +0.03 | +0.021 | +0.032 | |
| 5 DORA machinery | +0.02 | +0.007 | +0.013 | |
| 6 citation discipline | +0.15 | 0 | +0.09 (on top of 1) | needs Fix 8 first |
| 7 summarizer/observability | — | — | fidelity +0.03 | |
| 8 curation pass | — | — | enables Fix 6 | |

Cumulative ceiling ≈ 0.995 mean F1; realistic 0.90–0.94. Priorities 1+2+6
are subtractive and carry roughly three quarters of the total gain.
