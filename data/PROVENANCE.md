# Corpus Provenance

Provenance record for the curated corpus in `data/regulations/`, kept once at
the corpus level per locked decision Q8 (issue #1).

## Source documents

| Document | File | Full name | OJ reference | Source URL |
| --- | --- | --- | --- | --- |
| EU AI Act | `ai-act.json` | Regulation (EU) 2024/1689 — Artificial Intelligence Act | OJ L 2024/1689, 12.7.2024 | https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=OJ:L_202401689 |
| GDPR | `gdpr.json` | Regulation (EU) 2016/679 — General Data Protection Regulation | OJ L 119, 4.5.2016, p. 1 | https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32016R0679 |
| DORA | `dora.json` | Regulation (EU) 2022/2554 — Digital Operational Resilience Act | OJ L 333, 27.12.2022, p. 1 | https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32022R2554 |

Each document's metadata also declares its **perimeter provisions** (#86) —
the provisions that decide who the Regulation covers (DORA Article 2; GDPR
Articles 2, 3, and 4; AI Act Articles 2 and 6) — the curated table the Live
workflow's engagement gate reads; the gate's ADR records the policy.

## Extraction

- **Extraction tool:** lexplorer-generated JSON (locked decision Q8 keeps this format for the MVP).
- **Extraction date:** 19/08/2026. The corpus was subsequently verified against EUR-Lex on 2026-08-20 during the demo-scenario research pass (`research/DEMO_SCENARIO_FINDINGS.md`); treat that as the last-verified date.
- **Language:** English-only. Questions in other languages are answered in English from this corpus.

## Known gaps

- **AI Act recital text is absent.** All 180 recitals carry numbers and per-article references but no wording; recitals are cited by number only.
- **GDPR recital text is absent.** Same as above for all 173 recitals.
- **DORA carries full recital text** for all 106 recitals — the only document whose recitals can be quoted.
- **AI Act annex numbering is shifted by one** relative to the official text: the JSON `annexes` array is indexed 0–12, so `annexes[2]` has id `annex-3` and is titled "High-Risk AI Systems Referred to in Article 6(2)" — the official Annex III. Citations use official numbering and resolve via the corpus id.
- **Spain-specific national requirements are outside the corpus entirely** (e.g. Spanish Credit Agreements law, AEPD guidance).

## Evaluation ground truth

`citations.json` is not corpus: it is the maintainer's hand-authored record of
the Live eval's expected citations (issue #51). Each scenario entry lists the
provisions an answer should cite, each with the relevance summary the
summary-fidelity judge compares against and the operator's per-provision
Strength rating (#56). It is independent of pipeline output
(#7 precedent): every relevance string and Strength is transcribed into
`backend/src/eval_harness.py` text-verbatim — the pinning test
(`test_live_eval.py::test_shipped_ground_truth_transcribes_the_operators_citations_file`)
compares each case's contents against this file regardless of order, so any
drift fails loudly.

The Findings grouping those citations are hand-authored once in the harness
under the same #7 precedent — never derived from or validated against pipeline
output, so the eval can disagree with the code. They carry statements only:
since #58 the operator's per-provision Strength ratings are the single source
of expected Citation strength, riding the Citations alone. Authoring
conventions:

- The relevance standard is one written rule, the **permissive materiality
  rubric** (the #94 curation standard, resolved on the T1 ledger (#95) and
  approved by the operator without corrections on 2026-09-10): a
  Finding/Citation stays expected wherever a fair argument exists for it —
  a provision earns expectation where it decides a duty the Scenario's
  facts put in play (the duty itself, the duty-bearer, the duty-scope, or
  the duty set conditionally for genuinely open conditions) or carries a
  live engagement-threshold question of whether the Regulation reaches the
  Scenario at all (open entity-status questions, exclusions, regime
  allocation across the named Regulations); decoration is spurious. The
  delete-it test decides the border: if the provision were deleted from the
  expected set, would an otherwise-complete answer be missing a duty it
  must state or an engagement-threshold question it must carry? No →
  decoration → spurious. "Live" is load-bearing: where the facts settle a
  Regulation's reach outright and the citation does no allocation work, the
  scope recital is decoration — recruitment/employee gdpr Art 2/3 stayed
  deleted from the #85 pass, while the insurer's and investment firm's
  dora Art 2 stays (their financial-entity status does allocation work the
  scenario asks about). The lineage: the stricter **two-job rule** (the #85
  curation standard, distilled by the run analysis, holdout-checked on #81,
  approved by the operator on #79) curated the ten keys first, and the
  permissive rubric only loosens — the #94 curation pass (#96) applied it
  across all ten keys, adding the operator's 36 approved entries and
  removing nothing, so the same provision is no longer expected in one
  Scenario and penalized in another. Every future key edit is proposed
  against this rubric as a per-item diff the operator can veto, and the
  edited cases' definition hashes change, so old run checkpoints refuse and
  the next run starts under a fresh run id (ADR-0013).
- Related articles cluster into one expectation (one Finding statement);
  ranges expand into their separate Article targets; sub-references
  like "point 5(b)" collapse to the Annex they refine — the structural targets
  coverage compares.
- Each provision is cited by exactly one Finding per scenario: its relevance
  summary is authored once, matching the one-summary-per-provision shape
  `expected_relevance_summaries` enforces.
- The per-provision Strength rating on each citations-record entry (and its
  verbatim transcription on the expected Citation) follows CONTEXT.md:
  strong when a provision names the situation outright, moderate where the
  claim is derived or contingent on facts the Corpus cannot settle (entity
  status, designation), weak where provisions only supply framing. These
  ratings are the expected half of the one strength-bearing score —
  coverage recall.
