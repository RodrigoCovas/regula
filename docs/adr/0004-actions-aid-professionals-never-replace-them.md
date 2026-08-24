# Actions aid legal professionals; they never replace their judgment

Issue #24 found Live mode returning `actions: []` on every successful answer while the demo curated six next-steps, even though the glossary's Answer promises Findings, Citations, and Actions. We decided Actions remain part of every Answer but are canonically referral-voiced: each names something only a professional can settle — a contingency the Corpus cannot determine, or verification of cited provisions against the company's actual situation — never a compliance directive, never a presumption that a Regulation applies. Regula's purpose is to help users find relevant regulations so qualified professionals can act on them; a directive Action would be the tool giving legal advice. Live Actions come from a conservative LLM pass over kept Findings, grounded so every emitted Action resolves to a Finding's citations (rejections logged in the detailed trace), plus a standing seek-counsel hand-off so the sheet is never empty.

## Considered and rejected

- Directive compliance tasks (the demo's original "run a DPIA / design human oversight" items): they presume applicability before any human has settled scope.
- Deterministic templates instead of an LLM pass: honest by construction, but the phrasing doesn't scale as Regulations and Scenarios multiply.
- Seeding from discarded Unsupported claims: Verdicts don't distinguish corpus-silent from contradicted (`supported: bool` only), so referrals could legitimize ideas the evidence already undercut; moderate Findings already carry every surviving contingency, with citations attached.

## Consequences

- Minimum anchor strength for a seeded Action is moderate; weak Findings may enrich an anchored Action but never stand alone.
- The not-legal-advice boundary moves out of `DEMO_ACTIONS` into `known_limitations` — a Known limitation is never an Action — and the demo's remaining Actions are rewritten to referral voice.
- Live Citations populate `source_short_name` from corpus metadata (folded into #24).
