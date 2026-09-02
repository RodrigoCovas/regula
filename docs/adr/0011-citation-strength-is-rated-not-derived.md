# Citation strength is rated, not derived

Strength agreement (the third component of ADR-0010) compared the expected side's operator per-provision ratings against a produced side that *derived* Citation strength from Finding strength through the max-rule. Those are different axes — how load-bearing a provision is for the scenario versus how directly evidence supports one claim — so the component measured the model's scale-usage habits rather than substance (2026-08-31 GLM-5.3-flash run: mean 0.42, with one scenario labelling 15 of 16 statements strong). Citation strength is now the Summarizer's *rated* centrality of each cited provision: strong when the provision directly imposes or decides the obligations the Answer turns on, moderate when it is a supporting duty or factor, weak when it is definitional or framing — judged only from the content of the Findings citing it, through the same grounding gate as Provision relevance. The max-rule derivation is retired from the produced side and from the Answer's citation badges: users and the eval now see the same rating, so the eval measures what the product shows. A provision whose rating is missing or invalid keeps its relevance but carries no strength — agreement is computed over rated provisions only, never defaulted. Finding strength keeps its own roles: finding display, action anchoring, answer ordering.

## Considered options

- **Retire the metric entirely** — rejected: provision-level centrality is product-useful, and the expected side's operator ratings already exist; only the produced half was broken.
- **A new parallel term ("Provision centrality") alongside the derived Citation strength** — rejected: a third strength-family term invites drift; the glossary redefinition keeps two terms, not three.
- **Have the Verifier rate provisions** — rejected: claim-evidence support is the Verifier's axis; answer-wide provision centrality is exactly the Summarizer's job, and it already writes per-provision statements.

## Consequences

- The Summarizer's schema gains an optional rating; a partial Summarizer pass degrades agreement silently (fewer rated provisions) rather than failing the run.
- Expected-side coverage weights are unchanged — they still read the operator's ratings through the max-rule on the expected side only; the shared max-rule now serves one side, not both.
