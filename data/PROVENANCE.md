# Corpus Provenance

Provenance record for the curated corpus in `data/regulations/`, kept once at
the corpus level per locked decision Q8 (issue #1).

## Source documents

| Document | File | Full name | OJ reference | Source URL |
| --- | --- | --- | --- | --- |
| EU AI Act | `ai-act.json` | Regulation (EU) 2024/1689 — Artificial Intelligence Act | OJ L 2024/1689, 12.7.2024 | https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=OJ:L_202401689 |
| GDPR | `gdpr.json` | Regulation (EU) 2016/679 — General Data Protection Regulation | OJ L 119, 4.5.2016, p. 1 | https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32016R0679 |
| DORA | `dora.json` | Regulation (EU) 2022/2554 — Digital Operational Resilience Act | OJ L 333, 27.12.2022, p. 1 | https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32022R2554 |

## Extraction

- **Extraction tool:** lexplorer-generated JSON (locked decision Q8 keeps this format for the MVP).
- **Extraction date:** not recorded in the JSON files themselves. The corpus was verified against EUR-Lex on 2026-08-20 during the demo-scenario research pass (`research/DEMO_SCENARIO_FINDINGS.md`); treat that as the last-verified date.
- **Language:** English-only. Questions in other languages are answered in English from this corpus.

## Known gaps

- **AI Act recital text is absent.** All 180 recitals carry numbers and per-article references but no wording; recitals are cited by number only.
- **GDPR recital text is absent.** Same as above for all 173 recitals.
- **DORA carries full recital text** for all 106 recitals — the only document whose recitals can be quoted.
- **AI Act annex numbering is shifted by one** relative to the official text: the JSON `annexes` array is indexed 0–12, so `annexes[2]` has id `annex-3` and is titled "High-Risk AI Systems Referred to in Article 6(2)" — the official Annex III. Citations use official numbering and resolve via the corpus id.
- **Spain-specific national requirements are outside the corpus entirely** (e.g. Spanish Credit Agreements law, AEPD guidance).
