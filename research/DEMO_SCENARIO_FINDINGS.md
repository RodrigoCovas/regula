# Demo Scenario — Evidence-Backed Findings

## Scenario and question

> **Regulatory question:** "We are building an AI system that evaluates loan applications in Spain. What EU regulations could apply and what should we investigate before deployment?"

**Corpus reviewed:** the three source documents in `data/regulations/`:

- `ai-act.json` — Regulation (EU) 2024/1689 (EU AI Act), applies from 2 August 2026
- `gdpr.json` — Regulation (EU) 2016/679 (GDPR), applies since 25 May 2018
- `dora.json` — Regulation (EU) 2022/2554 (DORA), applies since 17 January 2025

**Corpus caveats (read before citing):**

- The AI Act and GDPR JSON files contain article and recital *numbers* and per-article recital references, but **no recital text**. Only the DORA JSON carries recital text. Recital numbers below are therefore cited where an article references them, but their wording cannot be quoted from this corpus.
- The AI Act `annexes` array is indexed 0–12 and the annex *numbers* are shifted by one vs. the official text: `annexes[2]` has id `annex-3` and is titled **"High-Risk AI Systems Referred to in Article 6(2)"** — this is the official **Annex III** list. All citations below use the official numbering and note the corpus id.
- GDPR Article 22 **is present** in the corpus (automated individual decision-making), as is AI Act Article 6 and the Annex III high-risk list.

---

## Findings

### Finding 1 — Creditworthiness-evaluation AI is a "high-risk" AI system under the EU AI Act

**Claim.** An AI system that evaluates the creditworthiness of natural persons or establishes their credit score (outside pure fraud detection) is a high-risk AI system under the AI Act, so the full high-risk regime applies.

**Citations.**

- `ai-act` Article 6(2): "In addition to the high-risk AI systems referred to in paragraph 1, AI systems referred to in Annex III shall be considered to be high-risk."
- `ai-act` Annex III (corpus id `annex-3`), point 5 ("Access to and enjoyment of essential private services and essential public services and benefits"), point (b): "AI systems intended to be used to evaluate the creditworthiness of natural persons or establish their credit score, with the exception of AI systems used for the purpose of detecting financial fraud."
- `ai-act` Article 6(3): "Notwithstanding the first subparagraph, an AI system referred to in Annex III shall always be considered to be high-risk where the AI system performs profiling of natural persons."
- `ai-act` Article 6(4): "A provider who considers that an AI system referred to in Annex III is not high-risk shall document its assessment before that system is placed on the market or put into service."
- `ai-act` Article 6 recital refs: recitals 46–48 and 50–63 (text not in corpus).

**Evidence.** Article 6(2) incorporates the whole of Annex III into the high-risk category, and Annex III point 5(b) names creditworthiness evaluation / credit scoring directly, with an express carve-out only for financial-fraud detection. Article 6(3) adds that the "narrow procedural task" derogation cannot rescue a system that performs profiling of natural persons — which credit scoring typically does (see GDPR Art 4(4)). Article 6(4) still requires a documented, registerable assessment if the provider claims the system is *not* high-risk. This is in force from 2 August 2026 (Article 6(2) is not deferred; only Article 6(1)'s product-based route is, via Article 113(3)(c)).

**Strength: STRONG.** Direct, binding classification rule naming the exact use case.

---

### Finding 2 — Provider obligations for the high-risk system

**Claim.** The provider of the creditworthiness AI must comply with the full high-risk requirements: risk-management system, data governance and bias measures, transparency/instructions for use, human oversight, accuracy/robustness/cybersecurity — plus conformity assessment, CE marking and registration in the EU database.

**Citations.**

- `ai-act` Article 16 (introductory and points (a), (f), (h), (i)): "Providers of high-risk AI systems shall: (a) ensure that their high-risk AI systems are compliant with the requirements set out in Section 2; … (f) ensure that the high-risk AI system undergoes the relevant conformity assessment procedure as referred to in Article 43…; (h) affix the CE marking…; (i) comply with the registration obligations referred to in Article 49(1)."
- `ai-act` Article 9(2) (risk management): "a continuous iterative process planned and run throughout the entire lifecycle of a high-risk AI system… (a) the identification and analysis of the known and the reasonably foreseeable risks that the high-risk AI system can pose to health, safety or fundamental rights…"
- `ai-act` Article 10(2)–(3) (data governance): data sets "shall be relevant, sufficiently representative, and to the best extent possible, free of errors and complete in view of the intended purpose"; Article 10(2)(f) requires examining "possible biases that are likely to… lead to discrimination prohibited under Union law."
- `ai-act` Article 13(1)–(2) (transparency / instructions for use).
- `ai-act` Article 14(1) (human oversight: systems "can be effectively overseen by natural persons during the period in which they are in use").
- `ai-act` Article 15(1) (accuracy, robustness, cybersecurity).
- `ai-act` Article 72 (post-market monitoring) and Article 71 (EU database for Annex III systems); recital refs for Art 13 (47, 72), Art 14 (48–49, 73–74), Art 15 (50, 74–75).
- `ai-act` Article 99(4): non-compliance with provider/deployer obligations can draw fines "up to 15 000 000 EUR or… 3 % of its total worldwide annual turnover" (recital refs 142–146).

**Evidence.** Chapter III Section 2 (Articles 8–15) lays down the requirements; Article 16 makes the provider legally responsible for them and links them to conformity assessment (Art 43), the EU declaration of conformity (Art 47) and CE marking (Art 48). Registration in the EU database (Arts 49/71) is publicly visible. Sanctions for breach follow from Article 99(4).

**Strength: STRONG.** Binding obligations that attach automatically once the system is high-risk under Finding 1.

---

### Finding 3 — Deployer obligations for the high-risk system

**Claim.** As the deployer (or customer-deployer), the company must assign human oversight, monitor the system, keep automatically generated logs for at least six months, ensure input data is relevant and representative, and inform applicants that they are subject to a high-risk AI system.

**Citations.**

- `ai-act` Article 26(2): "Deployers shall assign human oversight to natural persons who have the necessary competence, training and authority, as well as the necessary support."
- `ai-act` Article 26(4): "to the extent the deployer exercises control over the input data, that deployer shall ensure that input data is relevant and sufficiently representative in view of the intended purpose."
- `ai-act` Article 26(6): deployers "shall keep the logs automatically generated by that high-risk AI system… for a period appropriate to the intended purpose… of at least six months."
- `ai-act` Article 26(11): "deployers of high-risk AI systems referred to in Annex III that make decisions or assist in making decisions related to natural persons shall inform the natural persons that they are subject to the use of the high-risk AI system."
- `ai-act` Article 26 recital refs: recitals 82–95 (text not in corpus).

**Evidence.** Article 26 converts the design-time human-oversight measures (Art 14) into operational deployer duties and adds a transparency duty toward the affected applicants themselves. It also references data-protection interplay: Article 26(9) tells deployers to use the Article 13 information to carry out a GDPR Article 35 DPIA, and Articles 26(5)–(6) recognise that financial institutions subject to Union financial-services governance rules (i.e., DORA-covered entities) may satisfy some of these duties through those frameworks.

**Strength: STRONG.** Direct obligations on the party that uses the system for loan decisions.

---

### Finding 4 — Mandatory Fundamental Rights Impact Assessment before deployment

**Claim.** Before the first use, the deployer of a high-risk creditworthiness system must perform a Fundamental Rights Impact Assessment (FRIA) and notify its results to the market surveillance authority.

**Citations.**

- `ai-act` Article 27(1): "Prior to deploying a high-risk AI system referred to in Article 6(2)… deployers of high-risk AI systems referred to in points 5 (b) and (c) of Annex III, shall perform an assessment of the impact on fundamental rights that the use of such system may produce," covering (a)–(f): processes, duration/frequency, affected categories, specific risks of harm, human-oversight implementation, and measures for when risks materialise.
- `ai-act` Article 27(2): the obligation applies "to the first use of the high-risk AI system" and must be kept up to date.
- `ai-act` Article 27(3): "the deployer shall notify the market surveillance authority of its results, submitting the filled-out template…"
- `ai-act` Article 27(4): where a GDPR Article 35 DPIA already exists, the FRIA "shall complement" it.
- `ai-act` Article 27 recital ref: recital 96 (text not in corpus).

**Evidence.** Annex III point 5(b) deployers are explicitly singled out in Article 27(1) as FRIA-obligated — the strongest possible textual hook for the demo scenario. It is deployer-side and therefore distinct from the provider-side requirements of Finding 2.

**Strength: STRONG.** The Article names the exact class of deployer in the scenario.

---

### Finding 5 — Right to explanation for affected applicants

**Claim.** An applicant subject to a loan decision taken on the basis of the high-risk system's output has a right to a clear, meaningful explanation of the AI's role in the decision and the main elements of the decision.

**Citations.**

- `ai-act` Article 86(1): "Any affected person subject to a decision which is taken by the deployer on the basis of the output from a high-risk AI system listed in Annex III, with the exception of systems listed under point 2 thereof, and which produces legal effects or similarly significantly affects that person… shall have the right to obtain from the deployer clear and meaningful explanations of the role of the AI system in the decision-making procedure and the main elements of the decision taken."
- `ai-act` Article 86(3): "This Article shall apply only to the extent that the right referred to in paragraph 1 is not otherwise provided for under Union law."
- `ai-act` Article 86 recital ref: recital 158 (text not in corpus).

**Evidence.** Loan refusals produce legal effects / significantly affect the applicant, so Article 86(1) is directly engaged. Article 86(3) interacts with the GDPR Article 15/22/13 information rights (Findings G3/G4), so in practice the two regimes must be reconciled rather than treated as independent.

**Strength: STRONG.** Direct right, on point, with a cross-reference to GDPR.

---

### Finding 6 — AI literacy duty

**Claim.** Both providers and deployers must ensure a sufficient level of AI literacy among their staff who operate or use the AI system.

**Citations.**

- `ai-act` Article 4: "Providers and deployers of AI systems shall take measures to ensure, to their best extent, a sufficient level of AI literacy of their staff and other persons dealing with the operation and use of AI systems on their behalf… considering the persons or groups of persons on whom the AI systems are to be used."
- `ai-act` Article 4 recital ref: recital 20 (text not in corpus).

**Evidence.** This duty applies to all AI systems, not only high-risk ones, and binds both provider and deployer — relevant as an immediate, low-cost compliance item.

**Strength: MODERATE.** Binding but general; it does not attach specifically to the creditworthiness use case.

---

### Finding 7 — Transparency toward natural persons interacting with the AI system

**Claim.** If the system (or any interface around it) interacts directly with applicants, the provider must design it so that persons are informed they are interacting with an AI system.

**Citations.**

- `ai-act` Article 50(1): "Providers shall ensure that AI systems intended to interact directly with natural persons are designed and developed in such a way that the natural persons concerned are informed that they are interacting with an AI system, unless this is obvious from the point of view of a natural person who is reasonably well-informed, observant and circumspect…"
- `ai-act` Article 50 recital refs: recitals 117, 132–134 (text not in corpus).

**Evidence.** This is a distinct transparency layer from the Article 26(11) "you are subject to a high-risk system" notice; it applies to any direct interaction (e.g., a chatbot or application form) regardless of high-risk status. Its applicability is factual: does the applicant interact with the AI directly?

**Strength: MODERATE.** Binding, but contingent on the interaction design of the product.

---

### Finding 8 — Definitions (AI system, provider, deployer, profiling) — framing only

**Claim.** The system qualifies as an "AI system"; the company will be a "provider" and/or "deployer"; credit scoring is a form of "profiling" as cross-referenced into the AI Act.

**Citations.**

- `ai-act` Article 3(1) ("AI system"), Article 3(3) ("provider"), Article 3(4) ("deployer"), Article 3(52) ("profiling" = GDPR Art 4(4)). Recital refs for Art 3: 12–19 (text not in corpus).

**Evidence.** Article 3 supplies the vocabulary that the operative articles above depend on; on its own it imposes no obligations and supports no specific duty.

**Strength: WEAK.** Definitions article; load-bearing only in combination with Findings 1–7.

---

### Finding 9 — Credit scoring is not a prohibited practice

**Claim.** The AI Act does not prohibit creditworthiness scoring; the prohibitions in Article 5 target other practices (social scoring, biometric categorisation, emotion inference, criminal-risk prediction, etc.).

**Citations.**

- `ai-act` Article 5(1), in particular (c) social scoring ("evaluation or classification of natural persons… based on their social behaviour or known, inferred or predicted personal or personality characteristics, with the social score leading to… detrimental or unfavourable treatment") and (d) criminal-risk assessment. Recital refs for Art 5: 28–45 (text not in corpus).

**Evidence.** The listed prohibitions are exhaustive; credit scoring as such does not appear, which is why Annex III point 5(b) (Finding 1) is the correct characterisation: the activity is lawful but strictly regulated as high-risk.

**Strength: STRONG** as a scope-clarifying (negative) finding: it blocks a common misconception and is directly verifiable from Article 5.

---

### Finding 10 — Penalties exposure

**Claim.** Breach of the provider/deployer obligations for the high-risk system exposes the company to administrative fines up to €15 million or 3% of worldwide annual turnover.

**Citations.**

- `ai-act` Article 99(4): "Non-compliance of an AI system with any of the following provisions related to operators… shall be subject to administrative fines of up to 15 000 000 EUR or… up to 3 % of its total worldwide annual turnover…: (a) obligations of providers pursuant to Article 16; … (e) obligations of deployers pursuant to Article 26." Recital refs: 142–146.

**Evidence.** Directly links the obligations in Findings 2 and 3 to a quantified sanction ceiling; fines are imposed by Member States (Art 99(1)–(2)).

**Strength: MODERATE.** Directly verifiable, but sanctions are enforced nationally and amount depends on individual circumstances.

---

## GDPR findings

### Finding G1 — Restriction on fully automated decisions

**Claim.** Under GDPR, an applicant must not be subject to a loan decision based solely on automated processing, including profiling, that produces legal effects or similarly significantly affects them.

**Citations.**

- `gdpr` Article 22(1): "The data subject shall have the right not to be subject to a decision based solely on automated processing, including profiling, which produces legal effects concerning him or her or similarly significantly affects him or her."
- `gdpr` Article 22 recital refs: recitals 71, 72, 91 (text not in corpus).

**Evidence.** A loan grant/refusal is a decision with legal or similarly significant effects, and scoring-based evaluation is profiling (GDPR Art 4(4)). Note the provision uses "based solely on automated processing" — so the outcome depends on whether meaningful human involvement is built in; this is exactly the design question to investigate.

**Strength: STRONG.** Direct rule on point.

---

### Finding G2 — Contract-necessity exception and mandatory safeguards

**Claim.** Even where the decision is necessary for entering into/performing the loan contract, the controller must still give the applicant the right to human intervention, to express their point of view, and to contest the decision.

**Citations.**

- `gdpr` Article 22(2)(a): "Paragraph 1 shall not apply if the decision… is necessary for entering into, or performance of, a contract between the data subject and a data controller."
- `gdpr` Article 22(3): "the data controller shall implement suitable measures to safeguard the data subject's rights and freedoms and legitimate interests, at least the right to obtain human intervention on the part of the controller, to express his or her point of view and to contest the decision."
- `gdpr` Article 22(4): such decisions must not be based on special categories of data unless Article 9(2)(a)/(g) applies with suitable safeguards.

**Evidence.** The contract-necessity exception (22(2)(a)) is the natural basis for loan decisions, but it does not waive the safeguards in 22(3). This is the core GDPR compliance design question for the scenario.

**Strength: STRONG.** Direct rule on point.

---

### Finding G3 — Data Protection Impact Assessment (DPIA) is required

**Claim.** The credit-scoring processing requires a DPIA before processing begins, because it is a systematic, extensive automated evaluation of personal aspects on which legally effective decisions are based.

**Citations.**

- `gdpr` Article 35(1): "Where a type of processing in particular using new technologies… is likely to result in a high risk to the rights and freedoms of natural persons, the controller shall, prior to the processing, carry out an assessment of the impact of the envisaged processing operations on the protection of personal data."
- `gdpr` Article 35(3)(a): a DPIA "shall in particular be required in the case of: (a) a systematic and extensive evaluation of personal aspects relating to natural persons which is based on automated processing, including profiling, and on which decisions are based that produce legal effects concerning the natural person or similarly significantly affect the natural person."
- `gdpr` Article 35 recital refs: recitals 75, 84, 89–93 (text not in corpus).

**Evidence.** Loan scoring is the textbook 35(3)(a) case. The AI Act's FRIA (Finding 4) expressly complements the DPIA (AI Act Art 27(4)), so the two assessments should be run together.

**Strength: STRONG.** Direct, named criterion.

---

### Finding G4 — Transparency about automated decision-making

**Claim.** The controller must tell applicants about the automated decision-making / profiling — including meaningful information about the logic involved, significance and consequences — both at collection time and upon access requests.

**Citations.**

- `gdpr` Article 13(2)(f): the controller shall provide "the existence of automated decision-making, including profiling, referred to in Article 22(1) and (4) and, at least in those cases, meaningful information about the logic involved, as well as the significance and the envisaged consequences of such processing for the data subject."
- `gdpr` Article 15(1)(h): identical right "in those cases" upon an access request.
- Recital refs: Art 13 → recitals 60–62; Art 15 → recitals 63–64 (text not in corpus).

**Evidence.** These are the concrete "explainability" obligations that sit alongside the AI Act's Article 86 right to explanation and Article 13/26(11) transparency duties.

**Strength: STRONG.** Direct, named information obligations.

---

### Finding G5 — Lawful basis must be established

**Claim.** Processing loan-application personal data requires a lawful basis under Article 6(1); for a loan decision this will typically be contract necessity, with legitimate interests or consent as alternatives that each carry different conditions.

**Citations.**

- `gdpr` Article 6(1) (points (a)–(f)); Article 6(1)(b) "processing is necessary for the performance of a contract to which the data subject is party or in order to take steps at the request of the data subject prior to entering into a contract"; Article 6(1)(f) legitimate interests.
- `gdpr` Article 6 recital refs: recitals 40–50 and 155 (text not in corpus).

**Evidence.** The choice of basis determines what the controller can do with the data and whether it can further-process it (e.g., train or refine the model on historical applicants), which interacts with AI Act Article 10(2)(b) data-origin obligations.

**Strength: MODERATE.** Binding, but which basis applies is a case-by-case determination.

---

### Finding G6 — Special-category data boundary (credit data is *not* special-category data)

**Claim.** Ordinary credit-scoring inputs (income, employment, payment history, existing debts) are ordinary personal data, not "special categories"; the Article 9 prohibition only engages if the system touches race/ethnicity, health, biometric, genetic or other listed data — in which case processing is prohibited unless an Article 9(2) exception applies.

**Citations.**

- `gdpr` Article 9(1): "Processing of personal data revealing racial or ethnic origin, political opinions, religious or philosophical beliefs, or trade union membership, and the processing of genetic data, biometric data… data concerning health or data concerning a natural person's sex life or sexual orientation shall be prohibited."
- `gdpr` Article 9(2): exhaustive exceptions.
- `ai-act` Article 10(5): providers "may exceptionally process special categories of personal data" for bias detection/correction only under strict conditions (necessity, pseudonymisation, deletion, etc.).
- Recital refs: Art 9 → recitals 51–56 (text not in corpus).

**Evidence.** The claim that "credit scoring data is special-category data" is **not** supported by the corpus (see Not supported below); the real risk is the system *indirectly* using or inferring protected attributes (e.g., health data via medical leave history), and the AI Act separately allows limited special-category processing for bias detection.

**Strength: MODERATE.** Clarifies scope and points to the actual Art 9 risk.

---

### Finding G7 — Data-protection principles and accountability

**Claim.** The controller is accountable for demonstrating compliance with the GDPR principles (lawfulness, fairness, transparency, purpose limitation, data minimisation, accuracy, storage limitation, security), and must implement data protection by design and by default.

**Citations.**

- `gdpr` Article 5(1) and Article 5(2) ("The controller shall be responsible for, and be able to demonstrate compliance with, paragraph 1 ('accountability').")
- `gdpr` Article 25(1)–(2) (data protection by design and by default).
- `gdpr` Article 83(5)(b): infringements of "the data subjects' rights pursuant to Articles 12 to 22" are subject to fines "up to 20 000 000 EUR, or… up to 4 % of the total worldwide annual turnover."
- Recital refs: Art 5 → recital 39; Art 25 → recital 78 (text not in corpus).

**Evidence.** General obligations that frame everything else and carry the headline GDPR fine tier; they translate into concrete tasks (records of processing, data-minimised design, accuracy controls on training data).

**Strength: MODERATE.** Binding but general; they support claims about process obligations rather than specific loan-decision rules.

---

## DORA findings

### Finding D1 — DORA applies directly if the company is itself a financial entity

**Claim.** If the company is itself a financial entity (in practice: a licensed credit institution that originates loans), DORA applies in full — ICT risk-management framework, incident management and reporting, digital operational resilience testing, and ICT third-party risk management.

**Citations.**

- `dora` Article 2(1)(a) (credit institutions) — Article 2(1)(u) (ICT third-party service providers); Article 2(2): entities in Article 2(1)(a)–(t) are "financial entities."
- `dora` Article 5 (governance and organisation) and Article 6 (ICT risk management framework), including Article 6(1): "Financial entities shall have a sound, comprehensive and well-documented ICT risk management framework as part of their overall risk management system."
- `dora` Chapter III (incident classification and reporting, Articles 17–23) and Article 24 (digital operational resilience testing programme).
- `dora` Article 2 recital refs: recitals 37, 40–41 and 70; Article 6 recital refs: recitals 34 and 51 (recital text present in corpus).

**Evidence.** DORA's obligations attach to "financial entities" and to ICT third-party service providers. Whether the company is a credit institution (which requires a licence under Directive 2013/36/EU) is a threshold question to investigate; a company that *only evaluates* loan applications without granting credit may fall outside Article 2(1)(a)–(t). Note also DORA's proportionality carve-outs (Article 4; simplified framework for small entities, Article 16; recitals 38, 43).

**Strength: MODERATE.** The obligations are binding, but whether they bind *this* company depends on its authorisation status, which the corpus cannot answer.

---

### Finding D2 — If the AI is supplied to banks, the company is an ICT third-party service provider

**Claim.** Where the company supplies the loan-evaluation AI as a service to financial entities, it is an "ICT third-party service provider" under DORA; its bank customers must then manage ICT third-party risk against it — through the register of information, pre-contracting due diligence and mandatory contractual provisions — and the provider may itself be designated "critical" and overseen by the ESAs.

**Citations.**

- `dora` Article 3(19): "'ICT third-party service provider' means an undertaking providing ICT services"; Article 3(21) ("ICT services").
- `dora` Article 2(1)(u) — such providers are in scope of the Regulation.
- `dora` Article 28(1)–(3) (ICT third-party risk as an integral component of the ICT risk-management framework; register of all ICT-service contracts) and Article 28(4) (due diligence before contracting).
- `dora` Article 30(1)–(2) (key contractual provisions: full written contract with service levels; data availability/integrity/confidentiality; access/recovery/return of data; cooperation with authorities; termination rights).
- `dora` Article 31 (designation of critical ICT third-party service providers and ESA oversight).
- `dora` Recitals 62–69 and 71–73 (text present in corpus), e.g. Recital 63: "this Regulation should cover a wide range of ICT third-party service providers, including providers of cloud computing services, software, data analytics services and providers of data centre services"; Recital 29 on key principles for managing ICT third-party risk.

**Evidence.** The obligations in Chapter V run to the financial entity (the customer), but they create concrete contractual and audit obligations that a supplier of credit-scoring software must accept; designation under Article 31 imposes direct ESA oversight on the provider itself.

**Strength: MODERATE.** Directly verifiable definitions and duties, but application requires the company to be a supplier to financial entities — a factual determination.

---

### Finding D3 — DORA regulates digital operational resilience, not lending decisions

**Claim.** DORA does not regulate the substance of credit decisions, creditworthiness assessment or automated decision-making; its subject matter is the security of network and information systems and ICT risk supporting financial entities' operations.

**Citations.**

- `dora` Article 1(1): "this Regulation lays down uniform requirements concerning the security of network and information systems supporting the business processes of financial entities."
- `dora` Article 3(5): "'ICT risk' means any reasonably identifiable circumstance in relation to the use of network and information systems… which may compromise the security of the network and information systems… or of the provision of services."
- `dora` Recital 2 (text present in corpus): digitalisation covers "lending and funding operations… credit rating," i.e., DORA's interest in credit is about the ICT that supports it, not the credit decision itself.

**Evidence.** This is a scope-clarifying finding: DORA answers "how resilient must the systems be," while the AI Act and GDPR answer "how may the decisions be made and data handled." Overclaiming DORA as a credit/AI rule is not supported by the corpus.

**Strength: MODERATE.** Correctly scopes DORA's role in the answer; verifiable from Articles 1 and 3(5) and Recital 2.

---

## Not supported by the corpus

The following claims, which a naive answer might make, are **not** supported by the source documents:

1. **"Credit scoring data is special-category (sensitive) data."** Not supported — GDPR Article 9(1) enumerates race/ethnicity, political opinions, religion, trade-union membership, genetic, biometric, health, sex life/orientation; financial/credit data is ordinary personal data. What the corpus *does* support: if the system touches those listed categories, processing is prohibited unless an Article 9(2) exception applies (G6).

2. **"DORA applies to every fintech."** Not supported — DORA's scope (Article 2) is limited to the listed financial entities (a)–(t) and ICT third-party service providers (u); a company that merely evaluates loan applications and is not a licensed financial entity or a supplier to one is outside scope (D1/D2).

3. **"DORA governs AI decisions / automated credit decisions."** Not supported — DORA's subject matter is digital operational resilience and ICT risk (Article 1, Article 3(5)); it contains no provisions on creditworthiness, automated decision-making or profiling (D3).

4. **"The AI Act prohibits automated credit scoring."** Not supported — Article 5's prohibited practices list (social scoring, emotion inference, biometric categorisation, criminal-risk prediction, etc.) does not include credit scoring; the correct characterisation is "lawful but high-risk" (Findings 1 and 9).

5. **"AI Act Article 6(1)-style product high-risk obligations are currently applicable."** Not supported — Article 113(3)(c) defers Article 6(1) and the corresponding obligations to 2 August 2027; Article 6(2) + Annex III (the route for creditworthiness AI) already apply from 2 August 2026.

6. **Recital text for the AI Act and GDPR.** Not present in the corpus. Article-level recital *references* exist (e.g., GDPR Article 22 references recitals 71, 72, 91; AI Act Article 6 references recitals 46–48 and 50–63), but the wording of those recitals cannot be quoted from `ai-act.json` or `gdpr.json`. Only DORA recitals carry full text.

7. **Spain-specific requirements** (e.g., obligations under the Spanish Credit Agreements law or national AI Act / AEPD guidance, or the EU Consumer Credit Directive / Mortgage Credit Directive, which would shape what "investigate before deployment" means in Spain) are outside the corpus entirely.

8. **Whether the company will be "provider" vs "deployer"** for the AI Act is a factual question the corpus cannot resolve — if the company builds the model itself it is a provider (Article 3(3)); if it buys and operates a third-party model it is a deployer (Article 3(4)); the corpus supports both characterisations and their different obligation sets (Findings 2 vs 3), but cannot decide which applies.

---

*Research notes — Regula demo scenario. Citations verified against `data/regulations/ai-act.json`, `gdpr.json`, `dora.json` on 2026-08-20. This is a research prototype, not legal advice.*