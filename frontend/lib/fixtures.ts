import type { AnalyzeResponse } from "./contract";

export const demoAnalyzeResponse: AnalyzeResponse = {
  "answer": {
    "findings": [
      {
        "statement": "An AI system that evaluates the creditworthiness of natural persons or establishes their credit score is a high-risk AI system under the AI Act, so the full high-risk obligations apply.",
        "strength": "strong",
        "citations": [
          {
            "source_id": "ai-act",
            "source_short_name": "EU AI Act",
            "article_number": 6,
            "recital_number": null,
            "annex_number": null,
            "section": "Classification of AI Systems as High-Risk",
            "provision": "Article 6(2)",
            "quote": "Irrespective of whether an AI system is placed on the market or put into service independently of the products referred to in points (a) and (b), that AI system shall be considered to be high-risk where both of the following conditions are fulfilled: (a) the AI system is intended to be used as a saf..."
          },
          {
            "source_id": "ai-act",
            "source_short_name": "EU AI Act",
            "article_number": null,
            "recital_number": null,
            "annex_number": 3,
            "section": "Annexes",
            "provision": "Annex III point 5(b)",
            "quote": "High-risk AI systems pursuant to Article 6(2) are the AI systems listed in any of the following areas:\n\n1. Biometrics, in so far as their use is permitted under relevant Union or national law:\n(a) remote biometric identification systems. This shall not include AI systems intended to be used for biom..."
          }
        ]
      },
      {
        "statement": "As deployer, the company must assign human oversight, keep automatically generated logs for at least six months, and inform applicants that they are subject to a high-risk AI system.",
        "strength": "strong",
        "citations": [
          {
            "source_id": "ai-act",
            "source_short_name": "EU AI Act",
            "article_number": 26,
            "recital_number": null,
            "annex_number": null,
            "section": "Obligations of Providers and Deployers",
            "provision": "Article 26(2), (4), (6), (11)",
            "quote": "Deployers of high-risk AI systems shall take appropriate technical and organisational measures to ensure they use such systems in accordance with the instructions for use accompanying the systems, pursuant to paragraphs 3 and 6. Deployers shall assign human oversight to natural persons who have the ..."
          }
        ]
      },
      {
        "statement": "Before first deployment, the deployer of the creditworthiness system must perform a Fundamental Rights Impact Assessment and notify its results to the market surveillance authority.",
        "strength": "strong",
        "citations": [
          {
            "source_id": "ai-act",
            "source_short_name": "EU AI Act",
            "article_number": 27,
            "recital_number": null,
            "annex_number": null,
            "section": "Obligations of Providers and Deployers",
            "provision": "Article 27(1)-(4)",
            "quote": "Prior to deploying a high-risk AI system referred to in Article 6(2), with the exception of high-risk AI systems intended to be used in the area listed in point 2 of Annex III, deployers that are bodies governed by public law, or are private entities providing public services, and deployers of high-..."
          }
        ]
      },
      {
        "statement": "Applicants subject to a loan decision based on the system's output have a right to a clear and meaningful explanation of the role of the AI system in the decision.",
        "strength": "strong",
        "citations": [
          {
            "source_id": "ai-act",
            "source_short_name": "EU AI Act",
            "article_number": 86,
            "recital_number": null,
            "annex_number": null,
            "section": "Enforcement",
            "provision": "Article 86(1)",
            "quote": "Any affected person subject to a decision which is taken by the deployer on the basis of the output from a high-risk AI system listed in Annex III, with the exception of systems listed under point 2 thereof, and which produces legal effects or similarly significantly affects that person in a way tha..."
          }
        ]
      },
      {
        "statement": "GDPR restricts decisions based solely on automated processing, including profiling, that produce legal or similarly significant effects; loan scoring is such a decision, and even the contract-necessity exception still requires human intervention and contest rights.",
        "strength": "strong",
        "citations": [
          {
            "source_id": "gdpr",
            "source_short_name": "GDPR",
            "article_number": 22,
            "recital_number": null,
            "annex_number": null,
            "section": "Right to Object and Automated Individual Decision-Making",
            "provision": "Article 22(1), (2)(a), (3)",
            "quote": "The data subject shall have the right not to be subject to a decision based solely on automated processing, including profiling, which produces legal effects concerning him or her or similarly significantly affects him or her. Paragraph 1 shall not apply if the decision: (a) is necessary for enterin..."
          },
          {
            "source_id": "gdpr",
            "source_short_name": "GDPR",
            "article_number": null,
            "recital_number": 71,
            "annex_number": null,
            "section": "Recitals",
            "provision": "Recital 71",
            "quote": null
          }
        ]
      },
      {
        "statement": "The credit-scoring processing requires a data protection impact assessment before it starts, because it is a systematic and extensive automated evaluation on which legally effective decisions are based.",
        "strength": "strong",
        "citations": [
          {
            "source_id": "gdpr",
            "source_short_name": "GDPR",
            "article_number": 35,
            "recital_number": null,
            "annex_number": null,
            "section": "Data Protection Impact Assessment and Prior Consultation",
            "provision": "Article 35(1), (3)(a)",
            "quote": "Where a type of processing in particular using new technologies, and taking into account the nature, scope, context and purposes of the processing, is likely to result in a high risk to the rights and freedoms of natural persons, the controller shall, prior to the processing, carry out an assessment..."
          }
        ]
      },
      {
        "statement": "DORA applies in full only if the company is itself a licensed financial entity; otherwise its main relevance is through ICT third-party risk where the AI is supplied to financial entities.",
        "strength": "moderate",
        "citations": [
          {
            "source_id": "dora",
            "source_short_name": "DORA",
            "article_number": 2,
            "recital_number": null,
            "annex_number": null,
            "section": "General provisions",
            "provision": "Article 2(1)(a), (2)",
            "quote": "Without prejudice to paragraphs 3 and 4, this Regulation applies to the following entities: For the purposes of this Regulation, entities referred to in paragraph 1, points (a) to (t), shall collectively be referred to as 'financial entities'. This Regulation does not apply to: Member States may exc..."
          }
        ]
      },
      {
        "statement": "DORA governs the digital operational resilience of financial entities, not the substance of credit decisions; credit scoring itself is regulated by the AI Act and GDPR, not DORA.",
        "strength": "moderate",
        "citations": [
          {
            "source_id": "dora",
            "source_short_name": "DORA",
            "article_number": 1,
            "recital_number": null,
            "annex_number": null,
            "section": "General provisions",
            "provision": "Article 1(1)",
            "quote": "In order to achieve a high common level of digital operational resilience, this Regulation lays down uniform requirements concerning the security of network and information systems supporting the business processes of financial entities as follows: In relation to financial entities identified as ess..."
          }
        ]
      },
      {
        "statement": "The system qualifies as an AI system; the company will be a provider and/or deployer; credit scoring is a form of profiling as cross-referenced into the AI Act.",
        "strength": "weak",
        "citations": [
          {
            "source_id": "ai-act",
            "source_short_name": "EU AI Act",
            "article_number": 3,
            "recital_number": null,
            "annex_number": null,
            "section": "General Provisions",
            "provision": "Article 3(1), (3), (4), (52)",
            "quote": "For the purposes of this Regulation, the following definitions apply: (1) 'AI system' means a machine-based system that is designed to operate with varying levels of autonomy and that may exhibit adaptiveness after deployment, and that, for explicit or implicit objectives, infers, from the input it ..."
          }
        ]
      }
    ],
    "actions": [
      "Have a qualified professional determine whether the company is a provider or a deployer under the AI Act, since the obligation sets differ (AI Act Article 3(3)-(4)).",
      "Have a qualified professional confirm whether the company is a licensed financial entity under DORA Article 2, which decides whether DORA applies in full.",
      "Have a qualified professional assess whether a GDPR data protection impact assessment (Article 35(3)(a)) and an AI Act Fundamental Rights Impact Assessment (Article 27) are required before first deployment.",
      "Have a qualified professional verify that human oversight accompanies the credit decision process (AI Act Article 26; GDPR Article 22(3)), so decisions are not solely automated.",
      "Have a qualified professional confirm how applicants will be informed of the AI system's role in decisions (AI Act Article 86; GDPR Articles 13(2)(f) and 15(1)(h)).",
      "Have a qualified legal professional verify these findings against the company's actual situation before acting on them."
    ],
    "citations": [
      {
        "source_id": "ai-act",
        "source_short_name": "EU AI Act",
        "article_number": 6,
        "recital_number": null,
        "annex_number": null,
        "section": "Classification of AI Systems as High-Risk",
        "provision": "Article 6(2)",
        "quote": "Irrespective of whether an AI system is placed on the market or put into service independently of the products referred to in points (a) and (b), that AI system shall be considered to be high-risk where both of the following conditions are fulfilled: (a) the AI system is intended to be used as a saf..."
      },
      {
        "source_id": "ai-act",
        "source_short_name": "EU AI Act",
        "article_number": null,
        "recital_number": null,
        "annex_number": 3,
        "section": "Annexes",
        "provision": "Annex III point 5(b)",
        "quote": "High-risk AI systems pursuant to Article 6(2) are the AI systems listed in any of the following areas:\n\n1. Biometrics, in so far as their use is permitted under relevant Union or national law:\n(a) remote biometric identification systems. This shall not include AI systems intended to be used for biom..."
      },
      {
        "source_id": "ai-act",
        "source_short_name": "EU AI Act",
        "article_number": 26,
        "recital_number": null,
        "annex_number": null,
        "section": "Obligations of Providers and Deployers",
        "provision": "Article 26(2), (4), (6), (11)",
        "quote": "Deployers of high-risk AI systems shall take appropriate technical and organisational measures to ensure they use such systems in accordance with the instructions for use accompanying the systems, pursuant to paragraphs 3 and 6. Deployers shall assign human oversight to natural persons who have the ..."
      },
      {
        "source_id": "ai-act",
        "source_short_name": "EU AI Act",
        "article_number": 27,
        "recital_number": null,
        "annex_number": null,
        "section": "Obligations of Providers and Deployers",
        "provision": "Article 27(1)-(4)",
        "quote": "Prior to deploying a high-risk AI system referred to in Article 6(2), with the exception of high-risk AI systems intended to be used in the area listed in point 2 of Annex III, deployers that are bodies governed by public law, or are private entities providing public services, and deployers of high-..."
      },
      {
        "source_id": "ai-act",
        "source_short_name": "EU AI Act",
        "article_number": 86,
        "recital_number": null,
        "annex_number": null,
        "section": "Enforcement",
        "provision": "Article 86(1)",
        "quote": "Any affected person subject to a decision which is taken by the deployer on the basis of the output from a high-risk AI system listed in Annex III, with the exception of systems listed under point 2 thereof, and which produces legal effects or similarly significantly affects that person in a way tha..."
      },
      {
        "source_id": "gdpr",
        "source_short_name": "GDPR",
        "article_number": 22,
        "recital_number": null,
        "annex_number": null,
        "section": "Right to Object and Automated Individual Decision-Making",
        "provision": "Article 22(1), (2)(a), (3)",
        "quote": "The data subject shall have the right not to be subject to a decision based solely on automated processing, including profiling, which produces legal effects concerning him or her or similarly significantly affects him or her. Paragraph 1 shall not apply if the decision: (a) is necessary for enterin..."
      },
      {
        "source_id": "gdpr",
        "source_short_name": "GDPR",
        "article_number": null,
        "recital_number": 71,
        "annex_number": null,
        "section": "Recitals",
        "provision": "Recital 71",
        "quote": null
      },
      {
        "source_id": "gdpr",
        "source_short_name": "GDPR",
        "article_number": 35,
        "recital_number": null,
        "annex_number": null,
        "section": "Data Protection Impact Assessment and Prior Consultation",
        "provision": "Article 35(1), (3)(a)",
        "quote": "Where a type of processing in particular using new technologies, and taking into account the nature, scope, context and purposes of the processing, is likely to result in a high risk to the rights and freedoms of natural persons, the controller shall, prior to the processing, carry out an assessment..."
      },
      {
        "source_id": "dora",
        "source_short_name": "DORA",
        "article_number": 2,
        "recital_number": null,
        "annex_number": null,
        "section": "General provisions",
        "provision": "Article 2(1)(a), (2)",
        "quote": "Without prejudice to paragraphs 3 and 4, this Regulation applies to the following entities: For the purposes of this Regulation, entities referred to in paragraph 1, points (a) to (t), shall collectively be referred to as 'financial entities'. This Regulation does not apply to: Member States may exc..."
      },
      {
        "source_id": "dora",
        "source_short_name": "DORA",
        "article_number": 1,
        "recital_number": null,
        "annex_number": null,
        "section": "General provisions",
        "provision": "Article 1(1)",
        "quote": "In order to achieve a high common level of digital operational resilience, this Regulation lays down uniform requirements concerning the security of network and information systems supporting the business processes of financial entities as follows: In relation to financial entities identified as ess..."
      },
      {
        "source_id": "ai-act",
        "source_short_name": "EU AI Act",
        "article_number": 3,
        "recital_number": null,
        "annex_number": null,
        "section": "General Provisions",
        "provision": "Article 3(1), (3), (4), (52)",
        "quote": "For the purposes of this Regulation, the following definitions apply: (1) 'AI system' means a machine-based system that is designed to operate with varying levels of autonomy and that may exhibit adaptiveness after deployment, and that, for explicit or implicit objectives, infers, from the input it ..."
      }
    ]
  },
  "trace": {
    "workflow": "planner -> researcher -> verifier -> proposer",
    "summary": "Planner identified automated credit decisions, profiling, and ICT risk as research targets; Researcher retrieved provisions from the AI Act, GDPR, and DORA; Verifier recorded each anticipated-but-unsupported claim as rejected \u2014 no Evidence in the Corpus supports it.",
    "unsupported_claims_discarded": [
      "Credit scoring data is special-category (sensitive) data.",
      "DORA applies to every fintech.",
      "DORA governs AI decisions / automated credit decisions.",
      "The AI Act prohibits automated credit scoring.",
      "AI Act Article 6(1)-style product high-risk obligations are currently applicable.",
      "Recital text for the AI Act and GDPR is available in the corpus.",
      "Spain-specific requirements are covered by the corpus.",
      "Whether the company is provider vs deployer can be determined from the corpus."
    ]
  },
  "detailed_trace": [
    {
      "step": "planner",
      "action": "identify topics: automated credit decisions, high-risk AI, profiling, data protection, ICT risk"
    },
    {
      "step": "researcher",
      "action": "retrieve high-risk provisions from AI Act, GDPR, and DORA via corpus_lookup",
      "retrieved": [
        {
          "source_id": "ai-act",
          "kind": "article",
          "number": 6,
          "section": "Classification of AI Systems as High-Risk",
          "provision": "Article 6(2)",
          "text": "Irrespective of whether an AI system is placed on the market or put into service independently of the products referred to in points (a) and (b), that AI system shall be considered to be high-risk where both of the following conditions are fulfilled: (a) the AI system is intended to be used as a safety component of a product, or the AI system is itself a product, covered by the Union harmonisation legislation listed in Annex I; (b) the product whose safety component pursuant to point (a) is the AI system, or the AI system itself as a product, is required to undergo a third-party conformity assessment, with a view to the placing on the market or the putting into service of that product pursuant to the Union harmonisation legislation listed in Annex I. In addition to the high-risk AI systems referred to in paragraph 1, AI systems referred to in Annex III shall be considered to be high-risk. By derogation from paragraph 2, an AI system referred to in Annex III shall not be considered to b"
        },
        {
          "source_id": "ai-act",
          "kind": "annex",
          "number": 3,
          "section": "Annexes",
          "provision": "Annex III point 5(b)",
          "text": "High-risk AI systems pursuant to Article 6(2) are the AI systems listed in any of the following areas:\n\n1. Biometrics, in so far as their use is permitted under relevant Union or national law:\n(a) remote biometric identification systems. This shall not include AI systems intended to be used for biometric verification the sole purpose of which is to confirm that a specific natural person is the person he or she claims to be;\n(b) AI systems intended to be used for biometric categorisation, according to sensitive or protected attributes or characteristics based on the inference of those attributes or characteristics;\n(c) AI systems intended to be used for emotion recognition.\n\n2. Critical infrastructure: AI systems intended to be used as safety components in the management and operation of critical digital infrastructure, road traffic, or in the supply of water, gas, heating or electricity.\n\n3. Education and vocational training:\n(a) AI systems intended to be used to determine access or ad"
        },
        {
          "source_id": "ai-act",
          "kind": "article",
          "number": 26,
          "section": "Obligations of Providers and Deployers",
          "provision": "Article 26(2), (4), (6), (11)",
          "text": "Deployers of high-risk AI systems shall take appropriate technical and organisational measures to ensure they use such systems in accordance with the instructions for use accompanying the systems, pursuant to paragraphs 3 and 6. Deployers shall assign human oversight to natural persons who have the necessary competence, training and authority, as well as the necessary support. The obligations set out in paragraphs 1 and 2, are without prejudice to other deployer obligations under Union or national law and to the deployer's freedom to organise its own resources and activities for the purpose of implementing the human oversight measures indicated by the provider. Without prejudice to paragraphs 1 and 2, to the extent the deployer exercises control over the input data, that deployer shall ensure that input data is relevant and sufficiently representative in view of the intended purpose of the high-risk AI system. Deployers shall monitor the operation of the high-risk AI system on the basi"
        },
        {
          "source_id": "ai-act",
          "kind": "article",
          "number": 27,
          "section": "Obligations of Providers and Deployers",
          "provision": "Article 27(1)-(4)",
          "text": "Prior to deploying a high-risk AI system referred to in Article 6(2), with the exception of high-risk AI systems intended to be used in the area listed in point 2 of Annex III, deployers that are bodies governed by public law, or are private entities providing public services, and deployers of high-risk AI systems referred to in points 5 (b) and (c) of Annex III, shall perform an assessment of the impact on fundamental rights that the use of such system may produce. For that purpose, deployers shall perform an assessment consisting of: (a) a description of the deployer's processes in which the high-risk AI system will be used in line with its intended purpose; (b) a description of the period of time within which, and the frequency with which, each high-risk AI system is intended to be used; (c) the categories of natural persons and groups likely to be affected by its use in the specific context; (d) the specific risks of harm likely to have an impact on the categories of natural person"
        },
        {
          "source_id": "ai-act",
          "kind": "article",
          "number": 86,
          "section": "Enforcement",
          "provision": "Article 86(1)",
          "text": "Any affected person subject to a decision which is taken by the deployer on the basis of the output from a high-risk AI system listed in Annex III, with the exception of systems listed under point 2 thereof, and which produces legal effects or similarly significantly affects that person in a way that they consider to have an adverse impact on their health, safety or fundamental rights shall have the right to obtain from the deployer clear and meaningful explanations of the role of the AI system in the decision-making procedure and the main elements of the decision taken. Paragraph 1 shall not apply to the use of AI systems for which exceptions from, or restrictions to, the obligation under that paragraph follow from Union or national law in compliance with Union law. This Article shall apply only to the extent that the right referred to in paragraph 1 is not otherwise provided for under Union law."
        },
        {
          "source_id": "gdpr",
          "kind": "article",
          "number": 22,
          "section": "Right to Object and Automated Individual Decision-Making",
          "provision": "Article 22(1), (2)(a), (3)",
          "text": "The data subject shall have the right not to be subject to a decision based solely on automated processing, including profiling, which produces legal effects concerning him or her or similarly significantly affects him or her. Paragraph 1 shall not apply if the decision: (a) is necessary for entering into, or performance of, a contract between the data subject and a data controller; (b) is authorised by Union or Member State law to which the controller is subject and which also lays down suitable measures to safeguard the data subject's rights and freedoms and legitimate interests; or (c) is based on the data subject's explicit consent. In the cases referred to in points (a) and (c) of paragraph 2, the data controller shall implement suitable measures to safeguard the data subject's rights and freedoms and legitimate interests, at least the right to obtain human intervention on the part of the controller, to express his or her point of view and to contest the decision. Decisions referr"
        },
        {
          "source_id": "gdpr",
          "kind": "recital",
          "number": 71,
          "section": "Recitals",
          "provision": "Recital 71",
          "text": null
        },
        {
          "source_id": "gdpr",
          "kind": "article",
          "number": 35,
          "section": "Data Protection Impact Assessment and Prior Consultation",
          "provision": "Article 35(1), (3)(a)",
          "text": "Where a type of processing in particular using new technologies, and taking into account the nature, scope, context and purposes of the processing, is likely to result in a high risk to the rights and freedoms of natural persons, the controller shall, prior to the processing, carry out an assessment of the impact of the envisaged processing operations on the protection of personal data. A single assessment may address a set of similar processing operations that present similar high risks. The controller shall seek the advice of the data protection officer, where designated, when carrying out a data protection impact assessment. A data protection impact assessment referred to in paragraph 1 shall in particular be required in the case of: (a) a systematic and extensive evaluation of personal aspects relating to natural persons which is based on automated processing, including profiling, and on which decisions are based that produce legal effects concerning the natural person or similarly"
        },
        {
          "source_id": "dora",
          "kind": "article",
          "number": 2,
          "section": "General provisions",
          "provision": "Article 2(1)(a), (2)",
          "text": "Without prejudice to paragraphs 3 and 4, this Regulation applies to the following entities: For the purposes of this Regulation, entities referred to in paragraph 1, points (a) to (t), shall collectively be referred to as 'financial entities'. This Regulation does not apply to: Member States may exclude from the scope of this Regulation entities referred to in Article 2(5), points (4) to (23), of Directive 2013/36/EU that are located within their respective territories. Where a Member State makes use of such option, it shall inform the Commission thereof as well as of any subsequent changes thereto. The Commission shall make that information publicly available on its website or other easily accessible means."
        },
        {
          "source_id": "dora",
          "kind": "article",
          "number": 1,
          "section": "General provisions",
          "provision": "Article 1(1)",
          "text": "In order to achieve a high common level of digital operational resilience, this Regulation lays down uniform requirements concerning the security of network and information systems supporting the business processes of financial entities as follows: In relation to financial entities identified as essential or important entities pursuant to national rules transposing Article 3 of Directive (EU) 2022/2555, this Regulation shall be considered a sector-specific Union legal act for the purposes of Article 4 of that Directive. This Regulation is without prejudice to the responsibility of Member States' regarding essential State functions concerning public security, defence and national security in accordance with Union law."
        },
        {
          "source_id": "ai-act",
          "kind": "article",
          "number": 3,
          "section": "General Provisions",
          "provision": "Article 3(1), (3), (4), (52)",
          "text": "For the purposes of this Regulation, the following definitions apply: (1) 'AI system' means a machine-based system that is designed to operate with varying levels of autonomy and that may exhibit adaptiveness after deployment, and that, for explicit or implicit objectives, infers, from the input it receives, how to generate outputs such as predictions, content, recommendations, or decisions that can influence physical or virtual environments; (2) 'risk' means the combination of the probability of an occurrence of harm and the severity of that harm; (3) 'provider' means a natural or legal person, public authority, agency or other body that develops an AI system or a general-purpose AI model or that has an AI system or a general-purpose AI model developed and places it on the market or puts the AI system into service under its own name or trademark, whether for payment or free of charge; (4) 'deployer' means a natural or legal person, public authority, agency or other body using an AI sy"
        }
      ],
      "tool_calls": [
        {
          "tool": "corpus_lookup",
          "input": {
            "source_id": "ai-act",
            "kind": "article",
            "number": 6
          },
          "status": "found"
        },
        {
          "tool": "corpus_lookup",
          "input": {
            "source_id": "ai-act",
            "kind": "annex",
            "number": 3
          },
          "status": "found"
        },
        {
          "tool": "corpus_lookup",
          "input": {
            "source_id": "ai-act",
            "kind": "article",
            "number": 26
          },
          "status": "found"
        },
        {
          "tool": "corpus_lookup",
          "input": {
            "source_id": "ai-act",
            "kind": "article",
            "number": 27
          },
          "status": "found"
        },
        {
          "tool": "corpus_lookup",
          "input": {
            "source_id": "ai-act",
            "kind": "article",
            "number": 86
          },
          "status": "found"
        },
        {
          "tool": "corpus_lookup",
          "input": {
            "source_id": "gdpr",
            "kind": "article",
            "number": 22
          },
          "status": "found"
        },
        {
          "tool": "corpus_lookup",
          "input": {
            "source_id": "gdpr",
            "kind": "recital",
            "number": 71
          },
          "status": "found"
        },
        {
          "tool": "corpus_lookup",
          "input": {
            "source_id": "gdpr",
            "kind": "article",
            "number": 35
          },
          "status": "found"
        },
        {
          "tool": "corpus_lookup",
          "input": {
            "source_id": "dora",
            "kind": "article",
            "number": 2
          },
          "status": "found"
        },
        {
          "tool": "corpus_lookup",
          "input": {
            "source_id": "dora",
            "kind": "article",
            "number": 1
          },
          "status": "found"
        },
        {
          "tool": "corpus_lookup",
          "input": {
            "source_id": "ai-act",
            "kind": "article",
            "number": 3
          },
          "status": "found"
        }
      ]
    },
    {
      "step": "verifier",
      "action": "record the curated anticipated-but-unsupported claims as rejected (no Evidence in the Corpus supports this claim) and tag each finding with its evidence strength",
      "claim_decisions": [
        {
          "claim": "Credit scoring data is special-category (sensitive) data.",
          "status": "rejected",
          "reason": "no Evidence in the Corpus supports this claim"
        },
        {
          "claim": "DORA applies to every fintech.",
          "status": "rejected",
          "reason": "no Evidence in the Corpus supports this claim"
        },
        {
          "claim": "DORA governs AI decisions / automated credit decisions.",
          "status": "rejected",
          "reason": "no Evidence in the Corpus supports this claim"
        },
        {
          "claim": "The AI Act prohibits automated credit scoring.",
          "status": "rejected",
          "reason": "no Evidence in the Corpus supports this claim"
        },
        {
          "claim": "AI Act Article 6(1)-style product high-risk obligations are currently applicable.",
          "status": "rejected",
          "reason": "no Evidence in the Corpus supports this claim"
        },
        {
          "claim": "Recital text for the AI Act and GDPR is available in the corpus.",
          "status": "rejected",
          "reason": "no Evidence in the Corpus supports this claim"
        },
        {
          "claim": "Spain-specific requirements are covered by the corpus.",
          "status": "rejected",
          "reason": "no Evidence in the Corpus supports this claim"
        },
        {
          "claim": "Whether the company is provider vs deployer can be determined from the corpus.",
          "status": "rejected",
          "reason": "no Evidence in the Corpus supports this claim"
        }
      ]
    }
  ],
  "known_limitations": [
    "Known limitation: the corpus is English-only; questions in other languages are answered in English.",
    "Known limitation: Regula is a research prototype, not legal advice; answers require verification by a qualified professional."
  ]
};

export const insufficientEvidenceResponse: AnalyzeResponse = {
  "answer": {
    "findings": [],
    "actions": [
      "Rephrase the Regulatory question using the vocabulary of the Corpus \u2014 EU AI Act, GDPR, or DORA terms such as 'high-risk AI system', 'automated decision-making', or 'ICT third-party risk'.",
      "Check whether the obligation you are asking about falls outside the Corpus (e.g. national requirements of a single member state)."
    ],
    "citations": []
  },
  "trace": {
    "workflow": "planner -> researcher -> verifier -> proposer",
    "summary": "Planner identified research targets (employment-law obligations for Spanish restaurant booking software); retrieval returned no Chunks relevant enough, so no Claims were drafted or verified \u2014 the Corpus holds nothing for this question.",
    "unsupported_claims_discarded": []
  },
  "detailed_trace": [
    {
      "step": "planner",
      "action": "decompose the Regulatory question into research targets",
      "research_targets": [
        "employment-law obligations for Spanish restaurant booking software"
      ]
    },
    {
      "step": "researcher",
      "action": "retrieve Evidence exclusively via the retrieve_chunks tool, then draft Claims grounded in it",
      "retrieved": [],
      "tool_calls": []
    },
    {
      "step": "verifier",
      "action": "check each Claim against the retrieved Evidence, tag its Strength, discard Unsupported claims",
      "claim_decisions": []
    },
    {
      "step": "proposer",
      "action": "no Evidence was retrieved: the Insufficient-evidence path was served and the Proposer made no LLM call",
      "action_decisions": []
    }
  ],
  "known_limitations": [
    "Known limitation: the corpus is English-only; questions in other languages are answered in English.",
    "Known limitation: Regula is a research prototype, not legal advice; answers require verification by a qualified professional.",
    "Known limitation: the Corpus holds nothing relevant enough to answer this question, so no Findings were produced."
  ]
};

export const notAvailableResponse: AnalyzeResponse = {
  "answer": {
    "findings": [],
    "actions": [
      "The Corpus is not ingested yet, so Live mode has nothing to retrieve from.",
      "Ingest it once with: python -m backend.src.ingest",
      "Inside the Docker stack, run: docker compose exec backend python -m backend.src.ingest",
      "Ingestion takes effect immediately \u2014 re-run your request afterwards; no restart is needed.",
      "Set REGULA_MODE=demo (the default) to analyze the canonical Spanish fintech scenario via scenario.id 'spanish-fintech-startup-uses-9e165169'."
    ],
    "citations": []
  },
  "trace": {
    "workflow": "not-available",
    "summary": "Live mode selected but the vector store is empty; no retrieval performed.",
    "unsupported_claims_discarded": []
  },
  "detailed_trace": [],
  "known_limitations": [
    "Known limitation: the corpus is English-only; questions in other languages are answered in English."
  ]
};
