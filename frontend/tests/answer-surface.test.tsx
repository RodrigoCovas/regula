import assert from "node:assert/strict";
import { test } from "node:test";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { AnswerSurface } from "../components/AnswerSurface";
import { citationLabel } from "../components/CitationList";
import { FindingCard } from "../components/FindingCard";
import { StrengthBadge } from "../components/StrengthBadge";
import { demoAnalyzeResponse } from "../lib/fixtures";
import type { AnalyzeResponse } from "../lib/contract";
import { visibleMarkup } from "./escape";

function render(response = demoAnalyzeResponse): string {
  return renderToStaticMarkup(
    React.createElement(AnswerSurface, { response: response }),
  );
}

test("renders every Finding statement from the fixture", () => {
  const markup = render();
  for (const finding of demoAnalyzeResponse.answer.findings) {
    assert.ok(markup.includes(visibleMarkup(finding.statement)));
  }
});

test("renders Strength badges distinctly per level", () => {
  const strong = renderToStaticMarkup(
    React.createElement(StrengthBadge, { strength: "strong" }),
  );
  const moderate = renderToStaticMarkup(
    React.createElement(StrengthBadge, { strength: "moderate" }),
  );
  const weak = renderToStaticMarkup(
    React.createElement(StrengthBadge, { strength: "weak" }),
  );
  assert.ok(strong.includes(">strong<"));
  assert.ok(moderate.includes(">moderate<"));
  assert.ok(weak.includes(">weak<"));
});

test("renders each Finding with its Strength badge", () => {
  for (const finding of demoAnalyzeResponse.answer.findings) {
    const markup = renderToStaticMarkup(
      React.createElement(FindingCard, { finding: finding }),
    );
    assert.ok(markup.includes(visibleMarkup(finding.statement)));
    assert.ok(markup.includes(`>${finding.strength}<`));
  }
});

test("renders Citations within each Finding", () => {
  const markup = render();
  assert.ok(markup.includes("DORA — Article 17"));
  assert.ok(markup.includes("GDPR — Article 32"));
  assert.ok(markup.includes("DORA — Article 29"));
  const findings = demoAnalyzeResponse.answer.findings;
  for (const finding of findings) {
    for (const citation of finding.citations) {
      const label = citationLabel(citation);
      assert.ok(markup.includes(label), `missing citation label ${label}`);
    }
  }
});

test("renders every Citation contract field", () => {
  const markup = render();
  assert.ok(markup.includes("source_id: dora"));
  assert.ok(markup.includes("article_number: 17"));
  assert.ok(
    markup.includes("ICT-related incident management, classification and reporting"),
  );
  // The bank demo cites articles only; the recital and annex field rows stay
  // covered with synthetic per-Finding Citations, as the Live path cites all
  // three kinds.
  const withRecitalAndAnnex: AnalyzeResponse = {
    ...demoAnalyzeResponse,
    answer: {
      ...demoAnalyzeResponse.answer,
      findings: [
        {
          ...demoAnalyzeResponse.answer.findings[0],
          citations: [
            ...demoAnalyzeResponse.answer.findings[0].citations,
            {
              source_id: "gdpr",
              source_short_name: "GDPR",
              article_number: null,
              recital_number: 71,
              annex_number: null,
              section: "Recitals",
              provision: "Recital 71",
              quote: null,
            },
            {
              source_id: "ai-act",
              source_short_name: "EU AI Act",
              article_number: null,
              recital_number: null,
              annex_number: 3,
              section: "Annexes",
              provision: "Annex III point 5(b)",
              quote: null,
            },
          ],
        },
        ...demoAnalyzeResponse.answer.findings.slice(1),
      ],
    },
  };
  const markupWithRecitalAndAnnex = renderToStaticMarkup(
    React.createElement(AnswerSurface, { response: withRecitalAndAnnex }),
  );
  assert.ok(markupWithRecitalAndAnnex.includes("recital_number: 71"));
  assert.ok(markupWithRecitalAndAnnex.includes("annex_number: 3"));
});

test("renders every Answer action", () => {
  const markup = render();
  for (const action of demoAnalyzeResponse.answer.actions) {
    assert.ok(markup.includes(visibleMarkup(action)));
  }
});

test("renders Known limitations on every response", () => {
  const markup = render();
  for (const limitation of demoAnalyzeResponse.known_limitations) {
    assert.ok(markup.includes(limitation));
  }
});

test("renders the Execution trace summary inline", () => {
  const markup = render();
  assert.ok(markup.includes(visibleMarkup(demoAnalyzeResponse.trace.workflow)));
  assert.ok(markup.includes(visibleMarkup(demoAnalyzeResponse.trace.summary)));
  for (const claim of demoAnalyzeResponse.trace.unsupported_claims_discarded) {
    assert.ok(markup.includes(visibleMarkup(claim)));
  }
});

test("renders the detailed trace collapsed by default", () => {
  const markup = render();
  assert.ok(markup.includes("<details"), "expected a collapsible expander");
  assert.ok(
    !markup.includes("<details open"),
    "detailed trace must be collapsed by default",
  );
  assert.ok(markup.includes("Detailed execution trace"));
});

test("renders detailed trace steps, retrieved passages, tool calls, and claim decisions behind the expander", () => {
  const markup = render();
  assert.ok(markup.includes(">planner<"));
  assert.ok(markup.includes(">researcher<"));
  assert.ok(markup.includes(">verifier<"));
  assert.ok(markup.includes("Retrieved passages ("));
  assert.ok(markup.includes("Tool calls ("));
  assert.ok(markup.includes("corpus_lookup dora article 17 → found"));
  assert.ok(markup.includes("Claim decisions ("));
  assert.ok(
    markup.includes(
      "The outage is automatically a personal data breach under the GDPR.",
    ),
  );
  assert.ok(markup.includes(">rejected<"));
});

test("renders surviving Actions and the standing hand-off on the main answer surface, and action decisions in the detailed trace", () => {
  const survivingAction =
    "Have a professional verify the high-risk classification.";
  const handOff =
    "Have a qualified legal professional verify these findings against the company's actual situation before acting on them.";
  const responseWithActionDecisions: AnalyzeResponse = {
    ...demoAnalyzeResponse,
    answer: {
      ...demoAnalyzeResponse.answer,
      actions: [survivingAction, handOff],
    },
    detailed_trace: [
      ...(demoAnalyzeResponse.detailed_trace ?? []),
      {
        step: "proposer",
        action:
          "distill the kept Findings into referral Actions, each grounded in a kept Finding's Citations",
        action_decisions: [
          {
            action: survivingAction,
            status: "kept",
            reason: null,
            dropped_refs: [],
          },
          {
            action: "An action grounded on a Finding label.",
            status: "rejected",
            reason:
              "invalid grounding: F1 name Findings, not their Citations — use the C-labels shown under each Finding",
            dropped_refs: ["F1"],
          },
        ],
      },
    ],
  };
  const markup = renderToStaticMarkup(
    React.createElement(AnswerSurface, { response: responseWithActionDecisions }),
  );
  assert.ok(
    markup.includes(visibleMarkup(survivingAction)),
    "surviving grounded Action renders on the main answer surface",
  );
  assert.ok(
    markup.includes(visibleMarkup(handOff)),
    "standing professional hand-off renders on the main answer surface",
  );
  assert.ok(markup.includes(">proposer<"));
  assert.ok(markup.includes("Action proposals (2)"));
  assert.ok(markup.includes(">kept<"));
  assert.ok(
    markup.includes(
      visibleMarkup("An action grounded on a Finding label."),
    ),
  );
  assert.ok(markup.includes(">rejected<"));
  assert.ok(markup.includes("invalid grounding"));
  assert.ok(markup.includes("F1"));
  assert.ok(markup.includes("dropped refs: F1"));
});

test("does not render the dedicated panels for a full answer", () => {
  const markup = render();
  assert.ok(!markup.includes("Insufficient evidence"));
  assert.ok(!markup.includes("Not available"));
});
