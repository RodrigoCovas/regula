import assert from "node:assert/strict";
import { test } from "node:test";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { AnswerSurface } from "../components/AnswerSurface";
import { citationLabel } from "../components/CitationList";
import { StrengthBadge } from "../components/StrengthBadge";
import { demoAnalyzeResponse } from "../lib/fixtures";
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
  assert.ok(strong.includes("text-emerald"));
  assert.ok(moderate.includes("text-amber"));
  assert.ok(weak.includes("text-slate"));
  assert.ok(strong.includes(">strong<"));
  assert.ok(moderate.includes(">moderate<"));
  assert.ok(weak.includes(">weak<"));
});

test("renders each Finding with its Strength badge next to its statement", () => {
  const markup = render();
  for (const finding of demoAnalyzeResponse.answer.findings) {
    const statement = visibleMarkup(finding.statement);
    const statementStart = markup.indexOf(statement);
    const badgeStart = statementStart + statement.length;
    assert.ok(markup.slice(badgeStart, badgeStart + 500).includes(finding.strength));
  }
});

test("renders Citations within each Finding", () => {
  const markup = render();
  assert.ok(markup.includes("EU AI Act — Article 6(2)"));
  assert.ok(markup.includes("GDPR — Recital 71"));
  assert.ok(markup.includes("DORA — Article 2(1)(a), (2)"));
  const findings = demoAnalyzeResponse.answer.findings;
  for (const finding of findings) {
    for (const citation of finding.citations) {
      const label = citationLabel(citation);
      assert.ok(markup.includes(label), `missing citation label ${label}`);
    }
  }
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
  assert.ok(markup.includes("corpus_lookup ai-act article 6 → found"));
  assert.ok(markup.includes("Claim decisions ("));
  assert.ok(
    markup.includes(
      "Credit scoring data is special-category (sensitive) data.",
    ),
  );
  assert.ok(markup.includes(">rejected<"));
});

test("does not render the dedicated panels for a full answer", () => {
  const markup = render();
  assert.ok(!markup.includes("Insufficient evidence"));
  assert.ok(!markup.includes("Not available"));
});
