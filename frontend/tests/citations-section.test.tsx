import assert from "node:assert/strict";
import { test } from "node:test";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { CitationsSection } from "../components/CitationsSection";
import { demoAnalyzeResponse } from "../lib/fixtures";
import type { Citation } from "../lib/contract";
import { visibleMarkup } from "./escape";

const demoCitations = demoAnalyzeResponse.answer.citations;

test("renders every demo Citation label", () => {
  const markup = renderToStaticMarkup(
    React.createElement(CitationsSection, { citations: demoCitations }),
  );
  assert.ok(markup.includes("EU AI Act — Article 6(2)"));
  assert.ok(markup.includes("GDPR — Recital 71"));
  assert.ok(markup.includes("DORA — Article 2(1)(a), (2)"));
});

test("renders the Provision relevance statement on every demo Citation", () => {
  const markup = renderToStaticMarkup(
    React.createElement(CitationsSection, { citations: demoCitations }),
  );
  for (const citation of demoCitations) {
    assert.ok(
      markup.includes(visibleMarkup(citation.relevance ?? "")),
      `missing relevance for ${citation.provision}`,
    );
  }
});

test("renders the Citation-strength badge per Citation", () => {
  const markup = renderToStaticMarkup(
    React.createElement(CitationsSection, { citations: demoCitations }),
  );
  // The locked demo content spans all three Strength levels.
  assert.ok(markup.includes(">strong<"));
  assert.ok(markup.includes(">moderate<"));
  assert.ok(markup.includes(">weak<"));
});

function bareCitation(): Citation {
  return {
    ...demoCitations[0],
    relevance: null,
    strength: null,
  };
}

test("renders a Citation without relevance or strength bare", () => {
  const markup = renderToStaticMarkup(
    React.createElement(CitationsSection, { citations: [bareCitation()] }),
  );
  assert.ok(markup.includes("EU AI Act — Article 6(2)"));
  assert.ok(!markup.includes(">strong<"), "no badge without a Citation strength");
});

test("renders nothing when the Answer cites no provision", () => {
  const markup = renderToStaticMarkup(
    React.createElement(CitationsSection, { citations: [] }),
  );
  assert.equal(markup, "");
});
