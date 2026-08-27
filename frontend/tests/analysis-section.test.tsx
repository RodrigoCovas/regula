import assert from "node:assert/strict";
import { test } from "node:test";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { AnalysisSection } from "../components/AnalysisSection";

function renderSection(): string {
  return renderToStaticMarkup(React.createElement(AnalysisSection));
}

test("renders the scenario form in its idle state", () => {
  const markup = renderSection();
  assert.ok(markup.includes("Scenario description"));
  assert.ok(markup.includes("Regulatory question"));
  assert.ok(markup.includes("Analyze"));
  assert.ok(markup.includes("Try the demo scenario"));
});
