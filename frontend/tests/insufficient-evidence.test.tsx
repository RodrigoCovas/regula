import assert from "node:assert/strict";
import { test } from "node:test";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { AnswerSurface } from "../components/AnswerSurface";
import { insufficientEvidenceResponse } from "../lib/fixtures";
import { visibleMarkup } from "./escape";

const markup = renderToStaticMarkup(
  React.createElement(AnswerSurface, {
    response: insufficientEvidenceResponse,
  }),
);

test("empty Findings render the dedicated Insufficient-evidence panel", () => {
  assert.ok(markup.includes("Insufficient evidence"));
  assert.ok(markup.includes(visibleMarkup(insufficientEvidenceResponse.trace.summary)));
});

test("renders no bare Findings list when Findings are empty", () => {
  assert.ok(!markup.includes("Findings ("));
});

test("renders the insufficient-evidence Known limitation", () => {
  for (const limitation of insufficientEvidenceResponse.known_limitations) {
    assert.ok(markup.includes(limitation));
  }
});

test("renders the narrowing Actions under How to proceed", () => {
  assert.ok(markup.includes("How to proceed"));
  for (const action of insufficientEvidenceResponse.answer.actions) {
    assert.ok(markup.includes(visibleMarkup(action)));
  }
});

test("renders the collapsed detailed trace behind the Insufficient-evidence panel", () => {
  assert.ok(markup.includes("<details"), "expected a collapsible expander");
  assert.ok(
    !markup.includes("<details open"),
    "detailed trace must be collapsed by default",
  );
  assert.ok(markup.includes("Detailed execution trace (4 steps)"));
  assert.ok(markup.includes(">planner<"));
  assert.ok(markup.includes(">proposer<"));
});
