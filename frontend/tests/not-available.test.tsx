import assert from "node:assert/strict";
import { test } from "node:test";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { AnswerSurface } from "../components/AnswerSurface";
import { notAvailableResponse } from "../lib/fixtures";
import { visibleMarkup } from "./escape";

function render(response = notAvailableResponse): string {
  return renderToStaticMarkup(
    React.createElement(AnswerSurface, { response: response }),
  );
}

test("Not-available responses render as the named domain outcome", () => {
  const markup = render();
  assert.ok(markup.includes("Not available"));
  assert.ok(markup.includes(visibleMarkup(notAvailableResponse.trace.summary)));
});

test("never renders a Not-available response as a generic error", () => {
  const markup = render();
  assert.ok(!markup.includes("Error"));
  assert.ok(!markup.includes("error"));
});

test("renders the recovery Actions under How to proceed", () => {
  const markup = render();
  assert.ok(markup.includes("How to proceed"));
  for (const action of notAvailableResponse.answer.actions) {
    assert.ok(markup.includes(visibleMarkup(action)));
  }
});

test("renders Known limitations", () => {
  const markup = render();
  for (const limitation of notAvailableResponse.known_limitations) {
    assert.ok(markup.includes(limitation));
  }
});

test("renders no Findings and no Insufficient-evidence panel", () => {
  const markup = render();
  assert.ok(!markup.includes("Findings ("));
  assert.ok(!markup.includes("Insufficient evidence"));
});

test("renders no empty detailed-trace expander", () => {
  const markup = render();
  assert.ok(!markup.includes("<details"));
});
