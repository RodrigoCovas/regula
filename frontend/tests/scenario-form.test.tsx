import assert from "node:assert/strict";
import { test } from "node:test";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { ScenarioForm } from "../components/ScenarioForm";

function render(initialDescription = ""): string {
  return renderToStaticMarkup(
    React.createElement(ScenarioForm, { initialDescription }),
  );
}

test("takes only a description and a question — no id or title fields", () => {
  const markup = render();
  assert.ok(markup.includes("Scenario description"));
  assert.ok(markup.includes("Regulatory question"));
  assert.equal((markup.match(/<textarea/g) ?? []).length, 1);
  assert.equal((markup.match(/<input/g) ?? []).length, 1);
});

test("renders the Analyze and demo buttons", () => {
  const markup = render();
  assert.ok(markup.includes(">Analyze<"));
  assert.ok(markup.includes("Try the demo scenario"));
});

test("never surfaces the canonical demo id anywhere in the UI", () => {
  const markup = render();
  assert.ok(!markup.includes("spanish-fintech"));
});

test("shows a live preview of the derived title and id", () => {
  const markup = render("the Spanish fintech lending company");
  assert.ok(markup.includes("the Spanish fintech lending company"));
  assert.ok(markup.includes("spanish-fintech-lending-company-"));
});

test("shows a placeholder preview while the description is empty", () => {
  const markup = render();
  assert.ok(markup.includes("Derived scenario"));
  assert.ok(markup.includes("Title and id derive from the description as you type."));
});
