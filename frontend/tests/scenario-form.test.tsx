import assert from "node:assert/strict";
import { test } from "node:test";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { DerivedScenarioPreview } from "../components/DerivedScenarioPreview";
import { ScenarioForm } from "../components/ScenarioForm";
import { demoScenarioInput } from "../lib/scenario-id";

function renderForm(): string {
  return renderToStaticMarkup(React.createElement(ScenarioForm));
}

function renderPreview(description: string): string {
  return renderToStaticMarkup(
    React.createElement(DerivedScenarioPreview, { description }),
  );
}

test("takes only a description and a question — no id or title fields", () => {
  const markup = renderForm();
  assert.ok(markup.includes("Scenario description"));
  assert.ok(markup.includes("Regulatory question"));
  assert.equal((markup.match(/<textarea/g) ?? []).length, 1);
  assert.equal((markup.match(/<input[^>]*type="text"/g) ?? []).length, 1);
});

test("the mode toggle defaults to Demo on every load (issue #48)", () => {
  const markup = renderForm();
  const demoInput = markup.match(/<input[^>]*value="demo"[^>]*>/)?.[0] ?? "";
  assert.match(demoInput, /checked/i);
});

test("renders no readiness checklist while Demo is selected", () => {
  const markup = renderForm();
  assert.ok(!markup.includes("not ready"));
  assert.ok(!markup.includes("docker compose exec"));
});

test("renders the Analyze and demo buttons", () => {
  const markup = renderForm();
  assert.ok(markup.includes(">Analyze<"));
  assert.ok(markup.includes("Try the demo scenario"));
});

test("prevents Firefox from restoring the submit button state across reloads", () => {
  const markup = renderForm();
  const submitButton = markup.match(/<button[^>]*type="submit"[^>]*>/)?.[0];
  assert.match(submitButton ?? "", /autocomplete="off"/i);
});

test("never surfaces the canonical demo id anywhere in the UI", () => {
  const markup = renderForm();
  assert.ok(!markup.includes(demoScenarioInput.scenario.id));
});

test("the preview shows the derived title and id for a description", () => {
  const markup = renderPreview("the Spanish fintech lending company");
  assert.ok(markup.includes("Spanish Fintech Lending Company"));
  assert.ok(markup.includes("spanish-fintech-lending-company-"));
});

test("the preview shows a placeholder while the description is empty", () => {
  const markup = renderPreview("");
  assert.ok(markup.includes("Derived scenario"));
  assert.ok(
    markup.includes("Title and id derive from the description as you type."),
  );
  assert.ok(!markup.includes("spanish-fintech-lending"));
});
