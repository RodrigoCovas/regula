import assert from "node:assert/strict";
import { test } from "node:test";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { ModeToggle } from "../components/ModeToggle";

function renderToggle(mode: "demo" | "live"): string {
  return renderToStaticMarkup(
    React.createElement(ModeToggle, { mode, onChange: () => {} }),
  );
}

test("renders Demo and Live as a radio pair", () => {
  const markup = renderToggle("demo");
  assert.ok(markup.includes('value="demo"'));
  assert.ok(markup.includes('value="live"'));
  assert.ok(markup.includes("Demo"));
  assert.ok(markup.includes("Live"));
});

test("defaults to Demo selected on every load", () => {
  const markup = renderToggle("demo");
  const demoInput = markup.match(/<input[^>]*value="demo"[^>]*>/)?.[0] ?? "";
  const liveInput = markup.match(/<input[^>]*value="live"[^>]*>/)?.[0] ?? "";
  assert.match(demoInput, /checked/i);
  assert.doesNotMatch(liveInput, /checked/i);
});

test("reflects Live when it is the selected mode", () => {
  const markup = renderToggle("live");
  const demoInput = markup.match(/<input[^>]*value="demo"[^>]*>/)?.[0] ?? "";
  const liveInput = markup.match(/<input[^>]*value="live"[^>]*>/)?.[0] ?? "";
  assert.match(liveInput, /checked/i);
  assert.doesNotMatch(demoInput, /checked/i);
});

test("the radios share one group name so only one mode can be selected", () => {
  const markup = renderToggle("demo");
  assert.equal((markup.match(/name="mode"/g) ?? []).length, 2);
});
