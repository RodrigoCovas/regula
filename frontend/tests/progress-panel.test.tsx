import assert from "node:assert/strict";
import { test } from "node:test";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { ProgressPanel } from "../components/ProgressPanel";
import type { ProgressSnapshot } from "../lib/progress";

function renderPanel(snapshots: ProgressSnapshot[]): string {
  return renderToStaticMarkup(React.createElement(ProgressPanel, { snapshots }));
}

const plannerSnapshot: ProgressSnapshot = {
  request_id: "run-1",
  phase: "planner",
  message: "decomposing the Regulatory question into research targets",
  transitions: [
    {
      phase: "planner",
      message: "decomposing the Regulatory question into research targets",
      elapsed_ms: 410,
    },
  ],
};

const researcherSnapshot: ProgressSnapshot = {
  request_id: "run-1",
  phase: "researcher",
  message: "retrieving Evidence from the Corpus",
  transitions: [
    plannerSnapshot.transitions[0],
    {
      phase: "researcher",
      message: "retrieving Evidence from the Corpus",
      elapsed_ms: 2310,
    },
  ],
};

test("renders a starting message when no snapshots have arrived", () => {
  const markup = renderPanel([]);
  assert.ok(markup.includes("Starting analysis"));
});

test("renders the latest phase and message", () => {
  const markup = renderPanel([plannerSnapshot]);
  assert.ok(markup.includes("Planner"));
  assert.ok(markup.includes("decomposing the Regulatory question into research targets"));
});

test("lists every received transition with its elapsed time", () => {
  const markup = renderPanel([plannerSnapshot, researcherSnapshot]);
  assert.ok(markup.includes("Planner"));
  assert.ok(markup.includes("Researcher"));
  assert.ok(markup.includes("0.4s"));
  assert.ok(markup.includes("2.3s"));
});

test("renders the latest phase when it differs from the last transition", () => {
  const snapshot: ProgressSnapshot = {
    ...researcherSnapshot,
    phase: "verifier",
    message: "checking drafted Claims",
  };
  const markup = renderPanel([snapshot]);
  assert.ok(markup.includes("Verifier: checking drafted Claims"));
});
