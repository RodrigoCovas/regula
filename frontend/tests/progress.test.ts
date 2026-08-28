import assert from "node:assert/strict";
import { test } from "node:test";
import { notAvailableResponse } from "../lib/fixtures";
import {
  formatElapsed,
  isAnalyzeResponse,
  isProgressSnapshot,
  phaseLabel,
} from "../lib/progress";

const snapshot = {
  request_id: "run-1",
  phase: "researcher",
  message: "retrieving Evidence for the research targets",
  transitions: [
    {
      phase: "planner",
      message: "decomposing the Regulatory question into research targets",
      elapsed_ms: 410,
    },
    {
      phase: "researcher",
      message: "retrieving Evidence for the research targets",
      elapsed_ms: 2310,
    },
  ],
};

test("recognizes a progress snapshot by its transitions field", () => {
  assert.equal(isProgressSnapshot(snapshot), true);
});

test("rejects the Not-available reply the endpoint serves for unknown ids", () => {
  assert.equal(isProgressSnapshot(notAvailableResponse), false);
});

test("rejects arbitrary non-snapshot objects", () => {
  assert.equal(isProgressSnapshot({ phase: "planner", message: "x" }), false);
  assert.equal(isProgressSnapshot(null), false);
});

test("recognizes an AnalyzeResponse by its sibling fields", () => {
  assert.equal(isAnalyzeResponse(notAvailableResponse), true);
});

test("rejects partial or non-analyze shapes", () => {
  assert.equal(isAnalyzeResponse({ trace: { workflow: "noop" } }), false);
  assert.equal(isAnalyzeResponse({ answer: {}, trace: {}, known_limitations: [] }), false);
  assert.equal(isAnalyzeResponse(null), false);
});

test("phaseLabel maps the workflow-agent phases to display names", () => {
  assert.equal(phaseLabel("planner"), "Planner");
  assert.equal(phaseLabel("researcher"), "Researcher");
  assert.equal(phaseLabel("verifier"), "Verifier");
  assert.equal(phaseLabel("proposer"), "Proposer");
});

test("phaseLabel falls back to the raw phase for unknown phases", () => {
  assert.equal(phaseLabel("sidekick" as never), "sidekick");
});

test("formatElapsed renders sub-second durations with one decimal", () => {
  assert.equal(formatElapsed(410), "0.4s");
});

test("formatElapsed renders second-scale durations without decimals", () => {
  assert.equal(formatElapsed(2310), "2.3s");
});

test("formatElapsed renders minute-scale durations as minutes and seconds", () => {
  assert.equal(formatElapsed(65_000), "1m 05s");
  assert.equal(formatElapsed(123_000), "2m 03s");
});

test("recognizes a progress snapshot with a terminal error field", () => {
  const failedSnapshot = {
    request_id: "run-1",
    phase: "planner",
    message: "decomposing",
    transitions: [
      { phase: "planner", message: "decomposing", elapsed_ms: 1000 },
    ],
    error: "the LLM provider rejected the request",
  };
  assert.equal(isProgressSnapshot(failedSnapshot), true);
});
