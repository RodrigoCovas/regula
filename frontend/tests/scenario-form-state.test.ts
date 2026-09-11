import assert from "node:assert/strict";
import { test } from "node:test";
import {
  canSubmit,
  initialScenarioFormState,
  readinessGateOf,
  scenarioFormReducer,
  submissionOf,
  type ScenarioFormAction,
  type ScenarioFormState,
} from "../lib/scenario-form";
import { demoScenarioInput, deriveScenarioId } from "../lib/scenario-id";
import { KEY_MISSING_READINESS, READY_READINESS } from "./readiness-states";

function reduced(actions: ScenarioFormAction[]): ScenarioFormState {
  return actions.reduce(scenarioFormReducer, initialScenarioFormState);
}

function filledLive(actions: ScenarioFormAction[] = []): ScenarioFormState {
  return reduced([
    { type: "description-changed", description: "A Spanish fintech startup" },
    { type: "question-changed", question: "What regulations apply?" },
    { type: "mode-changed", mode: "live" },
    ...actions,
  ]);
}

test("the toggle defaults to Demo on every load", () => {
  assert.equal(initialScenarioFormState.mode, "demo");
  assert.deepEqual(readinessGateOf(initialScenarioFormState), { kind: "idle" });
});

test("selecting Live blocks submission until readiness is known", () => {
  const state = filledLive();
  assert.deepEqual(readinessGateOf(state), { kind: "checking" });
  assert.equal(canSubmit(state), false);
});

test("incomplete readiness blocks submission and names the missing prerequisites", () => {
  const state = filledLive([{ type: "readiness-known", readiness: KEY_MISSING_READINESS }]);
  const gate = readinessGateOf(state);
  assert.equal(gate.kind, "incomplete");
  assert.deepEqual(
    gate.kind === "incomplete" ? gate.missing.map((item) => item.key) : [],
    ["api_key_set"],
  );
  assert.equal(canSubmit(state), false);
});

test("a ready store unblocks Live submission for a filled form", () => {
  const state = filledLive([{ type: "readiness-known", readiness: READY_READINESS }]);
  assert.deepEqual(readinessGateOf(state), { kind: "ready" });
  assert.equal(canSubmit(state), true);
});

test("Live stays blocked while the readiness check is unavailable", () => {
  const state = filledLive([
    { type: "readiness-unavailable", message: "the backend could not be reached" },
  ]);
  assert.deepEqual(readinessGateOf(state), {
    kind: "unavailable",
    message: "the backend could not be reached",
  });
  assert.equal(canSubmit(state), false);
});

test("Demo mode never gates submission, whatever the last check said", () => {
  const state = reduced([
    { type: "description-changed", description: "A Spanish fintech startup" },
    { type: "question-changed", question: "What regulations apply?" },
    { type: "mode-changed", mode: "live" },
    { type: "readiness-known", readiness: KEY_MISSING_READINESS },
    { type: "mode-changed", mode: "demo" },
  ]);
  assert.deepEqual(readinessGateOf(state), { kind: "idle" });
  assert.equal(canSubmit(state), true);
});

test("a stale readiness error can never gate Demo mode — even directly constructed", () => {
  // Constructed straight past the reducer (whose mode-changed clears the
  // error) to pin the gate's own invariant: Demo never gates.
  const state: ScenarioFormState = {
    ...initialScenarioFormState,
    readiness: { kind: "unavailable", message: "the backend could not be reached" },
  };
  assert.deepEqual(readinessGateOf(state), { kind: "idle" });
  assert.equal(
    canSubmit({
      ...state,
      description: "A Spanish fintech startup",
      question: "What regulations apply?",
    }),
    true,
  );
});

test("switching back to Live starts a fresh check: a stale ready result never unblocks", () => {
  const state = filledLive([
    { type: "readiness-known", readiness: READY_READINESS },
    { type: "mode-changed", mode: "demo" },
    { type: "mode-changed", mode: "live" },
  ]);
  assert.deepEqual(readinessGateOf(state), { kind: "checking" });
  assert.equal(canSubmit(state), false);
});

test("an empty form cannot submit even when Live is ready", () => {
  const state = reduced([
    { type: "mode-changed", mode: "live" },
    { type: "readiness-known", readiness: READY_READINESS },
  ]);
  assert.equal(canSubmit(state), false);
});

test("the demo button fills the form without changing the selected mode", () => {
  const liveThenFilled = filledLive([
    { type: "readiness-known", readiness: READY_READINESS },
    { type: "demo-scenario-filled" },
  ]);
  assert.equal(liveThenFilled.mode, "live");
  assert.equal(liveThenFilled.description, demoScenarioInput.scenario.description);
  assert.equal(liveThenFilled.question, demoScenarioInput.question);

  const demoThenFilled = reduced([{ type: "demo-scenario-filled" }]);
  assert.equal(demoThenFilled.mode, "demo");
  assert.equal(demoThenFilled.description, demoScenarioInput.scenario.description);
  assert.equal(demoThenFilled.question, demoScenarioInput.question);
});

test("the demo fill keeps the readiness state untouched", () => {
  const state = filledLive([
    { type: "readiness-known", readiness: KEY_MISSING_READINESS },
    { type: "demo-scenario-filled" },
  ]);
  const gate = readinessGateOf(state);
  assert.deepEqual(
    gate.kind === "incomplete" ? gate.missing.map((item) => item.key) : [],
    ["api_key_set"],
  );
});

test("the submission carries the selected mode explicitly", () => {
  const liveInput = submissionOf(
    filledLive([{ type: "readiness-known", readiness: READY_READINESS }]),
  );
  assert.equal(liveInput.mode, "live");
  assert.equal(liveInput.question, "What regulations apply?");

  const demoInput = submissionOf(
    reduced([
      { type: "description-changed", description: "A Spanish fintech startup" },
      { type: "question-changed", question: "What regulations apply?" },
    ]),
  );
  assert.equal(demoInput.mode, "demo");
});

test("the demo fill submits the canonical scenario id, which no description derives to", () => {
  // The canonical id is an authored name, not a derived one (ADR-0005):
  // deriveScenarioId always appends a hash suffix, so the submission must
  // carry the pinned id or the demo trigger never fires.
  const input = submissionOf(reduced([{ type: "demo-scenario-filled" }]));
  assert.equal(input.scenario.id, demoScenarioInput.scenario.id);
  assert.equal(
    input.scenario.id,
    "bank-cloud-outage",
  );
  assert.notEqual(input.scenario.id, deriveScenarioId(input.scenario.description));
  assert.equal(input.scenario.description, demoScenarioInput.scenario.description);
  assert.equal(input.question, demoScenarioInput.question);
  assert.equal(input.mode, "demo");
});

test("editing the question after the demo fill keeps the canonical scenario (ADR-0006)", () => {
  const input = submissionOf(
    reduced([
      { type: "demo-scenario-filled" },
      { type: "question-changed", question: "Which DORA articles bind the bank?" },
    ]),
  );
  assert.equal(input.scenario.id, demoScenarioInput.scenario.id);
  assert.equal(input.scenario.description, demoScenarioInput.scenario.description);
  assert.equal(input.question, "Which DORA articles bind the bank?");
});

test("editing the description abandons the demo fill: the id re-derives", () => {
  const input = submissionOf(
    reduced([
      { type: "demo-scenario-filled" },
      { type: "description-changed", description: "A Spanish fintech startup" },
    ]),
  );
  assert.equal(input.scenario.id, deriveScenarioId("A Spanish fintech startup"));
  assert.notEqual(input.scenario.id, demoScenarioInput.scenario.id);
  assert.equal(input.scenario.description, "A Spanish fintech startup");
});

test("switching mode after the demo fill keeps the canonical scenario and carries the new mode", () => {
  const input = submissionOf(
    reduced([
      { type: "demo-scenario-filled" },
      { type: "mode-changed", mode: "live" },
    ]),
  );
  assert.equal(input.scenario.id, demoScenarioInput.scenario.id);
  assert.equal(input.mode, "live");
  assert.equal(input.question, demoScenarioInput.question);
});

test("typing the canonical description by hand never submits the canonical id (ADR-0005)", () => {
  const input = submissionOf(
    reduced([
      { type: "description-changed", description: demoScenarioInput.scenario.description },
      { type: "question-changed", question: demoScenarioInput.question },
    ]),
  );
  assert.notEqual(input.scenario.id, demoScenarioInput.scenario.id);
  assert.equal(input.scenario.id, deriveScenarioId(demoScenarioInput.scenario.description));
});
