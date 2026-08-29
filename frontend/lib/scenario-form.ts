import type { Mode, ScenarioInput } from "./contract";
import type { Readiness, ReadinessGate, ReadinessResult } from "./readiness";
import { liveReadinessGate, readinessBlocks } from "./readiness";
import { buildScenarioInput, demoScenarioInput } from "./scenario-id";

// The Demo/Live form's state (issue #48): a thin reducer so the toggle
// default, the readiness gating, and the demo button's fill-only behaviour
// stay pure and testable without a DOM. The component renders this state
// and dispatches these actions; nothing here touches storage, so the mode
// never persists across reloads.

export interface ScenarioFormState {
  description: string;
  question: string;
  mode: Mode;
  // What the latest readiness check attempt left behind — reset on every
  // mode change, so a Live selection always re-checks.
  readiness: ReadinessResult;
}

export const initialScenarioFormState: ScenarioFormState = {
  description: "",
  question: "",
  mode: "demo",
  readiness: { kind: "none" },
};

export type ScenarioFormAction =
  | { type: "description-changed"; description: string }
  | { type: "question-changed"; question: string }
  | { type: "mode-changed"; mode: Mode }
  // The demo button: populate the form only — never the selected mode,
  // never the readiness state.
  | { type: "demo-scenario-filled" }
  | { type: "readiness-known"; readiness: Readiness }
  | { type: "readiness-unavailable"; message: string };

export function scenarioFormReducer(
  state: ScenarioFormState,
  action: ScenarioFormAction,
): ScenarioFormState {
  switch (action.type) {
    case "description-changed":
      return { ...state, description: action.description };
    case "question-changed":
      return { ...state, question: action.question };
    case "mode-changed":
      // Every Live selection re-checks Readiness: a previous result must
      // never unblock a run on stale state.
      return {
        ...state,
        mode: action.mode,
        readiness: { kind: "none" },
      };
    case "demo-scenario-filled":
      return {
        ...state,
        description: demoScenarioInput.scenario.description,
        question: demoScenarioInput.question,
      };
    case "readiness-known":
      return { ...state, readiness: { kind: "known", readiness: action.readiness } };
    case "readiness-unavailable":
      return { ...state, readiness: { kind: "unavailable", message: action.message } };
  }
}

export function readinessGateOf(state: ScenarioFormState): ReadinessGate {
  // One delegate: liveReadinessGate owns the whole mapping — Demo never
  // gates (the mode check leads), and an unavailable check outranks a
  // completed one only inside Live mode.
  return liveReadinessGate(state.mode, state.readiness);
}

export function canSubmit(state: ScenarioFormState): boolean {
  const fieldsFilled = Boolean(state.description.trim() && state.question.trim());
  return fieldsFilled && !readinessBlocks(readinessGateOf(state));
}

export function submissionOf(state: ScenarioFormState): ScenarioInput {
  return buildScenarioInput(state.description, state.question, state.mode);
}
