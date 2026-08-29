import type { Mode, ScenarioInput } from "./contract";
import type { Readiness, ReadinessGate } from "./readiness";
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
  // The last completed readiness check, null while none is in force —
  // always null in Demo mode and reset on every mode change.
  readiness: Readiness | null;
  readinessError: string | null;
}

export const initialScenarioFormState: ScenarioFormState = {
  description: "",
  question: "",
  mode: "demo",
  readiness: null,
  readinessError: null,
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
        readiness: null,
        readinessError: null,
      };
    case "demo-scenario-filled":
      return {
        ...state,
        description: demoScenarioInput.scenario.description,
        question: demoScenarioInput.question,
      };
    case "readiness-known":
      return { ...state, readiness: action.readiness, readinessError: null };
    case "readiness-unavailable":
      return { ...state, readinessError: action.message };
  }
}

export function readinessGateOf(state: ScenarioFormState): ReadinessGate {
  // An unavailable check outranks a completed one: the message says why
  // Live stays blocked. The rest — Demo never gates, Live blocks until a
  // fresh check reads ready — belongs to liveReadinessGate alone.
  if (state.readinessError !== null) {
    return { kind: "unavailable", message: state.readinessError };
  }
  return liveReadinessGate(state.mode, state.readiness);
}

export function canSubmit(state: ScenarioFormState): boolean {
  const fieldsFilled = Boolean(state.description.trim() && state.question.trim());
  return fieldsFilled && !readinessBlocks(readinessGateOf(state));
}

export function submissionOf(state: ScenarioFormState): ScenarioInput {
  return buildScenarioInput(state.description, state.question, state.mode);
}
