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
  // Whether the form still carries the demo button's canonical fill (ADR-0005):
  // the canonical id is authored, never derived, so a filled form submits it
  // until a description edit re-derives the id. A question edit keeps the
  // scenario (ADR-0006); a mode change is orthogonal to the fill.
  demoScenarioFilled: boolean;
  // What the latest readiness check attempt left behind — reset on every
  // mode change, so a Live selection always re-checks.
  readiness: ReadinessResult;
}

export const initialScenarioFormState: ScenarioFormState = {
  description: "",
  question: "",
  mode: "demo",
  demoScenarioFilled: false,
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
      // A description edit abandons the demo fill: the edited text is no
      // longer the canonical Scenario, so the id re-derives and Demo mode
      // gets the honest Not-available response (ADR-0005).
      return {
        ...state,
        description: action.description,
        demoScenarioFilled: false,
      };
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
        demoScenarioFilled: true,
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
  // The current demo fill submits the pinned canonical Scenario (ADR-0005)
  // with the current question and mode; every other form content derives
  // its id.
  if (state.demoScenarioFilled) {
    return {
      scenario: demoScenarioInput.scenario,
      question: state.question,
      mode: state.mode,
    };
  }
  return buildScenarioInput(state.description, state.question, state.mode);
}
