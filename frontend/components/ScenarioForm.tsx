"use client";

import { useEffect, useReducer } from "react";
import type { FormEvent } from "react";
import type { ScenarioInput } from "../lib/contract";
import { fetchReadiness, ReadinessUnavailableError } from "../lib/readiness";
import {
  canSubmit,
  initialScenarioFormState,
  readinessGateOf,
  scenarioFormReducer,
  submissionOf,
} from "../lib/scenario-form";
import { DerivedScenarioPreview } from "./DerivedScenarioPreview";
import { ModeToggle } from "./ModeToggle";
import { ReadinessChecklist } from "./ReadinessChecklist";

export function ScenarioForm({
  onSubmit,
}: {
  onSubmit?: (input: ScenarioInput) => void;
}) {
  const [state, dispatch] = useReducer(
    scenarioFormReducer,
    initialScenarioFormState,
  );

  // Readiness is checked fresh on every Live selection (issue #48): the
  // gate must reflect the backend's state now, never a stale check. Demo
  // mode never probes anything.
  useEffect(() => {
    if (state.mode !== "live") {
      return;
    }
    let cancelled = false;
    fetchReadiness()
      .then((readiness) => {
        if (!cancelled) {
          dispatch({ type: "readiness-known", readiness });
        }
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          dispatch({
            type: "readiness-unavailable",
            message:
              error instanceof ReadinessUnavailableError
                ? error.message
                : "Live-mode readiness could not be checked.",
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [state.mode]);

  const gate = readinessGateOf(state);
  const submitEnabled = canSubmit(state);
  // Firefox can restore a button's dynamic disabled state across reloads.
  const submitButtonAttributes = { autoComplete: "off" };

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!submitEnabled) {
      return;
    }
    onSubmit?.(submissionOf(state));
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <ModeToggle
        mode={state.mode}
        onChange={(mode) => dispatch({ type: "mode-changed", mode })}
      />

      {gate.kind === "checking" ? (
        <p role="status" className="text-sm text-slate-500">
          Checking Live-mode readiness…
        </p>
      ) : null}

      {gate.kind === "incomplete" ? <ReadinessChecklist missing={gate.missing} /> : null}

      {gate.kind === "unavailable" ? (
        <p
          role="alert"
          className="rounded-md border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800"
        >
          {gate.message} Live submission stays blocked until Readiness is
          confirmed.
        </p>
      ) : null}

      <div className="space-y-1">
        <label
          htmlFor="scenario-description"
          className="text-sm font-medium text-slate-700"
        >
          Scenario description
        </label>
        <textarea
          id="scenario-description"
          value={state.description}
          onChange={(event) =>
            dispatch({ type: "description-changed", description: event.target.value })
          }
          rows={4}
          placeholder="Describe your company, product, and jurisdiction"
          className="w-full rounded-md border border-slate-300 p-3 text-sm"
        />
        <p className="text-xs text-slate-500">
          Demo mode answers only the canonical demo scenario — use the button
          below to fill this form. Your own scenario runs in Live mode, which
          needs the Live prerequisites.
        </p>
      </div>

      <div className="space-y-1">
        <label
          htmlFor="scenario-question"
          className="text-sm font-medium text-slate-700"
        >
          Regulatory question
        </label>
        <input
          id="scenario-question"
          type="text"
          value={state.question}
          onChange={(event) =>
            dispatch({ type: "question-changed", question: event.target.value })
          }
          placeholder="What regulations apply?"
          className="w-full rounded-md border border-slate-300 p-3 text-sm"
        />
      </div>

      <DerivedScenarioPreview description={state.description} />

      <div className="flex gap-3">
        <button
          {...submitButtonAttributes}
          type="submit"
          disabled={!submitEnabled}
          className="rounded-md bg-slate-900 px-4 py-2 text-sm font-semibold text-white disabled:opacity-50"
        >
          Analyze
        </button>
        <button
          type="button"
          onClick={() => dispatch({ type: "demo-scenario-filled" })}
          className="rounded-md border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-700"
        >
          Try the demo scenario
        </button>
      </div>
    </form>
  );
}
