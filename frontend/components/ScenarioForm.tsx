"use client";

import { useState } from "react";
import type { FormEvent } from "react";
import type { ScenarioInput } from "../lib/contract";
import { buildScenarioInput, demoScenarioInput } from "../lib/scenario-id";
import { DerivedScenarioPreview } from "./DerivedScenarioPreview";

export function ScenarioForm({
  onSubmit,
}: {
  onSubmit?: (input: ScenarioInput) => void;
}) {
  const [description, setDescription] = useState("");
  const [question, setQuestion] = useState("");

  const canAnalyze = Boolean(description.trim() && question.trim());

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canAnalyze) {
      return;
    }
    onSubmit?.(buildScenarioInput(description, question));
  }

  function handleDemoSubmit() {
    setDescription(demoScenarioInput.scenario.description);
    setQuestion(demoScenarioInput.question);
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="space-y-1">
        <label
          htmlFor="scenario-description"
          className="text-sm font-medium text-slate-700"
        >
          Scenario description
        </label>
        <textarea
          id="scenario-description"
          value={description}
          onChange={(event) => setDescription(event.target.value)}
          rows={4}
          placeholder="Describe your company, product, and jurisdiction"
          className="w-full rounded-md border border-slate-300 p-3 text-sm"
        />
        <p className="text-xs text-slate-500">
          The demo scenario is deterministic and requires no API key. Editing
          this or writing your own scenario requires an LLM API key.
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
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="What regulations apply?"
          className="w-full rounded-md border border-slate-300 p-3 text-sm"
        />
      </div>

      <DerivedScenarioPreview description={description} />

      <div className="flex gap-3">
        <button
          type="submit"
          disabled={!canAnalyze}
          className="rounded-md bg-slate-900 px-4 py-2 text-sm font-semibold text-white disabled:opacity-50"
        >
          Analyze
        </button>
        <button
          type="button"
          onClick={handleDemoSubmit}
          className="rounded-md border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-700"
        >
          Try the demo scenario
        </button>
      </div>
    </form>
  );
}
