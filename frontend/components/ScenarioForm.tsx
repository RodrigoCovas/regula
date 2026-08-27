"use client";

import { useState } from "react";
import type { FormEvent } from "react";
import type { ScenarioInput } from "../lib/contract";
import {
  buildScenarioInput,
  demoScenarioInput,
  deriveScenarioId,
} from "../lib/scenario-id";

export function ScenarioForm({
  initialDescription = "",
  onSubmit,
}: {
  initialDescription?: string;
  onSubmit?: (input: ScenarioInput) => void;
}) {
  const [description, setDescription] = useState(initialDescription);
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
    onSubmit?.(demoScenarioInput);
  }

  const derivedId = deriveScenarioId(description);

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

      <div className="rounded-md border border-slate-200 bg-slate-50 p-4 text-sm">
        <p className="font-medium text-slate-700">Derived scenario</p>
        {description ? (
          <dl className="mt-2 space-y-1">
            <div className="flex gap-2">
              <dt className="text-slate-500">Title</dt>
              <dd className="break-words">{description}</dd>
            </div>
            <div className="flex gap-2">
              <dt className="text-slate-500">Id</dt>
              <dd className="break-all font-mono">{derivedId}</dd>
            </div>
          </dl>
        ) : (
          <p className="text-slate-500">
            Title and id derive from the description as you type.
          </p>
        )}
      </div>

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
