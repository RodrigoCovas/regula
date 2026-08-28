"use client";

import { useState } from "react";
import type { AnalyzeResponse, ScenarioInput } from "../lib/contract";
import { runAnalysis, AnalyzeFailure } from "../lib/analyze-client";
import type { ProgressSnapshot } from "../lib/progress";
import { AnswerSurface } from "./AnswerSurface";
import { ProgressPanel } from "./ProgressPanel";
import { ScenarioForm } from "./ScenarioForm";

export type AnalysisStatus =
  | { kind: "idle" }
  | { kind: "running"; snapshots: ProgressSnapshot[]; input: ScenarioInput }
  | { kind: "done"; response: AnalyzeResponse; input: ScenarioInput }
  | { kind: "error"; message: string };

export function AnalysisSection() {
  const [status, setStatus] = useState<AnalysisStatus>({ kind: "idle" });

  async function handleSubmit(input: ScenarioInput) {
    setStatus({ kind: "running", snapshots: [], input });
    try {
      const response = await runAnalysis(input, {
        onProgress: (snapshot) => {
          setStatus((current) => {
            if (current.kind !== "running") {
              return current;
            }
            const lastSnapshot = current.snapshots[current.snapshots.length - 1];
            const lastPhase = lastSnapshot?.phase;
            if (lastPhase === snapshot.phase) {
              return current;
            }
            return { kind: "running", snapshots: [...current.snapshots, snapshot], input };
          });
        },
      });
      setStatus({ kind: "done", response, input });
    } catch (error) {
      const message =
        error instanceof AnalyzeFailure
          ? error.message
          : error instanceof Error
            ? error.message
            : "Analysis failed";
      setStatus({ kind: "error", message });
    }
  }

  return (
    <div className="space-y-8">
      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-slate-800">
          Analyze a scenario
        </h2>
        <ScenarioForm onSubmit={handleSubmit} />
      </section>

      {status.kind === "running" ? (
        <ProgressPanel snapshots={status.snapshots} />
      ) : null}

      {status.kind === "error" ? (
        <div
          role="alert"
          className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800"
        >
          {status.message}
        </div>
      ) : null}

      {status.kind === "done" ? (
        <section className="space-y-4">
          <h2 className="text-lg font-semibold text-slate-800">Answer</h2>
          {status.input.scenario.title ? (
            <p className="text-sm text-slate-600">
              {status.input.scenario.title}
            </p>
          ) : null}
          <AnswerSurface response={status.response} />
        </section>
      ) : null}
    </div>
  );
}
