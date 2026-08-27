import type { AnalyzeResponse } from "./contract";

export type WorkflowPhase = "planner" | "researcher" | "verifier" | "proposer";

export interface ProgressTransition {
  phase: WorkflowPhase;
  message: string;
  elapsed_ms: number;
}

export interface ProgressSnapshot {
  request_id: string;
  phase: WorkflowPhase | null;
  message: string | null;
  transitions: ProgressTransition[];
}

const PHASE_LABELS: Record<WorkflowPhase, string> = {
  planner: "Planner",
  researcher: "Researcher",
  verifier: "Verifier",
  proposer: "Proposer",
};

export function phaseLabel(phase: WorkflowPhase): string {
  return PHASE_LABELS[phase] ?? phase;
}

export function isProgressSnapshot(json: unknown): json is ProgressSnapshot {
  if (typeof json !== "object" || json === null) {
    return false;
  }
  const candidate = json as Record<string, unknown>;
  return (
    typeof candidate.request_id === "string" &&
    Array.isArray(candidate.transitions)
  );
}

export function isAnalyzeResponse(json: unknown): json is AnalyzeResponse {
  if (typeof json !== "object" || json === null) {
    return false;
  }
  const candidate = json as Record<string, unknown>;
  const answer = candidate.answer as Record<string, unknown> | null;
  const trace = candidate.trace as Record<string, unknown> | null;
  return (
    typeof answer === "object" &&
    answer !== null &&
    Array.isArray(answer.findings) &&
    Array.isArray(answer.actions) &&
    Array.isArray(answer.citations) &&
    typeof trace === "object" &&
    trace !== null &&
    typeof trace.workflow === "string" &&
    Array.isArray(candidate.known_limitations)
  );
}

export function formatElapsed(elapsedMs: number): string {
  const totalSeconds = Math.round(elapsedMs / 1000);
  if (totalSeconds < 60) {
    return `${(elapsedMs / 1000).toFixed(1)}s`;
  }
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = String(totalSeconds % 60).padStart(2, "0");
  return `${minutes}m ${seconds}s`;
}
