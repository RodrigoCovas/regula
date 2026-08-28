import type { AnalyzeResponse, ScenarioInput } from "./contract";
import type { ProgressSnapshot } from "./progress";
import { isAnalyzeResponse, isProgressSnapshot } from "./progress";

const REQUEST_ID_HEADER = "x-request-id";
const DEFAULT_POLL_INTERVAL_MS = 1000;
const DEFAULT_TIMEOUT_MS = 900000;

export class AnalyzeFailure extends Error {}

export type TimerHandle = unknown;

export interface AnalyzeClientDeps {
  fetchImpl?: (input: string, init?: RequestInit) => Promise<Response>;
  generateRequestId?: () => string;
  setTimeoutFn?: (callback: () => void, ms: number) => TimerHandle;
  clearTimeoutFn?: (handle: TimerHandle) => void;
  pollIntervalMs?: number;
  onProgress?: (snapshot: ProgressSnapshot) => void;
  timeoutMs?: number;
}

function defaultRequestId(): string {
  return crypto.randomUUID();
}

export async function runAnalysis(
  input: ScenarioInput,
  deps: AnalyzeClientDeps = {},
): Promise<AnalyzeResponse> {
  const fetchImpl = deps.fetchImpl ?? fetch;
  const generateRequestId = deps.generateRequestId ?? defaultRequestId;
  const setTimeoutFn = deps.setTimeoutFn ?? setTimeout;
  const clearTimeoutFn =
    deps.clearTimeoutFn ??
    ((handle: TimerHandle) => clearTimeout(handle as ReturnType<typeof setTimeout>));
  const pollIntervalMs = deps.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS;
  const timeoutMs = deps.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const requestId = generateRequestId();

  fetchImpl("/api/analyze", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      [REQUEST_ID_HEADER]: requestId,
    },
    body: JSON.stringify(input),
  }).catch(() => {});

  const startedAt = Date.now();

  while (true) {
    if (Date.now() - startedAt > timeoutMs) {
      throw new AnalyzeFailure("the analysis timed out");
    }
    await new Promise<void>((resolve) => setTimeoutFn(resolve, pollIntervalMs));
    try {
      const response = await fetchImpl(`/api/progress/${requestId}`);
      if (response.ok) {
        const json: unknown = await response.json();
        if (isAnalyzeResponse(json)) {
          return json;
        }
        if (isProgressSnapshot(json)) {
          if (json.error) {
            throw new AnalyzeFailure(json.error);
          }
          deps.onProgress?.(json);
        }
      }
    } catch (error) {
      if (error instanceof AnalyzeFailure) {
        throw error;
      }
    }
  }
}
