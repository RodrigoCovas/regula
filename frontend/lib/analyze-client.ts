import type { AnalyzeResponse, ScenarioInput } from "./contract";
import type { ProgressSnapshot } from "./progress";
import { isAnalyzeResponse, isProgressSnapshot } from "./progress";

const REQUEST_ID_HEADER = "x-request-id";
const DEFAULT_POLL_INTERVAL_MS = 1000;

export class AnalyzeFailure extends Error {}

export type TimerHandle = unknown;

export interface AnalyzeClientDeps {
  fetchImpl?: (input: string, init?: RequestInit) => Promise<Response>;
  generateRequestId?: () => string;
  setTimeoutFn?: (callback: () => void, ms: number) => TimerHandle;
  clearTimeoutFn?: (handle: TimerHandle) => void;
  pollIntervalMs?: number;
  onProgress?: (snapshot: ProgressSnapshot) => void;
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
  const requestId = generateRequestId();

  const analyzePromise = fetchImpl("/api/analyze", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      [REQUEST_ID_HEADER]: requestId,
    },
    body: JSON.stringify(input),
  });

  let settled = false;
  let pollTimer: TimerHandle = null;

  const schedulePoll = (): void => {
    pollTimer = setTimeoutFn(poll, pollIntervalMs);
  };

  const poll = async (): Promise<void> => {
    try {
      const response = await fetchImpl(`/api/progress/${requestId}`);
      if (response.ok) {
        const json: unknown = await response.json();
        if (isProgressSnapshot(json)) {
          deps.onProgress?.(json);
        }
      }
    } catch {
      // A lost poll tick must never fail the analysis: the POST carries the
      // answer; progress reads are best-effort.
    }
    if (!settled) {
      schedulePoll();
    }
  };

  schedulePoll();

  try {
    const response = await analyzePromise;
    if (!response.ok) {
      throw new AnalyzeFailure(
        `the backend answered the analysis with HTTP ${response.status}`,
      );
    }
    const json: unknown = await response.json();
    if (!isAnalyzeResponse(json)) {
      throw new AnalyzeFailure(
        "the backend answered the analysis with an unrecognized shape",
      );
    }
    return json;
  } finally {
    settled = true;
    if (pollTimer !== null) {
      clearTimeoutFn(pollTimer);
    }
  }
}
