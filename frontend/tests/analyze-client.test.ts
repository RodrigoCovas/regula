import assert from "node:assert/strict";
import { test } from "node:test";
import { runAnalysis, AnalyzeFailure } from "../lib/analyze-client";
import {
  demoAnalyzeResponse,
  demoNotAvailableResponse,
} from "../lib/fixtures";
import type { ProgressSnapshot } from "../lib/progress";
import { buildScenarioInput, demoScenarioInput } from "../lib/scenario-id";
import { jsonResponse, recordingFetch, type FetchCall } from "./fetch";

interface TimerHandle {
  ms: number;
  cb: () => void;
}

function snapshotOf(
  requestId: string,
  phase: ProgressSnapshot["phase"],
  message: string,
): ProgressSnapshot {
  return {
    request_id: requestId,
    phase,
    message,
    transitions: [...(phase ? [{ phase, message, elapsed_ms: 1000 }] : [])],
  };
}

function manualScheduler() {
  const scheduled: TimerHandle[] = [];
  return {
    scheduled,
    setTimeoutFn: (cb: () => void, ms: number): TimerHandle => {
      const timer = { ms, cb };
      scheduled.push(timer);
      return timer;
    },
    clearTimeoutFn: (timer: unknown): void => {
      const handle = timer as TimerHandle;
      const index = scheduled.indexOf(handle);
      if (index >= 0) {
        scheduled.splice(index, 1);
      }
    },
  };
}

async function flush(): Promise<void> {
  await new Promise((resolve) => setImmediate(resolve));
}

async function firePolls(
  scheduler: ReturnType<typeof manualScheduler>,
  count: number,
): Promise<void> {
  for (let i = 0; i < count; i += 1) {
    const timer = scheduler.scheduled.shift();
    if (!timer) {
      throw new Error("expected a scheduled poll timer");
    }
    timer.cb();
    await flush();
  }
}

function analyzeCallOf(calls: FetchCall[]): FetchCall {
  const call = calls.find((entry) => entry.url === "/api/analyze");
  if (!call) {
    throw new Error("expected a call to /api/analyze");
  }
  return call;
}

test("submits the Scenario to /api/analyze with the generated request id header", async () => {
  let pollCount = 0;
  const { calls, impl } = recordingFetch((call) => {
    if (call.url === "/api/analyze") {
      return jsonResponse({});
    }
    pollCount += 1;
    if (pollCount === 1) {
      return jsonResponse(demoAnalyzeResponse);
    }
    return jsonResponse(snapshotOf("run-1", "planner", "decomposing"));
  });

  const responsePromise = runAnalysis(demoScenarioInput, {
    fetchImpl: impl,
    generateRequestId: () => "run-1",
    setTimeoutFn: (cb) => setTimeout(cb, 0) as unknown as TimerHandle,
    clearTimeoutFn: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
    pollIntervalMs: 10,
  });

  const response = await responsePromise;
  assert.deepEqual(response, demoAnalyzeResponse);

  const analyzeCall = analyzeCallOf(calls);
  assert.equal(analyzeCall.init?.method, "POST");
  const headers = analyzeCall.init?.headers as Record<string, string>;
  assert.equal(headers["x-request-id"], "run-1");
  assert.equal(headers["content-type"], "application/json");
  assert.deepEqual(
    JSON.parse(String(analyzeCall.init?.body)),
    demoScenarioInput,
  );
});

test("polls /api/progress/{id} until it returns an AnalyzeResponse", async () => {
  let pollCount = 0;
  const { impl } = recordingFetch((call) => {
    if (call.url === "/api/analyze") {
      return jsonResponse({});
    }
    pollCount += 1;
    if (pollCount < 3) {
      return jsonResponse(snapshotOf("run-1", "researcher", "retrieving"));
    }
    return jsonResponse(demoAnalyzeResponse);
  });

  const response = await runAnalysis(demoScenarioInput, {
    fetchImpl: impl,
    generateRequestId: () => "run-1",
    setTimeoutFn: (cb) => setTimeout(cb, 0) as unknown as TimerHandle,
    clearTimeoutFn: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
    pollIntervalMs: 10,
  });

  assert.deepEqual(response, demoAnalyzeResponse);
  assert.equal(pollCount, 3);
});

test("reports progress snapshots to onProgress", async () => {
  let pollCount = 0;
  const phases: (string | null)[] = [];
  const { impl } = recordingFetch((call) => {
    if (call.url === "/api/analyze") {
      return jsonResponse({});
    }
    pollCount += 1;
    if (pollCount < 3) {
      const phase = pollCount === 1 ? "planner" : "researcher";
      return jsonResponse(snapshotOf("run-1", phase, `phase ${pollCount}`));
    }
    return jsonResponse(demoAnalyzeResponse);
  });

  await runAnalysis(demoScenarioInput, {
    fetchImpl: impl,
    generateRequestId: () => "run-1",
    setTimeoutFn: (cb) => setTimeout(cb, 0) as unknown as TimerHandle,
    clearTimeoutFn: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
    pollIntervalMs: 10,
    onProgress: (snapshot) => {
      phases.push(snapshot.phase);
    },
  });

  assert.deepEqual(phases, ["planner", "researcher"]);
});

test("times out when the analysis takes too long", async () => {
  const { impl } = recordingFetch((call) => {
    if (call.url === "/api/analyze") {
      return jsonResponse({});
    }
    return jsonResponse(snapshotOf("run-1", "researcher", "still working"));
  });

  await assert.rejects(
    runAnalysis(demoScenarioInput, {
      fetchImpl: impl,
      generateRequestId: () => "run-1",
      setTimeoutFn: (cb) => setTimeout(cb, 0) as unknown as TimerHandle,
      clearTimeoutFn: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
      pollIntervalMs: 10,
      timeoutMs: 50,
    }),
    (error: Error) => {
      assert.ok(error instanceof AnalyzeFailure);
      assert.match(error.message, /timed out/);
      return true;
    },
  );
});

test("throws AnalyzeFailure when the progress endpoint returns a terminal failure snapshot", async () => {
  let pollCount = 0;
  const { impl } = recordingFetch((call) => {
    if (call.url === "/api/analyze") {
      return jsonResponse({});
    }
    pollCount += 1;
    if (pollCount === 1) {
      return jsonResponse(snapshotOf("run-1", "planner", "decomposing"));
    }
    return jsonResponse({
      request_id: "run-1",
      phase: "planner",
      message: "decomposing",
      transitions: [{ phase: "planner", message: "decomposing", elapsed_ms: 1000 }],
      error: "the LLM provider rejected the request",
    });
  });

  await assert.rejects(
    runAnalysis(demoScenarioInput, {
      fetchImpl: impl,
      generateRequestId: () => "run-1",
      setTimeoutFn: (cb) => setTimeout(cb, 0) as unknown as TimerHandle,
      clearTimeoutFn: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
      pollIntervalMs: 10,
    }),
    (error: Error) => {
      assert.ok(error instanceof AnalyzeFailure);
      assert.match(error.message, /LLM provider rejected the request/);
      return true;
    },
  );
});

test("without an injected generator, the request id is a fresh UUID", async () => {
  let pollCount = 0;
  const { calls, impl } = recordingFetch((call) => {
    if (call.url === "/api/analyze") {
      return jsonResponse({});
    }
    pollCount += 1;
    if (pollCount === 1) {
      return jsonResponse(demoAnalyzeResponse);
    }
    return jsonResponse(snapshotOf("run-1", "planner", "decomposing"));
  });

  await runAnalysis(demoScenarioInput, {
    fetchImpl: impl,
    setTimeoutFn: (cb) => setTimeout(cb, 0) as unknown as TimerHandle,
    clearTimeoutFn: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
    pollIntervalMs: 10,
  });

  const analyzeCall = analyzeCallOf(calls);
  const headers = analyzeCall.init?.headers as Record<string, string>;
  assert.match(headers["x-request-id"], /^[0-9a-f]{8}-[0-9a-f-]{27}$/);
});

test("the request body carries the selected mode explicitly (issue #48)", async () => {
  let pollCount = 0;
  const { calls, impl } = recordingFetch((call) => {
    if (call.url === "/api/analyze") {
      return jsonResponse({});
    }
    pollCount += 1;
    if (pollCount === 1) {
      return jsonResponse(demoAnalyzeResponse);
    }
    return jsonResponse(snapshotOf("run-1", "planner", "decomposing"));
  });

  const liveInput = buildScenarioInput(
    "A Berlin insurtech piloting an AI claims triage tool",
    "What regulations apply?",
    "live",
  );
  await runAnalysis(liveInput, {
    fetchImpl: impl,
    generateRequestId: () => "run-1",
    setTimeoutFn: (cb) => setTimeout(cb, 0) as unknown as TimerHandle,
    clearTimeoutFn: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
    pollIntervalMs: 10,
  });

  const analyzeCall = analyzeCallOf(calls);
  const body = JSON.parse(String(analyzeCall.init?.body));
  assert.equal(body.mode, "live");
  assert.equal(body.scenario.description, liveInput.scenario.description);
  assert.equal(body.question, liveInput.question);
});

test("a custom Scenario in Demo mode yields the Not-available response with the demo-button hint", async () => {
  let pollCount = 0;
  const { impl } = recordingFetch((call) => {
    if (call.url === "/api/analyze") {
      return jsonResponse({});
    }
    pollCount += 1;
    if (pollCount === 1) {
      return jsonResponse(demoNotAvailableResponse);
    }
    return jsonResponse(snapshotOf("run-1", "planner", "decomposing"));
  });

  const customDemoInput = buildScenarioInput(
    "A Portuguese meal-delivery platform",
    "What regulations apply?",
    "demo",
  );
  const response = await runAnalysis(customDemoInput, {
    fetchImpl: impl,
    generateRequestId: () => "run-1",
    setTimeoutFn: (cb) => setTimeout(cb, 0) as unknown as TimerHandle,
    clearTimeoutFn: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
    pollIntervalMs: 10,
  });

  assert.equal(response.trace.workflow, "noop");
  assert.deepEqual(response.answer.findings, []);
  assert.ok(
    response.answer.actions.some((action) =>
      action.includes("Try the demo scenario"),
    ),
  );
});
