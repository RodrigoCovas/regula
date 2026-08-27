import assert from "node:assert/strict";
import { test } from "node:test";
import { runAnalysis } from "../lib/analyze-client";
import { demoAnalyzeResponse, notAvailableResponse } from "../lib/fixtures";
import type { ProgressSnapshot } from "../lib/progress";
import { demoScenarioInput } from "../lib/scenario-id";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
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

type FetchCall = { url: string; init?: RequestInit };

function recordingFetch(
  handler: (call: FetchCall) => Promise<Response> | Response,
) {
  const calls: FetchCall[] = [];
  const impl = (url: string, init?: RequestInit): Promise<Response> => {
    const call = { url, init };
    calls.push(call);
    return Promise.resolve(handler(call));
  };
  return { calls, impl };
}

interface TimerHandle {
  ms: number;
  cb: () => void;
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
  const { calls, impl } = recordingFetch(() => jsonResponse(demoAnalyzeResponse));
  const response = await runAnalysis(demoScenarioInput, {
    fetchImpl: impl,
    generateRequestId: () => "run-1",
  });
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

test("polls /api/progress/{id} at the configured cadence while the analysis is pending and stops once it settles", async () => {
  const pending = deferred<Response>();
  let pollCount = 0;
  const { calls, impl } = recordingFetch((call) => {
    if (call.url === "/api/analyze") {
      return pending.promise;
    }
    pollCount += 1;
    return jsonResponse(
      snapshotOf("run-1", "planner", "decomposing the question"),
    );
  });
  const scheduler = manualScheduler();

  const run = runAnalysis(demoScenarioInput, {
    fetchImpl: impl,
    generateRequestId: () => "run-1",
    setTimeoutFn: scheduler.setTimeoutFn,
    clearTimeoutFn: scheduler.clearTimeoutFn,
  });
  await flush();

  await firePolls(scheduler, 3);
  assert.equal(pollCount, 3, "polls fire while the analysis is pending");
  assert.deepEqual(
    scheduler.scheduled.map((timer) => timer.ms),
    [1000],
    "the poll cadence is one second",
  );

  pending.resolve(jsonResponse(demoAnalyzeResponse));
  await run;
  await flush();
  assert.equal(scheduler.scheduled.length, 0, "the poll loop is stopped once the run settles");
  const progressUrls = calls
    .filter((call) => call.url.startsWith("/api/progress/"))
    .map((call) => call.url);
  assert.deepEqual(progressUrls, [
    "/api/progress/run-1",
    "/api/progress/run-1",
    "/api/progress/run-1",
  ]);
});

test("the poll cadence is configurable", async () => {
  const pending = deferred<Response>();
  const { impl } = recordingFetch((call) =>
    call.url === "/api/analyze"
      ? pending.promise
      : jsonResponse(snapshotOf("run-1", null, "")),
  );
  const scheduler = manualScheduler();

  const run = runAnalysis(demoScenarioInput, {
    fetchImpl: impl,
    generateRequestId: () => "run-1",
    setTimeoutFn: scheduler.setTimeoutFn,
    clearTimeoutFn: scheduler.clearTimeoutFn,
    pollIntervalMs: 250,
  });
  await flush();
  assert.deepEqual(
    scheduler.scheduled.map((timer) => timer.ms),
    [250],
    "the configured cadence drives the timer",
  );

  pending.resolve(jsonResponse(demoAnalyzeResponse));
  await run;
});

test("reports each snapshot to onProgress in poll order", async () => {
  const pending = deferred<Response>();
  const phases: (string | null)[] = [];
  let tick = 0;
  const { impl } = recordingFetch((call) => {
    if (call.url === "/api/analyze") {
      return pending.promise;
    }
    tick += 1;
    const phase = ["planner", "researcher", "verifier", "proposer"][tick - 1];
    return jsonResponse(
      snapshotOf("run-1", phase as ProgressSnapshot["phase"], `${phase} message`),
    );
  });
  const scheduler = manualScheduler();

  const run = runAnalysis(demoScenarioInput, {
    fetchImpl: impl,
    generateRequestId: () => "run-1",
    setTimeoutFn: scheduler.setTimeoutFn,
    clearTimeoutFn: scheduler.clearTimeoutFn,
    onProgress: (snapshot) => phases.push(snapshot.phase),
  });
  await flush();
  await firePolls(scheduler, 4);

  assert.deepEqual(phases, ["planner", "researcher", "verifier", "proposer"]);

  pending.resolve(jsonResponse(demoAnalyzeResponse));
  await run;
});

test("ignores Not-available-shaped poll replies without failing the run", async () => {
  const pending = deferred<Response>();
  let reported = 0;
  const { impl } = recordingFetch((call) =>
    call.url === "/api/analyze"
      ? pending.promise
      : jsonResponse(notAvailableResponse),
  );
  const scheduler = manualScheduler();

  const run = runAnalysis(demoScenarioInput, {
    fetchImpl: impl,
    generateRequestId: () => "run-1",
    setTimeoutFn: scheduler.setTimeoutFn,
    clearTimeoutFn: scheduler.clearTimeoutFn,
    onProgress: () => {
      reported += 1;
    },
  });
  await flush();
  await firePolls(scheduler, 2);

  pending.resolve(jsonResponse(demoAnalyzeResponse));
  const response = await run;
  assert.deepEqual(response, demoAnalyzeResponse);
  assert.equal(reported, 0, "Not-available poll replies carry no progress");
});

test("a failing poll does not abort the analysis", async () => {
  const pending = deferred<Response>();
  let reported = 0;
  const { impl } = recordingFetch((call) =>
    call.url === "/api/analyze"
      ? pending.promise
      : Promise.reject(new TypeError("network down")),
  );
  const scheduler = manualScheduler();

  const run = runAnalysis(demoScenarioInput, {
    fetchImpl: impl,
    generateRequestId: () => "run-1",
    setTimeoutFn: scheduler.setTimeoutFn,
    clearTimeoutFn: scheduler.clearTimeoutFn,
    onProgress: () => {
      reported += 1;
    },
  });
  await flush();
  await firePolls(scheduler, 2);

  pending.resolve(jsonResponse(demoAnalyzeResponse));
  const response = await run;
  assert.deepEqual(response, demoAnalyzeResponse);
  assert.equal(reported, 0);
});

test("rejects naming the HTTP status when the backend fails", async () => {
  const impl = () => Promise.resolve(jsonResponse({ detail: "boom" }, 500));
  await assert.rejects(
    runAnalysis(demoScenarioInput, {
      fetchImpl: impl,
      generateRequestId: () => "run-1",
    }),
    /HTTP 500/,
  );
});

test("rejects when the response is not an AnalyzeResponse", async () => {
  const impl = () => Promise.resolve(jsonResponse({ unrelated: true }));
  await assert.rejects(
    runAnalysis(demoScenarioInput, {
      fetchImpl: impl,
      generateRequestId: () => "run-1",
    }),
    /unrecognized/,
  );
});

test("rejects when the backend cannot be reached at all", async () => {
  const impl = () => Promise.reject(new TypeError("fetch failed"));
  await assert.rejects(
    runAnalysis(demoScenarioInput, {
      fetchImpl: impl,
      generateRequestId: () => "run-1",
    }),
    /fetch failed/,
  );
});

test("without an injected generator, the request id is a fresh UUID", async () => {
  const { calls, impl } = recordingFetch(() => jsonResponse(demoAnalyzeResponse));
  await runAnalysis(demoScenarioInput, { fetchImpl: impl });
  const headers = calls[0].init?.headers as Record<string, string>;
  assert.match(headers["x-request-id"], /^[0-9a-f]{8}-[0-9a-f-]{27}$/);
});
