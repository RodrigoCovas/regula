import assert from "node:assert/strict";
import { test } from "node:test";
import {
  fetchReadiness,
  isLiveReady,
  missingPrerequisites,
  ReadinessUnavailableError,
  readinessBlocks,
  type Readiness,
} from "../lib/readiness";
import { jsonResponse, recordingFetch } from "./fetch";
import { READY_READINESS } from "./readiness-states";

test("fetchReadiness GETs /readiness and reports the three booleans", async () => {
  const { calls, impl } = recordingFetch(() => jsonResponse(READY_READINESS));

  const readiness = await fetchReadiness({ fetchImpl: impl });

  assert.deepEqual(readiness, READY_READINESS);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "/readiness");
  assert.equal(calls[0].init?.method, "GET");
});

test("a failing readiness endpoint is a ReadinessUnavailableError, never a crash", async () => {
  const { impl } = recordingFetch(() => jsonResponse({}, 500));

  await assert.rejects(
    fetchReadiness({ fetchImpl: impl }),
    (error: Error) => {
      assert.ok(error instanceof ReadinessUnavailableError);
      return true;
    },
  );
});

test("a network failure is a ReadinessUnavailableError too", async () => {
  const failing = (): Promise<Response> => Promise.reject(new Error("ECONNREFUSED"));

  await assert.rejects(
    fetchReadiness({ fetchImpl: failing }),
    (error: Error) => error instanceof ReadinessUnavailableError,
  );
});

test("a malformed readiness payload is a ReadinessUnavailableError", async () => {
  const { impl } = recordingFetch(() =>
    jsonResponse({ api_key_set: true, embedding_model_present: "yes" }),
  );

  await assert.rejects(
    fetchReadiness({ fetchImpl: impl }),
    (error: Error) => error instanceof ReadinessUnavailableError,
  );
});

test("the ready state passes isLiveReady", () => {
  assert.equal(isLiveReady(READY_READINESS), true);
});

test("each missing prerequisite alone fails isLiveReady", () => {
  const missingKey: Readiness = { ...READY_READINESS, api_key_set: false };
  const missingModel: Readiness = {
    ...READY_READINESS,
    embedding_model_present: false,
  };
  const missingCorpus: Readiness = { ...READY_READINESS, corpus_ingested: false };

  assert.equal(isLiveReady(missingKey), false);
  assert.equal(isLiveReady(missingModel), false);
  assert.equal(isLiveReady(missingCorpus), false);
});

test("a fully ready store names no missing prerequisites", () => {
  assert.deepEqual(missingPrerequisites(READY_READINESS), []);
});

test("each missing prerequisite is named individually with its verbatim command", () => {
  const keyMissing = missingPrerequisites({ ...READY_READINESS, api_key_set: false });
  assert.equal(keyMissing.length, 1);
  assert.equal(keyMissing[0].key, "api_key_set");
  assert.ok(keyMissing[0].label.includes("OPENROUTER_API_KEY"));
  assert.equal(
    keyMissing[0].command,
    'echo "OPENROUTER_API_KEY=your-key-here" >> backend/.env.local',
  );

  const modelMissing = missingPrerequisites({
    ...READY_READINESS,
    embedding_model_present: false,
  });
  assert.equal(modelMissing.length, 1);
  assert.equal(modelMissing[0].key, "embedding_model_present");
  assert.equal(
    modelMissing[0].command,
    "docker compose exec ollama ollama pull hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:F16",
  );

  const corpusMissing = missingPrerequisites({
    ...READY_READINESS,
    corpus_ingested: false,
  });
  assert.equal(corpusMissing.length, 1);
  assert.equal(corpusMissing[0].key, "corpus_ingested");
  assert.equal(
    corpusMissing[0].command,
    "docker compose exec backend python -m backend.src.ingest",
  );
});

test("every missing prerequisite is listed together, in checklist order", () => {
  const missing = missingPrerequisites({
    api_key_set: false,
    embedding_model_present: false,
    corpus_ingested: false,
  });
  assert.deepEqual(
    missing.map((item) => item.key),
    ["api_key_set", "embedding_model_present", "corpus_ingested"],
  );
});

test("the readiness gate blocks submission in every state but ready and idle", () => {
  assert.equal(readinessBlocks({ kind: "idle" }), false);
  assert.equal(readinessBlocks({ kind: "checking" }), true);
  assert.equal(readinessBlocks({ kind: "ready" }), false);
  assert.equal(readinessBlocks({ kind: "incomplete", missing: [] }), true);
  assert.equal(readinessBlocks({ kind: "unavailable", message: "x" }), true);
});
