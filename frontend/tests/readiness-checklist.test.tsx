import assert from "node:assert/strict";
import { test } from "node:test";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { missingPrerequisites, type Readiness } from "../lib/readiness";
import { ReadinessChecklist } from "../components/ReadinessChecklist";
import { visibleMarkup } from "./escape";

const READY_READINESS: Readiness = {
  api_key_set: true,
  embedding_model_present: true,
  corpus_ingested: true,
};

function render(readiness: Readiness): string {
  return renderToStaticMarkup(
    React.createElement(ReadinessChecklist, {
      missing: missingPrerequisites(readiness),
    }),
  );
}

test("lists exactly the missing prerequisites", () => {
  const markup = render({ ...READY_READINESS, api_key_set: false });
  assert.ok(markup.includes("Provider key"));
  assert.ok(!markup.includes("Embedding model"));
  assert.ok(!markup.includes("Ingested Corpus"));
});

test("renders the verbatim remediation command for each missing prerequisite", () => {
  const markup = render({ ...READY_READINESS, embedding_model_present: false });
  assert.ok(
    markup.includes(
      visibleMarkup(
        "docker compose exec ollama ollama pull hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:F16",
      ),
    ),
  );
});

test("renders the commands as text only — never a UI-triggered action", () => {
  const markup = render({
    api_key_set: false,
    embedding_model_present: false,
    corpus_ingested: false,
  });
  assert.ok(!markup.includes("<button"));
  assert.ok(!markup.includes("<a "));
});

test("renders nothing when every prerequisite holds", () => {
  const markup = render(READY_READINESS);
  assert.equal(markup, "");
});
