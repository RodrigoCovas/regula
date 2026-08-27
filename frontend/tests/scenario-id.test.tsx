import assert from "node:assert/strict";
import { test } from "node:test";
import {
  buildScenarioInput,
  demoScenarioInput,
  deriveScenarioId,
} from "../lib/scenario-id";

function slugOf(id: string): string {
  return id.slice(0, id.lastIndexOf("-"));
}

function suffixOf(id: string): string {
  return id.slice(id.lastIndexOf("-") + 1);
}

test("derivation is deterministic: the same description always yields the same id", () => {
  const description =
    "A Spanish fintech uses an AI system to score the creditworthiness of loan applicants";
  assert.equal(deriveScenarioId(description), deriveScenarioId(description));
});

test("stopwords are dropped from the slug", () => {
  const id = deriveScenarioId("the Spanish fintech lending company");
  assert.ok(id.startsWith("spanish-fintech-lending-company-"));
});

test("the slug keeps at most four content words in document order", () => {
  const id = deriveScenarioId(
    "spanish fintech lending company that automates credit decisions for banks",
  );
  assert.equal(slugOf(id), "spanish-fintech-lending-company");
});

test("the id ends with an 8-hex hash suffix", () => {
  const id = deriveScenarioId("Spanish fintech lending");
  assert.match(suffixOf(id), /^[0-9a-f]{8}$/);
});

test("the hash suffix changes when the description changes", () => {
  const a = deriveScenarioId("Spanish fintech lending");
  const b = deriveScenarioId("Spanish fintech lending in Barcelona");
  assert.equal(slugOf(a), "spanish-fintech-lending");
  assert.notEqual(suffixOf(a), suffixOf(b));
});

test("identical first words but a different tail: same slug, different id", () => {
  const a = deriveScenarioId("spanish fintech lending company in Madrid");
  const b = deriveScenarioId("spanish fintech lending company in Seville");
  assert.equal(slugOf(a), slugOf(b));
  assert.notEqual(a, b);
});

test("the hash prefix is a real sha256 prefix of the normalized description", () => {
  assert.equal(deriveScenarioId(""), "e3b0c442");
});

test("a description of only stopwords yields just the hash suffix", () => {
  const id = deriveScenarioId("the and of");
  assert.match(id, /^[0-9a-f]{8}$/);
});

test("the derived payload carries the description as title and the derived id", () => {
  const description = "Spanish fintech lending";
  const input = buildScenarioInput(description, "What regulations apply?");
  assert.equal(input.scenario.title, description);
  assert.equal(input.scenario.description, description);
  assert.equal(input.scenario.id, deriveScenarioId(description));
  assert.equal(input.question, "What regulations apply?");
});

test("the demo payload is the exact canonical scenario, with no user-set title", () => {
  assert.deepEqual(demoScenarioInput, {
    scenario: { id: "spanish-fintech", description: "Spanish fintech lending" },
    question: "What regulations apply?",
  });
});
