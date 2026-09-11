import assert from "node:assert/strict";
import { test } from "node:test";
import {
  buildScenarioInput,
  demoScenarioInput,
  deriveScenarioId,
  deriveScenarioTitle,
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

test("a known description yields its exact pinned id", () => {
  assert.equal(
    deriveScenarioId("Spanish fintech lending"),
    "spanish-fintech-lending-bcae705f",
  );
});

test("stopwords are dropped from the slug", () => {
  const id = deriveScenarioId("the Spanish fintech lending company");
  assert.ok(id.startsWith("spanish-fintech-lending-company-"));
});

test("pronouns, auxiliaries, and negation are stopwords too", () => {
  const id = deriveScenarioId("we do not offer lending for our clients");
  assert.ok(id.startsWith("offer-lending-clients-"));
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

test("the title is the first four content words title-cased", () => {
  assert.equal(
    deriveScenarioTitle("spanish fintech lending company"),
    "Spanish Fintech Lending Company",
  );
});

test("the title drops stopwords", () => {
  assert.equal(
    deriveScenarioTitle("the spanish fintech lending company"),
    "Spanish Fintech Lending Company",
  );
});

test("the title keeps at most four words", () => {
  assert.equal(
    deriveScenarioTitle(
      "spanish fintech lending company that automates credit decisions",
    ),
    "Spanish Fintech Lending Company",
  );
});

test("the title is empty for a description of only stopwords", () => {
  assert.equal(deriveScenarioTitle("the and of"), "");
});

test("the derived payload carries the derived title and the derived id", () => {
  const description = "Spanish fintech lending";
  const input = buildScenarioInput(description, "What regulations apply?", "demo");
  assert.equal(input.scenario.title, "Spanish Fintech Lending");
  assert.equal(input.scenario.description, description);
  assert.equal(input.scenario.id, deriveScenarioId(description));
  assert.equal(input.question, "What regulations apply?");
});

test("every submission carries the selected mode explicitly (issue #48)", () => {
  const description = "Spanish fintech lending";
  const demo = buildScenarioInput(description, "What regulations apply?", "demo");
  assert.equal(demo.mode, "demo");
  const live = buildScenarioInput(description, "What regulations apply?", "live");
  assert.equal(live.mode, "live");
});

test("the demo payload is the exact canonical scenario with enriched content", () => {
  // The canonical id is the bank cloud outage case's authored Scenario id
  // (issue #54): an exact-match trigger the description never derives to,
  // because deriveScenarioId always appends a hash suffix (ADR-0005).
  assert.equal(
    deriveScenarioId(demoScenarioInput.scenario.description),
    "bank-relies-external-cloud-f27c7a6d",
  );
  assert.notEqual(
    deriveScenarioId(demoScenarioInput.scenario.description),
    demoScenarioInput.scenario.id,
  );
  assert.deepEqual(demoScenarioInput, {
    scenario: {
      id: "bank-cloud-outage",
      title: "Bank Cloud Outage",
      description:
        "A bank relies on an external cloud provider to host critical systems used for online banking. A major technical failure at the cloud provider makes the bank's online banking services unavailable to customers for several hours.",
    },
    question:
      "What regulatory obligations should the bank consider in relation to this incident, its reliance on the cloud provider, and its data protection obligations towards customer data on the disrupted systems?",
    mode: "demo",
  });
});
