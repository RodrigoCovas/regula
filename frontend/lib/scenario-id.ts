import type { Mode, ScenarioInput } from "./contract";
import { sha256Hex } from "./sha256";

const STOPWORDS = new Set([
  "a",
  "an",
  "and",
  "are",
  "as",
  "at",
  "be",
  "been",
  "but",
  "by",
  "can",
  "could",
  "did",
  "do",
  "does",
  "for",
  "from",
  "had",
  "has",
  "have",
  "he",
  "her",
  "hers",
  "him",
  "his",
  "how",
  "i",
  "if",
  "in",
  "into",
  "is",
  "it",
  "its",
  "may",
  "might",
  "must",
  "no",
  "not",
  "of",
  "on",
  "or",
  "our",
  "ours",
  "shall",
  "she",
  "should",
  "so",
  "than",
  "that",
  "the",
  "their",
  "theirs",
  "them",
  "then",
  "there",
  "they",
  "this",
  "to",
  "us",
  "was",
  "we",
  "what",
  "when",
  "where",
  "which",
  "who",
  "whom",
  "whose",
  "will",
  "with",
  "would",
  "you",
  "your",
  "yours",
]);

const MAX_CONTENT_WORDS = 4;
const HASH_PREFIX_LENGTH = 8;

function contentWords(description: string): string[] {
  const tokens = description.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];
  return tokens.filter((word) => !STOPWORDS.has(word));
}

function normalizedDescription(description: string): string {
  return description.trim().toLowerCase().replace(/\s+/g, " ");
}

function titleCase(word: string): string {
  return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
}

export function deriveScenarioTitle(description: string): string {
  return contentWords(description)
    .slice(0, MAX_CONTENT_WORDS)
    .map(titleCase)
    .join(" ");
}

export function deriveScenarioId(description: string): string {
  const slug = contentWords(description)
    .slice(0, MAX_CONTENT_WORDS)
    .join("-");
  const hashPrefix = sha256Hex(normalizedDescription(description)).slice(
    0,
    HASH_PREFIX_LENGTH,
  );
  return slug ? `${slug}-${hashPrefix}` : hashPrefix;
}

export function buildScenarioInput(
  description: string,
  question: string,
  mode: Mode,
): ScenarioInput {
  return {
    scenario: {
      id: deriveScenarioId(description),
      title: deriveScenarioTitle(description),
      description,
    },
    question,
    mode,
  };
}

// The canonical demo Scenario (issue #54): the bank cloud outage case, the
// best-performing Live eval Scenario. Its id is the backend's one exact-match
// demo trigger (ADR-0005) — an authored name no description derives to, since
// deriveScenarioId always appends a hash suffix — so the demo button is the
// one UI path that sends it, and hand-typed Scenarios never trigger Demo.
export const demoScenarioInput: ScenarioInput = {
  scenario: {
    id: "bank-cloud-outage",
    title: "Bank Cloud Outage",
    description:
      "A bank relies on an external cloud provider to host critical systems used for online banking. A major technical failure at the cloud provider makes the bank's online banking services unavailable to customers for several hours.",
  },
  question:
    "What regulatory obligations should the bank consider in relation to this incident, its reliance on the cloud provider, and its data protection obligations towards customer data on the disrupted systems?",
  mode: "demo",
};
