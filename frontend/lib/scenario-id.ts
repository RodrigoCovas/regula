import type { ScenarioInput } from "./contract";
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
): ScenarioInput {
  return {
    scenario: {
      id: deriveScenarioId(description),
      title: deriveScenarioTitle(description),
      description,
    },
    question,
  };
}

export const demoScenarioInput: ScenarioInput = {
  scenario: {
    id: "spanish-fintech",
    title: deriveScenarioTitle(
      "A Spanish fintech startup that uses machine learning to assess creditworthiness for consumer loans. The platform automatically approves or denies applications based on applicant data including income, employment history, and spending patterns. The company operates only in Spain and plans to expand to other EU markets.",
    ),
    description:
      "A Spanish fintech startup that uses machine learning to assess creditworthiness for consumer loans. The platform automatically approves or denies applications based on applicant data including income, employment history, and spending patterns. The company operates only in Spain and plans to expand to other EU markets.",
  },
  question: "What regulations apply to our AI-based credit scoring platform?",
};
