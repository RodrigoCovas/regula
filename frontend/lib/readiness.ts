import type { Mode } from "./contract";

// Live-mode Readiness (CONTEXT.md) as the readiness endpoint reports it
// (issue #45): a configured provider key, the embedding model available on
// Ollama, and a non-empty ingested Corpus — each probed independently.
export interface Readiness {
  api_key_set: boolean;
  embedding_model_present: boolean;
  corpus_ingested: boolean;
}

export type ReadinessKey = keyof Readiness;

export class ReadinessUnavailableError extends Error {}

export function isReadiness(json: unknown): json is Readiness {
  if (typeof json !== "object" || json === null) {
    return false;
  }
  const candidate = json as Record<string, unknown>;
  const keys: ReadinessKey[] = [
    "api_key_set",
    "embedding_model_present",
    "corpus_ingested",
  ];
  return keys.every((key) => typeof candidate[key] === "boolean");
}

export interface ReadinessClientDeps {
  fetchImpl?: (input: string, init?: RequestInit) => Promise<Response>;
}

const READINESS_URL = "/readiness";

export async function fetchReadiness(
  deps: ReadinessClientDeps = {},
): Promise<Readiness> {
  const fetchImpl = deps.fetchImpl ?? fetch;
  let response: Response;
  try {
    response = await fetchImpl(READINESS_URL, { method: "GET" });
  } catch (error) {
    throw new ReadinessUnavailableError(
      `Live-mode readiness could not be checked (${String(error)}).`,
    );
  }
  if (!response.ok) {
    throw new ReadinessUnavailableError(
      `Live-mode readiness could not be checked (HTTP ${response.status}).`,
    );
  }
  const json: unknown = await response.json().catch(() => undefined);
  if (!isReadiness(json)) {
    throw new ReadinessUnavailableError(
      "Live-mode readiness could not be checked (unexpected response shape).",
    );
  }
  return json;
}

export function isLiveReady(readiness: Readiness): boolean {
  return (
    readiness.api_key_set &&
    readiness.embedding_model_present &&
    readiness.corpus_ingested
  );
}

// One checklist entry: a missing Live-mode prerequisite and its verbatim
// remediation command (README, "Live mode setup") — rendered as text only,
// never a UI-triggered action.
export interface ReadinessItem {
  key: ReadinessKey;
  label: string;
  note: string;
  command: string;
}

const CHECKLIST: ReadinessItem[] = [
  {
    key: "api_key_set",
    label: "Provider key (OPENROUTER_API_KEY)",
    note: "Set it in .env and restart the backend.",
    command: 'echo "OPENROUTER_API_KEY=your-key-here" >> .env',
  },
  {
    key: "embedding_model_present",
    label: "Embedding model",
    note: "Pull the embedding model into Ollama.",
    command:
      "docker compose exec ollama ollama pull hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:F16",
  },
  {
    key: "corpus_ingested",
    label: "Ingested Corpus",
    note: "Ingest the Corpus into the vector store; it takes effect immediately.",
    command: "docker compose exec backend python -m backend.src.ingest",
  },
];

export function missingPrerequisites(readiness: Readiness): ReadinessItem[] {
  return CHECKLIST.filter((item) => !readiness[item.key]);
}

// --- The form's readiness gate (issue #48) ---

// The readiness state behind the Demo/Live toggle, as the form derives it.
// Demo mode never gates; Live mode is hard-blocked until a fresh check reads
// ready — an unverifiable Readiness blocks too, never a degraded Live run.
export type ReadinessGate =
  | { kind: "idle" }
  | { kind: "checking" }
  | { kind: "ready" }
  | { kind: "incomplete"; missing: ReadinessItem[] }
  | { kind: "unavailable"; message: string };

// What one readiness check attempt left behind, as the form records it:
// nothing in force, a completed probe, or a probe that could not run.
export type ReadinessResult =
  | { kind: "none" }
  | { kind: "known"; readiness: Readiness }
  | { kind: "unavailable"; message: string };

export function liveReadinessGate(mode: Mode, result: ReadinessResult): ReadinessGate {
  // The mode check leads, so an error can only ever shape a Live gate —
  // a stale check outcome can never block a Demo submission.
  if (mode !== "live") {
    return { kind: "idle" };
  }
  switch (result.kind) {
    case "none":
      return { kind: "checking" };
    case "unavailable":
      return { kind: "unavailable", message: result.message };
    case "known":
      return isLiveReady(result.readiness)
        ? { kind: "ready" }
        : { kind: "incomplete", missing: missingPrerequisites(result.readiness) };
  }
}

export function readinessBlocks(gate: ReadinessGate): boolean {
  return gate.kind !== "idle" && gate.kind !== "ready";
}
