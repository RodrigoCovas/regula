import type { Readiness } from "../lib/readiness";

// The shared Readiness states behind the checklist and gating tests.
export const READY_READINESS: Readiness = {
  api_key_set: true,
  embedding_model_present: true,
  corpus_ingested: true,
};

export const KEY_MISSING_READINESS: Readiness = {
  ...READY_READINESS,
  api_key_set: false,
};
