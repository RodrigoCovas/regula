export type ProvisionKind = "article" | "recital" | "annex";

export type Strength = "strong" | "moderate" | "weak";

export const NOT_AVAILABLE_WORKFLOW = "not-available";
export const NOOP_WORKFLOW = "noop";

export function isNotAvailableWorkflow(workflow: string): boolean {
  return workflow === NOT_AVAILABLE_WORKFLOW || workflow === NOOP_WORKFLOW;
}

export interface Citation {
  source_id: string;
  source_short_name: string | null;
  article_number: number | null;
  recital_number: number | null;
  annex_number: number | null;
  section: string | null;
  provision: string | null;
  quote: string | null;
}

export interface Finding {
  statement: string;
  strength: Strength;
  citations: Citation[];
}

export interface Answer {
  findings: Finding[];
  actions: string[];
  citations: Citation[];
}

export interface Trace {
  workflow: string;
  summary: string;
  unsupported_claims_discarded: string[];
}

export interface ClaimDecision {
  claim: string;
  status: "kept" | "rejected";
  reason: string | null;
}

export interface ActionDecision {
  action: string;
  status: "kept" | "rejected";
  reason: string | null;
  dropped_refs: string[];
}

export interface ProvisionTarget {
  source_id: string;
  kind: ProvisionKind;
  number: number;
}

export interface RetrievedPassage extends ProvisionTarget {
  label?: string;
  section: string | null;
  provision: string | null;
  text: string | null;
}

export interface ToolCall {
  tool: string;
  input: Record<string, string | number>;
  status?: string;
  chunks_returned?: number;
}

export interface DetailedTraceStep {
  step: string;
  action: string;
  research_targets?: string[];
  retrieved?: RetrievedPassage[];
  tool_calls?: ToolCall[];
  claim_decisions?: ClaimDecision[];
  action_decisions?: ActionDecision[];
}

export interface AnalyzeResponse {
  answer: Answer;
  trace: Trace;
  detailed_trace: DetailedTraceStep[] | null;
  known_limitations: string[];
}
