export type ProvisionKind = "article" | "recital" | "annex";

export type Strength = "strong" | "moderate" | "weak";

export const NOT_AVAILABLE_WORKFLOW = "not-available";

export interface Scenario {
  id: string;
  title?: string;
  description: string;
}

// The per-run mode (ADR-0008): every analysis request carries it explicitly.
export type Mode = "demo" | "live";

export interface ScenarioInput {
  scenario: Scenario;
  question: string;
  mode: Mode;
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
  // Answer-wide fields (issue #47): the Citations section renders them from
  // the Answer's list only — a per-Finding Citation carries neither.
  relevance?: string | null;
  strength?: Strength | null;
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

export interface Decision {
  status: "kept" | "rejected";
  reason: string | null;
}

export interface ClaimDecision extends Decision {
  claim: string;
}

export interface ActionDecision extends Decision {
  action: string;
  dropped_refs: string[];
}

export interface RetrievedPassage {
  source_id: string;
  kind: ProvisionKind;
  number: number | null;
  section: string | null;
  provision: string | null;
  text: string | null;
  label?: string;
}

export interface ToolCall {
  tool: string;
  input: Record<string, string | number>;
  status?: string;
  chunks_returned?: number;
}

export interface SummaryDecision {
  ref: string;
  status: "kept" | "rejected";
  reason: string | null;
}

export interface DetailedTraceStep {
  step: string;
  action: string;
  research_targets?: string[];
  retrieved?: RetrievedPassage[];
  tool_calls?: ToolCall[];
  claim_decisions?: ClaimDecision[];
  action_decisions?: ActionDecision[];
  summary_decisions?: SummaryDecision[];
}

export interface AnalyzeResponse {
  answer: Answer;
  trace: Trace;
  detailed_trace: DetailedTraceStep[] | null;
  known_limitations: string[];
}
