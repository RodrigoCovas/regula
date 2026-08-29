import type { ReactNode } from "react";
import type {
  Decision,
  DetailedTraceStep,
} from "../lib/contract";
import { joinLabel } from "../lib/format";

function DecisionBadge({ status }: { status: Decision["status"] }) {
  const styles =
    status === "kept"
      ? "border-emerald-300 bg-emerald-100 text-emerald-800"
      : "border-rose-300 bg-rose-100 text-rose-800";
  return (
    <span
      className={`inline-block shrink-0 rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${styles}`}
    >
      {status}
    </span>
  );
}

function DecisionRow({
  decision,
  text,
  droppedRefs,
}: {
  decision: Decision;
  text: string;
  droppedRefs?: string[];
}) {
  return (
    <li className="flex items-start gap-2 text-xs">
      <DecisionBadge status={decision.status} />
      <span className="text-slate-600">
        {text}
        {decision.reason ? (
          <span className="text-slate-400"> — {decision.reason}</span>
        ) : null}
        {droppedRefs && droppedRefs.length > 0 ? (
          <span className="text-slate-400">
            {" "}
            (dropped refs: {droppedRefs.join(", ")})
          </span>
        ) : null}
      </span>
    </li>
  );
}

function StepSection({
  heading,
  count,
  children,
}: {
  heading: string;
  count: number;
  children: ReactNode;
}) {
  if (count === 0) {
    return null;
  }
  return (
    <div className="mt-2">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
        {heading} ({count})
      </p>
      {children}
    </div>
  );
}

function StepBlock({ step }: { step: DetailedTraceStep }) {
  const researchTargets = step.research_targets ?? [];
  const retrieved = step.retrieved ?? [];
  const toolCalls = step.tool_calls ?? [];
  const claimDecisions = step.claim_decisions ?? [];
  const actionDecisions = step.action_decisions ?? [];
  const summaryDecisions = step.summary_decisions ?? [];
  return (
    <li className="rounded-md border border-slate-200 bg-white p-3">
      <div className="mb-1 flex items-center gap-2">
        <span className="rounded bg-slate-800 px-2 py-0.5 font-mono text-xs text-white">
          {step.step}
        </span>
        <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">
          step
        </span>
      </div>
      <p className="text-sm text-slate-700">{step.action}</p>
      <StepSection heading="Research targets" count={researchTargets.length}>
        <ul className="mt-1 list-disc pl-5 text-xs text-slate-600">
          {researchTargets.map((target, index) => (
            <li key={index}>{target}</li>
          ))}
        </ul>
      </StepSection>
      <StepSection heading="Retrieved passages" count={retrieved.length}>
        <ul className="mt-1 space-y-2">
          {retrieved.map((passage, index) => (
            <li
              key={index}
              className="rounded border border-slate-100 bg-slate-50 p-2 text-xs"
            >
              <p className="font-medium text-slate-700">
                {joinLabel(passage.provision, passage.section, passage.label)}{" "}
                <span className="font-mono text-slate-400">
                  ({passage.source_id})
                </span>
              </p>
              {passage.text ? (
                <p className="mt-1 line-clamp-4 text-slate-500">
                  {passage.text}
                </p>
              ) : null}
            </li>
          ))}
        </ul>
      </StepSection>
      <StepSection heading="Tool calls" count={toolCalls.length}>
        <ul className="mt-1 space-y-1 font-mono text-xs text-slate-600">
          {toolCalls.map((call, index) => (
            <li key={index}>
              {call.tool} {Object.values(call.input).join(" ")} →{" "}
              {call.status ?? (call.chunks_returned !== undefined ? `returned ${call.chunks_returned} chunks` : "?")}
            </li>
          ))}
        </ul>
      </StepSection>
      <StepSection heading="Claim decisions" count={claimDecisions.length}>
        <ul className="mt-1 space-y-1">
          {claimDecisions.map((decision, index) => (
            <DecisionRow key={index} decision={decision} text={decision.claim} />
          ))}
        </ul>
      </StepSection>
      <StepSection heading="Action proposals" count={actionDecisions.length}>
        <ul className="mt-1 space-y-1">
          {actionDecisions.map((decision, index) => (
            <DecisionRow
              key={index}
              decision={decision}
              text={decision.action}
              droppedRefs={decision.dropped_refs}
            />
          ))}
        </ul>
      </StepSection>
      <StepSection heading="Summary decisions" count={summaryDecisions.length}>
        <ul className="mt-1 space-y-1">
          {summaryDecisions.map((decision, index) => (
            <DecisionRow key={index} decision={decision} text={decision.ref} />
          ))}
        </ul>
      </StepSection>
    </li>
  );
}

export function DetailedTrace({
  steps,
}: {
  steps: DetailedTraceStep[] | null;
}) {
  if (!steps || steps.length === 0) {
    return null;
  }
  return (
    <details className="rounded-lg border border-slate-200 bg-white">
      <summary className="cursor-pointer select-none rounded-lg px-4 py-3 text-sm font-medium text-slate-700 hover:bg-slate-50">
        Detailed execution trace ({steps.length} steps)
      </summary>
      <ol className="space-y-3 border-t border-slate-100 p-4">
        {steps.map((step, index) => (
          <StepBlock key={index} step={step} />
        ))}
      </ol>
    </details>
  );
}
