import type { DetailedTraceStep } from "../lib/contract";

function DecisionBadge({ status }: { status: "kept" | "rejected" }) {
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

function StepBlock({ step }: { step: DetailedTraceStep }) {
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
      {step.research_targets && step.research_targets.length > 0 ? (
        <div className="mt-2">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Research targets ({step.research_targets.length})
          </p>
          <ul className="mt-1 list-disc pl-5 text-xs text-slate-600">
            {step.research_targets.map((target, index) => (
              <li key={index}>{target}</li>
            ))}
          </ul>
        </div>
      ) : null}
      {step.retrieved && step.retrieved.length > 0 ? (
        <div className="mt-2">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Retrieved passages ({step.retrieved.length})
          </p>
          <ul className="mt-1 space-y-2">
            {step.retrieved.map((passage, index) => (
              <li
                key={index}
                className="rounded border border-slate-100 bg-slate-50 p-2 text-xs"
              >
                <p className="font-medium text-slate-700">
                  {[
                    passage.provision,
                    passage.section,
                    passage.label,
                  ]
                    .filter(Boolean)
                    .join(" — ")}{" "}
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
        </div>
      ) : null}
      {step.tool_calls && step.tool_calls.length > 0 ? (
        <div className="mt-2">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Tool calls ({step.tool_calls.length})
          </p>
          <ul className="mt-1 space-y-1 font-mono text-xs text-slate-600">
            {step.tool_calls.map((call, index) => (
              <li key={index}>
                {call.tool} {Object.values(call.input).join(" ")} →{" "}
                {call.status ?? (call.chunks_returned !== undefined ? `returned ${call.chunks_returned} chunks` : "?")}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {step.claim_decisions && step.claim_decisions.length > 0 ? (
        <div className="mt-2">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Claim decisions ({step.claim_decisions.length})
          </p>
          <ul className="mt-1 space-y-1">
            {step.claim_decisions.map((decision, index) => (
              <li key={index} className="flex items-start gap-2 text-xs">
                <DecisionBadge status={decision.status} />
                <span className="text-slate-600">
                  {decision.claim}
                  {decision.reason ? (
                    <span className="text-slate-400"> — {decision.reason}</span>
                  ) : null}
                </span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {step.action_decisions && step.action_decisions.length > 0 ? (
        <div className="mt-2">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
            Action proposals ({step.action_decisions.length})
          </p>
          <ul className="mt-1 space-y-1">
            {step.action_decisions.map((decision, index) => (
              <li key={index} className="flex items-start gap-2 text-xs">
                <DecisionBadge status={decision.status} />
                <span className="text-slate-600">
                  {decision.action}
                  {decision.reason ? (
                    <span className="text-slate-400"> — {decision.reason}</span>
                  ) : null}
                  {decision.dropped_refs.length > 0 ? (
                    <span className="text-slate-400">
                      {" "}
                      (dropped refs: {decision.dropped_refs.join(", ")})
                    </span>
                  ) : null}
                </span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
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
