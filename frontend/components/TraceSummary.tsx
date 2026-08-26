import type { Trace } from "../lib/contract";

export function TraceSummary({ trace }: { trace: Trace }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 text-sm">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span className="rounded bg-slate-100 px-2 py-0.5 font-mono text-xs text-slate-600">
          {trace.workflow}
        </span>
        <span className="text-xs text-slate-400">Execution trace</span>
      </div>
      <p className="text-slate-700">{trace.summary}</p>
      {trace.unsupported_claims_discarded.length > 0 ? (
        <div className="mt-3">
          <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">
            Unsupported claims discarded (
            {trace.unsupported_claims_discarded.length})
          </p>
          <ul className="list-disc space-y-1 pl-5 text-xs text-slate-500">
            {trace.unsupported_claims_discarded.map((claim, index) => (
              <li key={index}>{claim}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}
