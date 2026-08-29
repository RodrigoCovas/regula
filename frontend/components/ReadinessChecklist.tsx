import type { ReadinessItem } from "../lib/readiness";

// The Live-mode readiness checklist (issue #48): exactly the missing
// prerequisites, each with its verbatim remediation command as text.
// Commands are never rendered as buttons or links — the operator runs them
// themselves; the UI triggers no ingest and no pull.
export function ReadinessChecklist({ missing }: { missing: ReadinessItem[] }) {
  if (missing.length === 0) {
    return null;
  }
  return (
    <div className="rounded-md border border-amber-200 bg-amber-50 p-4 text-sm">
      <p className="font-semibold text-amber-900">
        Live mode is not ready — missing prerequisites:
      </p>
      <ul className="mt-2 space-y-3">
        {missing.map((item) => (
          <li key={item.key} className="space-y-1">
            <p className="font-medium text-amber-900">{item.label}</p>
            <p className="text-amber-800">{item.note}</p>
            <code className="block break-all rounded bg-white px-2 py-1 font-mono text-xs text-slate-800">
              {item.command}
            </code>
          </li>
        ))}
      </ul>
    </div>
  );
}
