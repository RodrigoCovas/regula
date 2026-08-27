import type { ProgressSnapshot } from "../lib/progress";
import { formatElapsed, phaseLabel } from "../lib/progress";

interface ProgressPanelProps {
  snapshots: ProgressSnapshot[];
}

function last<T>(items: T[]): T | undefined {
  return items[items.length - 1];
}

export function ProgressPanel({ snapshots }: ProgressPanelProps) {
  const latest = last(snapshots);
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
      <div className="flex items-center gap-3">
        <span className="relative flex h-3 w-3">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-slate-400 opacity-75" />
          <span className="relative inline-flex h-3 w-3 rounded-full bg-slate-500" />
        </span>
        <p className="text-sm font-medium text-slate-700">
          {latest && latest.phase ? (
            <>
              {phaseLabel(latest.phase)}: {latest.message}
            </>
          ) : (
            "Starting analysis…"
          )}
        </p>
      </div>

      {snapshots.length > 0 ? (
        <ol className="mt-4 space-y-2">
          {snapshots.map((snapshot, index) => {
            const transition = last(snapshot.transitions);
            if (!transition) {
              return null;
            }
            return (
              <li
                key={index}
                className="flex items-start justify-between gap-4 text-sm"
              >
                <span className="text-slate-700">
                  <span className="font-medium">
                    {phaseLabel(transition.phase)}
                  </span>
                  {" — "}
                  {transition.message}
                </span>
                <span className="shrink-0 text-slate-500">
                  {formatElapsed(transition.elapsed_ms)}
                </span>
              </li>
            );
          })}
        </ol>
      ) : null}
    </div>
  );
}
