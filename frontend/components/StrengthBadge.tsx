import type { Strength } from "../lib/contract";

const STRENGTH_STYLES: Record<Strength, string> = {
  strong: "border-emerald-300 bg-emerald-100 text-emerald-800",
  moderate: "border-amber-300 bg-amber-100 text-amber-800",
  weak: "border-slate-300 bg-slate-100 text-slate-600",
};

export function StrengthBadge({ strength }: { strength: Strength }) {
  return (
    <span
      className={`inline-block shrink-0 rounded-full border px-2.5 py-0.5 text-xs font-semibold uppercase tracking-wide ${STRENGTH_STYLES[strength]}`}
    >
      {strength}
    </span>
  );
}
