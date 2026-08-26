import type { Citation } from "../lib/contract";
import { citationLabel } from "./CitationList";

export function CitationsSection({ citations }: { citations: Citation[] }) {
  if (citations.length === 0) {
    return null;
  }
  return (
    <section aria-label="Citations">
      <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">
        Citations ({citations.length})
      </h2>
      <ul className="flex flex-wrap gap-2">
        {citations.map((citation, index) => (
          <li
            key={index}
            className="rounded-md border border-slate-200 bg-white px-2.5 py-1 text-xs text-slate-700"
          >
            {citationLabel(citation)}
          </li>
        ))}
      </ul>
    </section>
  );
}
