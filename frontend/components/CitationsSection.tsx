import type { Citation } from "../lib/contract";
import { citationLabel } from "./CitationList";
import { StrengthBadge } from "./StrengthBadge";

export function CitationsSection({ citations }: { citations: Citation[] }) {
  if (citations.length === 0) {
    return null;
  }
  return (
    <section aria-label="Citations">
      <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">
        Citations ({citations.length})
      </h2>
      <ul className="space-y-2">
        {citations.map((citation, index) => (
          <li
            key={index}
            className="rounded-md border border-slate-200 bg-white px-3 py-2 text-xs text-slate-700"
          >
            <div className="flex items-center gap-2">
              <span className="font-medium">{citationLabel(citation)}</span>
              {citation.strength ? (
                <StrengthBadge strength={citation.strength} />
              ) : null}
            </div>
            {citation.relevance ? (
              <p className="mt-1 leading-relaxed text-slate-600">
                {citation.relevance}
              </p>
            ) : null}
          </li>
        ))}
      </ul>
    </section>
  );
}
