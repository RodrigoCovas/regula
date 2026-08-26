import type { Citation } from "../lib/contract";

export function citationLabel(citation: Citation): string {
  return [citation.source_short_name, citation.provision]
    .filter(Boolean)
    .join(" — ");
}

export function CitationList({ citations }: { citations: Citation[] }) {
  if (citations.length === 0) {
    return null;
  }
  return (
    <ul className="mt-3 space-y-2 border-t border-slate-100 pt-3">
      {citations.map((citation, index) => (
        <li key={index} className="text-sm">
          <p className="font-medium text-slate-700">{citationLabel(citation)}</p>
          {citation.section ? (
            <p className="text-xs text-slate-500">{citation.section}</p>
          ) : null}
          {citation.quote ? (
            <blockquote className="mt-1 line-clamp-3 border-l-2 border-slate-200 pl-2 text-xs italic text-slate-500">
              “{citation.quote}”
            </blockquote>
          ) : null}
        </li>
      ))}
    </ul>
  );
}
