import type { Citation } from "../lib/contract";
import { joinLabel } from "../lib/format";

export function citationLabel(citation: Citation): string {
  return joinLabel(citation.source_short_name, citation.provision);
}

export function citationMeta(citation: Citation): string {
  return joinLabel(
    `source_id: ${citation.source_id}`,
    citation.article_number !== null
      ? `article_number: ${citation.article_number}`
      : null,
    citation.recital_number !== null
      ? `recital_number: ${citation.recital_number}`
      : null,
    citation.annex_number !== null
      ? `annex_number: ${citation.annex_number}`
      : null,
  );
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
          <p className="font-mono text-xs text-slate-400">
            {citationMeta(citation)}
          </p>
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
