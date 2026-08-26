import type { Finding } from "../lib/contract";
import { CitationList } from "./CitationList";
import { StrengthBadge } from "./StrengthBadge";

export function FindingCard({ finding }: { finding: Finding }) {
  return (
    <article className="rounded-lg border border-slate-200 bg-white p-5">
      <div className="flex items-start justify-between gap-4">
        <p className="text-slate-900">{finding.statement}</p>
        <StrengthBadge strength={finding.strength} />
      </div>
      <CitationList citations={finding.citations} />
    </article>
  );
}
