import type { AnalyzeResponse } from "../lib/contract";
import { NOT_AVAILABLE_WORKFLOW } from "../lib/contract";
import { CitationsSection } from "./CitationsSection";
import { DetailedTrace } from "./DetailedTrace";
import { FindingCard } from "./FindingCard";
import { KnownLimitations } from "./KnownLimitations";
import { NumberedList } from "./NumberedList";
import { OutcomePanel } from "./OutcomePanel";
import { TraceSummary } from "./TraceSummary";

export function AnswerSurface({ response }: { response: AnalyzeResponse }) {
  if (response.trace.workflow === NOT_AVAILABLE_WORKFLOW) {
    return <OutcomePanel response={response} kind="not-available" />;
  }
  if (response.answer.findings.length === 0) {
    return <OutcomePanel response={response} kind="insufficient-evidence" />;
  }
  return (
    <div className="space-y-6">
      <section aria-label="Findings" className="space-y-4">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
          Findings ({response.answer.findings.length})
        </h2>
        {response.answer.findings.map((finding, index) => (
          <FindingCard key={index} finding={finding} />
        ))}
      </section>
      <NumberedList
        ariaLabel="Actions"
        heading="Actions for a qualified professional"
        items={response.answer.actions}
      />
      <CitationsSection citations={response.answer.citations} />
      <section aria-label="Execution trace" className="space-y-3">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
          Execution trace
        </h2>
        <TraceSummary trace={response.trace} />
        <DetailedTrace steps={response.detailed_trace} />
      </section>
      <KnownLimitations limitations={response.known_limitations} />
    </div>
  );
}
