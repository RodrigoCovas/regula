import type { AnalyzeResponse } from "../lib/contract";
import { DetailedTrace } from "./DetailedTrace";
import { KnownLimitations } from "./KnownLimitations";
import { NumberedList } from "./NumberedList";

export type OutcomeKind = "insufficient-evidence" | "not-available";

const OUTCOME_STYLES: Record<
  OutcomeKind,
  { title: string; box: string; heading: string; detailedTrace: boolean }
> = {
  "insufficient-evidence": {
    title: "Insufficient evidence",
    box: "border-indigo-300 bg-indigo-50",
    heading: "text-indigo-900",
    detailedTrace: true,
  },
  "not-available": {
    title: "Not available",
    box: "border-amber-300 bg-amber-50",
    heading: "text-amber-900",
    detailedTrace: false,
  },
};

export function OutcomePanel({
  response,
  kind,
}: {
  response: AnalyzeResponse;
  kind: OutcomeKind;
}) {
  const styles = OUTCOME_STYLES[kind];
  return (
    <div className="space-y-4">
      <div className={`rounded-lg border p-5 ${styles.box}`}>
        <h2 className={`mb-2 text-lg font-semibold ${styles.heading}`}>
          {styles.title}
        </h2>
        <p className={`text-sm ${styles.heading}`}>{response.trace.summary}</p>
      </div>
      <NumberedList
        ariaLabel="How to proceed"
        heading="How to proceed"
        items={response.answer.actions}
      />
      {styles.detailedTrace ? (
        <DetailedTrace steps={response.detailed_trace} />
      ) : null}
      <KnownLimitations limitations={response.known_limitations} />
    </div>
  );
}
