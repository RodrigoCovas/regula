import { AnswerSurface } from "../components/AnswerSurface";
import { ScenarioForm } from "../components/ScenarioForm";
import {
  demoAnalyzeResponse,
  insufficientEvidenceResponse,
  notAvailableResponse,
} from "../lib/fixtures";

export default function Page() {
  return (
    <main className="mx-auto max-w-4xl space-y-12 px-4 py-10">
      <header className="space-y-2">
        <h1 className="text-3xl font-bold">Regula</h1>
        <p className="text-slate-600">
          Regulatory research and compliance assistant — describe a Scenario and
          ask a Regulatory question; the id derives from the description and the
          answer surface below mirrors the backend contract.
        </p>
      </header>

      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-slate-800">
          Analyze a scenario
        </h2>
        <ScenarioForm />
      </section>

      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-slate-800">
          Canonical demo scenario — full answer surface
        </h2>
        <AnswerSurface response={demoAnalyzeResponse} />
      </section>

      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-slate-800">
          Insufficient evidence
        </h2>
        <AnswerSurface response={insufficientEvidenceResponse} />
      </section>

      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-slate-800">Not available</h2>
        <AnswerSurface response={notAvailableResponse} />
      </section>
    </main>
  );
}
