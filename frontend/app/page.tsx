import { AnswerSurface } from "../components/AnswerSurface";
import {
  demoAnalyzeResponse,
  insufficientEvidenceResponse,
  noopDemoMissResponse,
  notAvailableResponse,
} from "../lib/fixtures";

export default function Page() {
  return (
    <main className="mx-auto max-w-4xl space-y-12 px-4 py-10">
      <header className="space-y-2">
        <h1 className="text-3xl font-bold">Regula</h1>
        <p className="text-slate-600">
          Regulatory research and compliance assistant — the full answer
          surface, rendered from typed fixtures that mirror the backend
          contract.
        </p>
      </header>

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

      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-slate-800">
          Unknown scenario in Demo mode — Not available
        </h2>
        <AnswerSurface response={noopDemoMissResponse} />
      </section>
    </main>
  );
}
