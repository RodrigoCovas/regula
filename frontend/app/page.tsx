import { AnalysisSection } from "../components/AnalysisSection";

export default function Page() {
  return (
    <main className="mx-auto max-w-4xl space-y-12 px-4 py-10">
      <header className="space-y-2">
        <h1 className="text-3xl font-bold">Regula</h1>
        <p className="text-slate-600">
          Regulatory research and compliance assistant — describe a Scenario and
          ask a Regulatory question. The id derives from the description, the
          frontend polls the backend for progress, and the answer renders through
          the full contract surface.
        </p>
      </header>

      <AnalysisSection />
    </main>
  );
}
