export function KnownLimitations({ limitations }: { limitations: string[] }) {
  return (
    <section
      aria-label="Known limitations"
      className="rounded-lg border border-slate-200 bg-slate-100 p-4"
    >
      <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">
        Known limitations
      </h2>
      <ul className="list-disc space-y-1 pl-5 text-sm text-slate-600">
        {limitations.map((limitation, index) => (
          <li key={index}>{limitation}</li>
        ))}
      </ul>
    </section>
  );
}
