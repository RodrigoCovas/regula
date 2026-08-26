export function NumberedList({
  ariaLabel,
  heading,
  items,
}: {
  ariaLabel: string;
  heading: string;
  items: string[];
}) {
  return (
    <section aria-label={ariaLabel}>
      <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">
        {heading}
      </h2>
      <ol className="list-decimal space-y-2 pl-5 text-sm text-slate-700">
        {items.map((item, index) => (
          <li key={index}>{item}</li>
        ))}
      </ol>
    </section>
  );
}
