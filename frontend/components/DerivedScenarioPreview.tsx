import { deriveScenarioId } from "../lib/scenario-id";

export function DerivedScenarioPreview({
  description,
}: {
  description: string;
}) {
  return (
    <div className="rounded-md border border-slate-200 bg-slate-50 p-4 text-sm">
      <p className="font-medium text-slate-700">Derived scenario</p>
      {description ? (
        <dl className="mt-2 space-y-1">
          <div className="flex gap-2">
            <dt className="text-slate-500">Title</dt>
            <dd className="break-words">{description}</dd>
          </div>
          <div className="flex gap-2">
            <dt className="text-slate-500">Id</dt>
            <dd className="break-all font-mono">{deriveScenarioId(description)}</dd>
          </div>
        </dl>
      ) : (
        <p className="text-slate-500">
          Title and id derive from the description as you type.
        </p>
      )}
    </div>
  );
}
