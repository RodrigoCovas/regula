"use client";

import type { Mode } from "../lib/contract";

export function ModeToggle({
  mode,
  onChange,
}: {
  mode: Mode;
  onChange: (mode: Mode) => void;
}) {
  return (
    <fieldset className="space-y-2">
      <legend className="text-sm font-medium text-slate-700">Mode</legend>
      <div className="flex flex-col gap-2 sm:flex-row sm:gap-8">
        <label className="flex items-start gap-2">
          <input
            type="radio"
            name="mode"
            value="demo"
            checked={mode === "demo"}
            onChange={() => onChange("demo")}
            className="mt-1"
          />
          <span>
            <span className="block text-sm font-semibold text-slate-800">
              Demo
            </span>
            <span className="block text-xs text-slate-500">
              The canonical Spanish fintech scenario — no API key needed.
            </span>
          </span>
        </label>
        <label className="flex items-start gap-2">
          <input
            type="radio"
            name="mode"
            value="live"
            checked={mode === "live"}
            onChange={() => onChange("live")}
            className="mt-1"
          />
          <span>
            <span className="block text-sm font-semibold text-slate-800">
              Live
            </span>
            <span className="block text-xs text-slate-500">
              Any scenario over the ingested Corpus — needs the Live
              prerequisites.
            </span>
          </span>
        </label>
      </div>
    </fieldset>
  );
}
