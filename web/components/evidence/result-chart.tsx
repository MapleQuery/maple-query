"use client";

import * as React from "react";
import { BarChart3, LineChart, X } from "lucide-react";
import {
  chartData,
  renderChartSvg,
  resolveChartSpec,
  type ChartOverrides,
  type ChartType,
} from "@/lib/chart";

export interface ResultChartProps {
  rows: Record<string, unknown>[];
  overrides?: ChartOverrides;
  onChange: (next: ChartOverrides) => void;
}

/**
 * The chart under a result table.
 *
 * It draws itself the first time without being asked: `resolveChartSpec`
 * infers a form from the rows, and a result the inference declines
 * simply renders nothing. That is deliberate — a chart the user has to
 * go and configure is one nobody sees, and the inference is free.
 *
 * The controls only ever write *overrides*, never a full spec, so a
 * block that was re-run against different columns falls back to the new
 * inference instead of pointing at a column that no longer exists.
 */
export function ResultChart({ rows, overrides, onChange }: ResultChartProps) {
  const spec = React.useMemo(
    () => resolveChartSpec(rows, overrides),
    [rows, overrides],
  );
  const columns = React.useMemo(
    () => Object.keys(rows[0] ?? {}),
    [rows],
  );

  const svg = React.useMemo(() => {
    if (!spec) return "";
    return renderChartSvg(chartData(rows, spec), spec, {
      valueLabel: spec.valueColumn,
    });
  }, [rows, spec]);

  if (overrides?.hidden) {
    return (
      <button
        type="button"
        onClick={() => onChange({ ...overrides, hidden: false })}
        className="inline-flex items-center gap-1.5 rounded-md border border-hairline bg-white px-2.5 py-1.5 text-xs text-muted transition-colors hover:border-navy hover:text-navy focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy"
      >
        <BarChart3 className="h-3.5 w-3.5" /> Show chart
      </button>
    );
  }

  if (!spec || !svg) return null;

  const setType = (type: ChartType) => onChange({ ...overrides, type });

  return (
    <figure className="rounded-xl border border-hairline bg-white p-4">
      <figcaption className="mb-3 flex flex-wrap items-center gap-2">
        <div className="flex rounded-md border border-hairline p-0.5">
          <TypeButton
            active={spec.type === "bar"}
            onClick={() => setType("bar")}
            label="Bar"
            icon={<BarChart3 className="h-3.5 w-3.5" />}
          />
          <TypeButton
            active={spec.type === "line"}
            onClick={() => setType("line")}
            label="Line"
            icon={<LineChart className="h-3.5 w-3.5" />}
          />
        </div>
        <ColumnSelect
          label="by"
          value={spec.categoryColumn}
          options={columns}
          onChange={(categoryColumn) =>
            onChange({ ...overrides, categoryColumn })
          }
        />
        <ColumnSelect
          label="of"
          value={spec.valueColumn}
          options={columns}
          onChange={(valueColumn) => onChange({ ...overrides, valueColumn })}
        />
        <button
          type="button"
          onClick={() => onChange({ ...overrides, hidden: true })}
          aria-label="Hide chart"
          className="ml-auto rounded p-1 text-muted hover:bg-surface-soft hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </figcaption>
      {/* The markup is built by our own renderer, which XML-escapes every
          value that comes from the result set; nothing here is raw data. */}
      <div
        className="[&>svg]:h-auto [&>svg]:w-full"
        dangerouslySetInnerHTML={{ __html: svg }}
      />
    </figure>
  );
}

function TypeButton({
  active,
  onClick,
  label,
  icon,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
  icon: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`inline-flex items-center gap-1.5 rounded px-2 py-1 text-xs transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy ${
        active
          ? "bg-surface-soft font-medium text-ink"
          : "text-muted hover:text-ink"
      }`}
    >
      {icon}
      {label}
    </button>
  );
}

function ColumnSelect({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
}) {
  return (
    <label className="inline-flex items-center gap-1 text-xs text-muted">
      {label}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="max-w-[180px] truncate rounded-md border border-hairline bg-white px-1.5 py-1 text-xs text-ink focus:border-navy focus:outline-none"
      >
        {options.map((c) => (
          <option key={c} value={c}>
            {c}
          </option>
        ))}
      </select>
    </label>
  );
}
