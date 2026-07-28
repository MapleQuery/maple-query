"use client";

import * as React from "react";
import { BarChart3, LineChart } from "lucide-react";
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
 * A chart of a result set, plus the controls to re-aim it.
 *
 * It draws itself without being asked: `resolveChartSpec` infers a form
 * from the rows, so inserting a chart is one click rather than a
 * configuration exercise. The controls only ever write *overrides*,
 * never a full spec, so rows that changed under it fall back to the new
 * inference instead of pointing at a column that no longer exists.
 *
 * Takes plain rows and knows nothing about notebooks, so chat and
 * explorer can mount it as-is.
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

  if (!spec || !svg) {
    return (
      <div className="rounded-xl border border-dashed border-hairline bg-surface-soft/40 px-4 py-6 text-sm text-muted">
        These rows have no column pair to plot — a chart needs a category
        and a number.
      </div>
    );
  }

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
