"use client";

import { cn } from "@/lib/utils";

export interface RowsTableProps {
  rows: Record<string, unknown>[];
  emptyMessage?: string;
  maxRows?: number;
  className?: string;
  caption?: React.ReactNode;
}

function formatCell(v: unknown): string {
  if (v === null || v === undefined) return "";
  if (typeof v === "number")
    return Number.isInteger(v) ? v.toLocaleString() : v.toString();
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

export function RowsTable({
  rows,
  emptyMessage = "No rows returned.",
  maxRows = 100,
  className,
  caption,
}: RowsTableProps) {
  if (rows.length === 0) {
    return (
      <div
        className={cn(
          "rounded-xl border border-dashed border-hairline bg-white/60 px-4 py-8 text-center text-sm text-muted",
          className,
        )}
      >
        {emptyMessage}
      </div>
    );
  }

  const first = rows[0] ?? {};
  const columns = Object.keys(first);
  const visible = rows.slice(0, maxRows);

  return (
    <div
      className={cn(
        "overflow-hidden rounded-xl border border-hairline bg-white",
        className,
      )}
    >
      {caption && (
        <div className="flex items-center justify-between border-b border-hairline bg-surface-soft px-4 py-2 font-mono text-[11px] text-muted">
          {caption}
        </div>
      )}
      <div className="max-h-[420px] overflow-auto">
        <table className="w-full border-collapse text-left text-sm">
          <thead className="sticky top-0 z-[1] bg-surface-soft">
            <tr>
              {columns.map((c) => (
                <th
                  key={c}
                  className="border-b border-hairline px-3 py-2 text-[11px] font-semibold uppercase tracking-wider text-muted"
                >
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="font-mono text-[12.5px] text-ink">
            {visible.map((row, i) => (
              <tr
                key={i}
                className="border-b border-hairline/60 last:border-0 hover:bg-surface-soft/60"
              >
                {columns.map((c) => (
                  <td
                    key={c}
                    className="whitespace-nowrap px-3 py-2 align-top"
                  >
                    {formatCell(row[c])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length > visible.length && (
        <div className="border-t border-hairline bg-surface-soft/60 px-4 py-2 text-[11px] text-muted">
          Showing first {visible.length} of {rows.length} rows.
        </div>
      )}
    </div>
  );
}
