"use client";

import * as React from "react";
import { useInView } from "./reveal";

const ROWS = [
  { department: "Public Services", total: "$612,340,118" },
  { department: "National Defence", total: "$487,010,220" },
  { department: "Shared Services", total: "$318,772,655" },
  { department: "Employment & Social Dev", total: "$204,918,073" },
  { department: "Global Affairs", total: "$142,308,914" },
];

export function NotebookDemo() {
  const [ref, inView] = useInView<HTMLDivElement>();
  const [visibleRows, setVisibleRows] = React.useState(0);

  React.useEffect(() => {
    if (!inView) {
      setVisibleRows(0);
      return;
    }
    if (visibleRows >= ROWS.length) {
      const t = window.setTimeout(() => setVisibleRows(0), 4200);
      return () => window.clearTimeout(t);
    }
    const t = window.setTimeout(
      () => setVisibleRows((n) => n + 1),
      380,
    );
    return () => window.clearTimeout(t);
  }, [inView, visibleRows]);

  return (
    <div
      ref={ref}
      className="relative overflow-hidden rounded-2xl border border-hairline bg-white shadow-xl"
    >
      <div className="flex items-center justify-between border-b border-hairline bg-surface-soft/60 px-4 py-2.5">
        <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted">
          Notebook · draft
        </span>
        <span className="font-mono text-[10px] text-muted">Autosaved</span>
      </div>

      <div className="space-y-4 px-5 py-5">
        <h3 className="font-display text-base font-semibold text-ink">
          IT consulting spend, 2023
        </h3>
        <p className="text-[13px] leading-relaxed text-body">
          A quick rollup of prime-contract spend across the top federal
          departments. Numbers land straight from the query below — no
          copy-paste, no drift.
        </p>

        <div className="rounded-lg border border-hairline bg-surface-soft/50 p-3">
          <div className="flex items-center gap-2 text-[10px] text-muted">
            <span className="font-mono uppercase tracking-[0.14em]">
              Query
            </span>
            <span>Top 5 departments by consulting spend</span>
          </div>
          <div className="mt-2 overflow-hidden rounded-md border border-hairline bg-white">
            <table className="w-full text-left text-[12px]">
              <thead className="border-b border-hairline bg-surface-soft/70 text-[10px] uppercase tracking-wider text-muted">
                <tr>
                  <th className="px-3 py-1.5 font-semibold">Department</th>
                  <th className="px-3 py-1.5 text-right font-semibold">
                    Total
                  </th>
                </tr>
              </thead>
              <tbody>
                {ROWS.map((r, i) => (
                  <tr
                    key={r.department}
                    className={
                      "border-b border-hairline/60 transition-all duration-500 last:border-0 " +
                      (i < visibleRows
                        ? "opacity-100 translate-y-0"
                        : "opacity-0 translate-y-1")
                    }
                  >
                    <td className="px-3 py-1.5 text-ink">{r.department}</td>
                    <td className="px-3 py-1.5 text-right font-mono tabular-nums text-navy">
                      {r.total}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <p className="text-[13px] leading-relaxed text-body">
          Rerun the query and the paragraph above updates with the new
          totals. Export the notebook and every figure carries its
          citation.
        </p>
      </div>
    </div>
  );
}
