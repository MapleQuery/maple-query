"use client";

import * as React from "react";
import { Database, ListTree, ShieldCheck, Table2 } from "lucide-react";
import { useInView } from "./reveal";

interface Card {
  kind: string;
  icon: React.ReactNode;
  title: string;
  body: string;
  accent: string;
}

const CARDS: Card[] = [
  {
    kind: "Dataset",
    icon: <Database className="h-3.5 w-3.5" />,
    title: "Federal contracts, disclosed",
    body: "contracts-2023-q1 · quarterly · 84 columns",
    accent: "bg-navy/10 text-navy",
  },
  {
    kind: "Columns",
    icon: <ListTree className="h-3.5 w-3.5" />,
    title: "department, prime_contract, spend_cad",
    body: "3 candidates, ranked by semantic distance",
    accent: "bg-teal/15 text-teal",
  },
  {
    kind: "SQL guard",
    icon: <ShieldCheck className="h-3.5 w-3.5" />,
    title: "Accepted · 42 MB dry run",
    body: "SELECT-only, document_id IN-list, LIMIT 100 clamp",
    accent: "bg-success/15 text-success",
  },
  {
    kind: "Result",
    icon: <Table2 className="h-3.5 w-3.5" />,
    title: "12 rows, 8 columns",
    body: "Streamed into the answer as inline citations",
    accent: "bg-coral/15 text-navy",
  },
];

export function EvidenceDemo() {
  const [ref, inView] = useInView<HTMLDivElement>();
  const [visible, setVisible] = React.useState(0);

  React.useEffect(() => {
    if (!inView) {
      setVisible(0);
      return;
    }
    if (visible >= CARDS.length) {
      const t = window.setTimeout(() => setVisible(0), 3800);
      return () => window.clearTimeout(t);
    }
    const t = window.setTimeout(() => setVisible((n) => n + 1), 520);
    return () => window.clearTimeout(t);
  }, [inView, visible]);

  return (
    <div
      ref={ref}
      className="relative overflow-hidden rounded-2xl border border-hairline bg-surface-soft/70 p-4 shadow-xl"
    >
      <div className="mb-3 flex items-center justify-between">
        <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-muted">
          Evidence rail
        </span>
        <span className="font-mono text-[10px] text-muted">live</span>
      </div>
      <div className="space-y-2.5">
        {CARDS.map((c, i) => (
          <div
            key={c.title}
            className={
              "rounded-xl border border-hairline bg-white p-3 transition-all duration-500 " +
              (i < visible
                ? "opacity-100 translate-y-0"
                : "opacity-0 translate-y-2")
            }
          >
            <div className="flex items-center gap-2">
              <span
                className={
                  "inline-flex items-center gap-1 rounded px-1.5 py-0.5 font-mono text-[9px] font-semibold uppercase tracking-[0.14em] " +
                  c.accent
                }
              >
                {c.icon}
                {c.kind}
              </span>
            </div>
            <p className="mt-1.5 text-[12.5px] font-medium text-ink">
              {c.title}
            </p>
            <p className="mt-0.5 font-mono text-[10.5px] text-muted">
              {c.body}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
