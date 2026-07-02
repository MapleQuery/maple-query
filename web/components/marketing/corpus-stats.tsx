"use client";

import * as React from "react";
import { getCorpusStats, type CorpusStats as CorpusStatsT } from "@/lib/api";

interface StatDef {
  label: string;
  render: (s: CorpusStatsT | null) => string;
}

const STATS: StatDef[] = [
  {
    label: "Documents",
    render: (s) => (s ? formatCompact(s.documents) + "+" : "—"),
  },
  {
    label: "Rows of data",
    render: (s) => (s ? formatCompact(s.rows) + "+" : "—"),
  },
  {
    label: "Answers cited or refused",
    render: () => "100%",
  },
];

export function CorpusStats() {
  const [stats, setStats] = React.useState<CorpusStatsT | null>(null);
  const [failed, setFailed] = React.useState(false);

  React.useEffect(() => {
    const controller = new AbortController();
    getCorpusStats(controller.signal)
      .then((s) => setStats(s))
      .catch((err) => {
        if ((err as Error).name === "AbortError") return;
        setFailed(true);
      });
    return () => controller.abort();
  }, []);

  return (
    <dl className="mt-12 grid max-w-lg grid-cols-3 gap-6">
      {STATS.map((s) => {
        const value =
          failed && s.label !== "Answers cited or refused"
            ? "—"
            : s.render(stats);
        return (
          <div key={s.label}>
            <dd className="text-3xl font-semibold tracking-tight text-ink tabular-nums">
              {value}
            </dd>
            <dt className="mt-1 text-xs text-muted">{s.label}</dt>
          </div>
        );
      })}
    </dl>
  );
}

function formatCompact(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "0";
  if (n >= 1_000_000_000) return trim(n / 1_000_000_000) + "B";
  if (n >= 1_000_000) return trim(n / 1_000_000) + "M";
  if (n >= 10_000) return Math.round(n / 1_000) + "K";
  if (n >= 1_000) return trim(n / 1_000) + "K";
  return n.toLocaleString();
}

function trim(n: number): string {
  return (Math.round(n * 10) / 10).toString().replace(/\.0$/, "");
}
