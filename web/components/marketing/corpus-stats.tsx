"use client";

import * as React from "react";
import { getCorpusStats, type CorpusStats as CorpusStatsT } from "@/lib/api";

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

  const docs = stats ? formatInt(stats.documents) : failed ? "3,700+" : null;
  const rows = stats ? formatInt(stats.rows) : failed ? "Millions" : null;

  return (
    <dl className="mt-12 grid max-w-xl gap-6 sm:grid-cols-3">
      <Callout
        headline={docs ?? "—"}
        label="Federal documents"
        sublabel="indexed to date"
        loading={docs === null}
      />
      <Callout
        headline={rows ?? "—"}
        label="Joinable rows"
        sublabel="across every dataset"
        loading={rows === null}
      />
      <Callout
        headline="Zero"
        label="Uncited answers"
        sublabel="the guard refuses first"
      />
    </dl>
  );
}

function Callout({
  headline,
  label,
  sublabel,
  loading = false,
}: {
  headline: string;
  label: string;
  sublabel: string;
  loading?: boolean;
}) {
  return (
    <div className="border-l-2 border-ink/10 pl-4">
      <dd
        className={
          "font-display text-2xl font-semibold tracking-tight text-ink tabular-nums md:text-3xl " +
          (loading ? "animate-pulse text-ink/30" : "")
        }
      >
        {loading ? "0,000" : headline}
      </dd>
      <dt className="mt-1 text-sm font-medium text-ink">{label}</dt>
      <p className="text-xs text-muted">{sublabel}</p>
    </div>
  );
}

function formatInt(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "0";
  return Math.round(n).toLocaleString("en-US");
}
