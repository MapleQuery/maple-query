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

  const datasets = pick(stats?.datasets, failed, "3,700+");
  const documents = pick(stats?.documents, failed, "14,000+");
  const rows = pick(stats?.rows, failed, "Millions");

  return (
    <dl className="mt-12 grid max-w-xl gap-6 sm:grid-cols-3">
      <Callout
        headline={datasets ?? "—"}
        label="Datasets"
        sublabel="unique packages on open.canada.ca"
        loading={datasets === null}
      />
      <Callout
        headline={documents ?? "—"}
        label="Documents"
        sublabel="CSV files within them"
        loading={documents === null}
      />
      <Callout
        headline={rows ?? "—"}
        label="Rows"
        sublabel="joinable across every dataset"
        loading={rows === null}
      />
    </dl>
  );
}

function pick(
  live: number | undefined,
  failed: boolean,
  fallback: string,
): string | null {
  if (typeof live === "number") return formatInt(live);
  if (failed) return fallback;
  return null;
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
