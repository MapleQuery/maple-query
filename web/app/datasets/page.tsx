"use client";

import * as React from "react";
import Link from "next/link";
import { Search, ArrowUpRight, Loader2 } from "lucide-react";
import { Input } from "@/components/ui/input";
import { listDatasets } from "@/lib/api";
import type { DatasetSummary } from "@/lib/types";
import { cn, truncate } from "@/lib/utils";

export default function DatasetsPage() {
  const [q, setQ] = React.useState("");
  const [committedQ, setCommittedQ] = React.useState("");
  const [items, setItems] = React.useState<DatasetSummary[]>([]);
  const [total, setTotal] = React.useState<number | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    const handle = setTimeout(() => setCommittedQ(q.trim()), 250);
    return () => clearTimeout(handle);
  }, [q]);

  React.useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    listDatasets({
      q: committedQ || undefined,
      limit: 40,
      signal: controller.signal,
    })
      .then((r) => {
        setItems(r.datasets);
        setTotal(r.total);
      })
      .catch((err) => {
        if ((err as Error).name === "AbortError") return;
        setError((err as Error).message || "Failed to load datasets");
        setItems([]);
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [committedQ]);

  return (
    <main className="mx-auto max-w-6xl px-4 py-10 md:px-6">
      <header className="mb-8">
        <p className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-navy">
          Corpus
        </p>
        <h1 className="font-display text-3xl font-medium tracking-tight text-ink md:text-4xl">
          Browse the datasets MapleQuery can cite
        </h1>
        <p className="mt-3 max-w-2xl text-body">
          Canadian federal open data, indexed by semantic type and searchable by
          plain-language description. Click a dataset to see its columns and
          sample rows.
        </p>
      </header>

      <div className="mb-6 flex items-center gap-2 rounded-xl border border-hairline bg-white px-3 py-2 shadow-sm focus-within:ring-2 focus-within:ring-navy">
        <Search className="h-4 w-4 text-muted" />
        <Input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search: housing grants, IT contracts, immigration…"
          className="border-0 bg-transparent shadow-none focus-visible:ring-0"
        />
        {loading && <Loader2 className="h-4 w-4 animate-spin text-muted" />}
        {total != null && !loading && (
          <span className="ml-auto shrink-0 font-mono text-[11px] text-muted">
            {total.toLocaleString()} datasets
          </span>
        )}
      </div>

      {error && (
        <div className="mb-6 rounded-lg border border-error/30 bg-error/10 px-4 py-3 text-sm text-error">
          {error}
        </div>
      )}

      {loading && items.length === 0 ? (
        <SkeletonGrid />
      ) : items.length === 0 ? (
        <p className="rounded-xl border border-dashed border-hairline bg-white/50 px-6 py-12 text-center text-sm text-muted">
          No datasets match &ldquo;{committedQ}&rdquo;.
        </p>
      ) : (
        <ul className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {items.map((d) => (
            <li key={d.package_id}>
              <Link
                href={`/datasets/${d.package_id}`}
                className="group flex h-full flex-col rounded-xl border border-hairline bg-white p-5 transition-all hover:-translate-y-0.5 hover:border-navy hover:shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy"
              >
                <div className="mb-2 flex items-start justify-between gap-2">
                  <span className="rounded bg-navy/10 px-2 py-0.5 font-mono text-[10px] font-semibold text-navy">
                    {d.grain ?? "dataset"}
                  </span>
                  {typeof d.distance === "number" && (
                    <span className="rounded bg-coral/15 px-2 py-0.5 font-mono text-[10px] font-semibold text-navy">
                      d={d.distance.toFixed(3)}
                    </span>
                  )}
                </div>
                <h3 className="font-display text-lg font-medium leading-snug text-ink">
                  {truncate(d.title || d.package_id, 90)}
                </h3>
                <p
                  className={cn(
                    "mt-2 line-clamp-4 flex-1 text-sm leading-relaxed text-body",
                  )}
                >
                  {d.summary || "No summary available."}
                </p>
                <div className="mt-4 flex items-center justify-between border-t border-hairline pt-3">
                  <span className="font-mono text-[10px] text-muted">
                    {truncate(d.package_id, 24)}
                  </span>
                  <span className="inline-flex items-center gap-1 text-xs font-medium text-navy group-hover:underline">
                    Open
                    <ArrowUpRight className="h-3 w-3" />
                  </span>
                </div>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}

function SkeletonGrid() {
  return (
    <ul className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      {Array.from({ length: 6 }).map((_, i) => (
        <li
          key={i}
          className="h-52 animate-pulse rounded-xl border border-hairline bg-white/60"
        />
      ))}
    </ul>
  );
}
