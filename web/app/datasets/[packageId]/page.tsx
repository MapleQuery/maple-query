"use client";

import * as React from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { ArrowLeft, Copy, Check, Download, Loader2 } from "lucide-react";
import { getDatasetColumns, getDatasetDocuments, listDatasets } from "@/lib/api";
import type { ColumnInfo, DatasetSummary, DocumentInfo } from "@/lib/types";

export default function DatasetDetailPage() {
  const params = useParams<{ packageId: string }>();
  const packageId = params?.packageId ?? "";

  const [summary, setSummary] = React.useState<DatasetSummary | null>(null);
  const [columns, setColumns] = React.useState<ColumnInfo[]>([]);
  const [documents, setDocuments] = React.useState<DocumentInfo[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    if (!packageId) return;
    const controller = new AbortController();
    setLoading(true);
    setError(null);

    Promise.all([
      // The backend has no single-dataset endpoint; the columns response carries
      // the package_id, and we lean on /datasets?q= for the human-readable
      // summary.
      getDatasetColumns(packageId, controller.signal),
      listDatasets({
        q: packageId,
        limit: 5,
        signal: controller.signal,
      }).catch(() => ({ datasets: [], total: 0 })),
      // Source files are additive; a failure hides the section rather
      // than erroring the whole page.
      getDatasetDocuments(packageId, controller.signal).catch(() => ({
        package_id: packageId,
        documents: [],
      })),
    ])
      .then(([cols, list, docs]) => {
        setColumns(cols.columns);
        setDocuments(docs.documents);
        setSummary(
          list.datasets.find((d) => d.package_id === packageId) ?? {
            package_id: packageId,
            title: packageId,
            summary: "",
          },
        );
      })
      .catch((err) => {
        if ((err as Error).name === "AbortError") return;
        setError((err as Error).message || "Failed to load dataset");
      })
      .finally(() => setLoading(false));

    return () => controller.abort();
  }, [packageId]);

  return (
    <main className="mx-auto max-w-5xl px-4 py-10 md:px-6">
      <Link
        href="/datasets"
        className="mb-6 inline-flex items-center gap-1 text-sm text-muted hover:text-ink"
      >
        <ArrowLeft className="h-4 w-4" />
        All datasets
      </Link>

      {loading ? (
        <div className="flex items-center gap-2 text-sm text-muted">
          <Loader2 className="h-4 w-4 animate-spin" /> Loading…
        </div>
      ) : error ? (
        <div className="rounded-lg border border-error/30 bg-error/10 px-4 py-3 text-sm text-error">
          {error}
        </div>
      ) : summary ? (
        <>
          <header className="mb-8">
            <div className="mb-3 flex items-center gap-2">
              <PackageIdChip packageId={packageId} />
              {summary.grain && (
                <span className="rounded bg-navy/10 px-2 py-0.5 font-mono text-[10px] font-semibold text-navy">
                  {summary.grain}
                </span>
              )}
            </div>
            <h1 className="font-display text-3xl font-medium tracking-tight text-ink md:text-4xl">
              {summary.title || packageId}
            </h1>
            {summary.summary && (
              <p className="mt-3 max-w-3xl text-body">{summary.summary}</p>
            )}
            <MetaGrid summary={summary} columnCount={columns.length} />
          </header>

          <section>
            <h2 className="mb-3 font-display text-lg font-medium text-ink">
              Columns
              <span className="ml-2 font-sans text-xs text-muted">
                {columns.length} tagged
              </span>
            </h2>
            <div className="overflow-hidden rounded-xl border border-hairline bg-white">
              <table className="w-full text-left text-sm">
                <thead className="border-b border-hairline bg-surface-soft">
                  <tr>
                    <Th>Name</Th>
                    <Th>Semantic type</Th>
                    <Th>Description</Th>
                    <Th>Sample values</Th>
                  </tr>
                </thead>
                <tbody>
                  {columns.map((c) => (
                    <tr
                      key={c.column_name}
                      className="border-b border-hairline/60 last:border-0 hover:bg-surface-soft/60"
                    >
                      <td className="whitespace-nowrap px-4 py-3 font-mono text-[12.5px] font-medium text-ink">
                        {c.column_name}
                      </td>
                      <td className="whitespace-nowrap px-4 py-3">
                        {c.semantic_type ? (
                          <span className="rounded-full bg-coral/15 px-2 py-0.5 font-mono text-[10px] font-semibold text-navy">
                            {c.semantic_type}
                          </span>
                        ) : null}
                      </td>
                      <td className="max-w-md px-4 py-3 text-body">
                        {c.description || null}
                      </td>
                      <td className="px-4 py-3 font-mono text-[11px] text-muted">
                        {(c.sample_values ?? [])
                          .slice(0, 3)
                          .map((v) => String(v))
                          .join(", ")}
                      </td>
                    </tr>
                  ))}
                  {columns.length === 0 && (
                    <tr>
                      <td colSpan={4} className="px-4 py-6 text-center text-sm text-muted">
                        No column metadata indexed yet.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>

          {documents.length > 0 && (
            <section className="mt-8">
              <h2 className="mb-3 font-display text-lg font-medium text-ink">
                Source files
                <span className="ml-2 font-sans text-xs text-muted">
                  {documents.length} file{documents.length === 1 ? "" : "s"}
                </span>
              </h2>
              <div className="overflow-hidden rounded-xl border border-hairline bg-white">
                <table className="w-full text-left text-sm">
                  <thead className="border-b border-hairline bg-surface-soft">
                    <tr>
                      <Th>Title</Th>
                      <Th>Format</Th>
                      <Th>Rows</Th>
                      <Th>Published</Th>
                      <Th>Download</Th>
                    </tr>
                  </thead>
                  <tbody>
                    {documents.map((d) => (
                      <tr
                        key={d.document_id}
                        className="border-b border-hairline/60 last:border-0 hover:bg-surface-soft/60"
                      >
                        <td className="max-w-md px-4 py-3 text-body">
                          {d.title || `${d.document_id.slice(0, 8)}…`}
                          {d.is_representative && (
                            <span
                              className="ml-2 rounded-full bg-coral/15 px-2 py-0.5 font-mono text-[10px] font-semibold text-navy"
                              title="The columns and sample values above were extracted from this file"
                            >
                              Enriched
                            </span>
                          )}
                        </td>
                        <td className="whitespace-nowrap px-4 py-3 font-mono text-[11px] uppercase text-muted">
                          {d.file_format}
                        </td>
                        <td className="whitespace-nowrap px-4 py-3 font-mono text-[12.5px] text-ink">
                          {d.row_count != null ? d.row_count.toLocaleString() : ""}
                        </td>
                        <td className="whitespace-nowrap px-4 py-3 font-mono text-[12.5px] text-muted">
                          {d.published_date ?? ""}
                        </td>
                        <td className="whitespace-nowrap px-4 py-3">
                          <a
                            href={d.source_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            download
                            className="inline-flex items-center gap-1 text-sm text-navy hover:text-ink"
                          >
                            <Download className="h-4 w-4" />
                            <span className="sr-only">
                              Download {d.title || d.document_id}
                            </span>
                          </a>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          <section className="mt-8">
            <div className="flex items-center justify-between rounded-xl border border-hairline bg-white p-5">
              <div>
                <h3 className="font-display text-lg font-medium text-ink">
                  Ready to query?
                </h3>
                <p className="text-sm text-muted">
                  Open the chat with this dataset as the starting frame.
                </p>
              </div>
              <Link
                href={`/chat?ctx=${encodeURIComponent(packageId)}`}
                className="rounded-md bg-coral px-4 py-2 text-sm font-medium text-ink transition-colors hover:bg-coral-active focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy"
              >
                Ask a question
              </Link>
            </div>
          </section>
        </>
      ) : null}
    </main>
  );
}

function PackageIdChip({ packageId }: { packageId: string }) {
  const [copied, setCopied] = React.useState(false);
  return (
    <button
      type="button"
      onClick={async () => {
        await navigator.clipboard.writeText(packageId);
        setCopied(true);
        setTimeout(() => setCopied(false), 1400);
      }}
      className="inline-flex items-center gap-1.5 rounded-md border border-hairline bg-white px-2 py-1 font-mono text-[11px] text-muted transition-colors hover:text-ink"
    >
      {copied ? <Check className="h-3 w-3 text-success" /> : <Copy className="h-3 w-3" />}
      {packageId}
    </button>
  );
}

function Th({ children }: { children: React.ReactNode }) {
  return (
    <th className="px-4 py-2.5 text-[11px] font-semibold uppercase tracking-wider text-muted">
      {children}
    </th>
  );
}

function MetaGrid({
  summary,
  columnCount,
}: {
  summary: DatasetSummary;
  columnCount: number;
}) {
  const entries: [string, string | number | null | undefined][] = [
    ["Columns", columnCount],
    ["Coverage", summary.date_range_start && summary.date_range_end
      ? `${summary.date_range_start} → ${summary.date_range_end}`
      : summary.date_range_start ?? ""],
    ["Measures", (summary.measures ?? []).slice(0, 3).join(", ")],
    ["Dimensions", (summary.dimensions ?? []).slice(0, 3).join(", ")],
  ];
  return (
    <dl className="mt-6 grid gap-px overflow-hidden rounded-xl border border-hairline bg-hairline sm:grid-cols-2 lg:grid-cols-4">
      {entries.map(([k, v]) => (
        <div key={k} className="bg-white px-4 py-3">
          <dt className="text-[11px] font-semibold uppercase tracking-wider text-muted">
            {k}
          </dt>
          <dd className="mt-1 truncate font-mono text-sm text-ink">
            {v == null || v === "" ? "" : String(v)}
          </dd>
        </div>
      ))}
    </dl>
  );
}
