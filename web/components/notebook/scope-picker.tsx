"use client";

import * as React from "react";
import { Check, Loader2, Search } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { listDatasets } from "@/lib/api";
import {
  rememberDatasetTitles,
  useDatasetTitles,
} from "@/lib/dataset-titles";
import type { DatasetSummary } from "@/lib/types";

export interface ScopePickerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Package ids currently pinned to the block. */
  selected: string[];
  onChange: (next: string[]) => void;
}

const DEBOUNCE_MS = 250;
const PAGE = 20;

/**
 * Pick the datasets a query block is scoped to.
 *
 * Scope already existed on a block, but only the agent could set it — a
 * user could accept a suggestion that happened to carry one, or drop the
 * one they were given, and nothing else. That left the interesting
 * failure unrecoverable: when a question finds the wrong dataset, the
 * user can see which one it should have used and has no way to say so.
 *
 * Selection applies immediately rather than on a confirm step, because
 * the block re-sends its scope on the next run and nothing here is
 * destructive — the same reason "drop scope" needs no confirmation.
 */
export function ScopePicker({
  open,
  onOpenChange,
  selected,
  onChange,
}: ScopePickerProps) {
  const [query, setQuery] = React.useState("");
  const [results, setResults] = React.useState<DatasetSummary[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  /** A pinned dataset is usually not on the current result page, so its
   * chip reads from the shared title cache — which backfills any id it
   * does not know and keeps the answer in localStorage. */
  const titles = useDatasetTitles(selected);

  React.useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      setLoading(true);
      setError(null);
      listDatasets({
        q: query.trim() || undefined,
        limit: PAGE,
        signal: controller.signal,
      })
        .then((res) => {
          setResults(res.datasets);
          // Every dataset the picker shows is a title the chips get for
          // free, so opening the picker warms the shared cache.
          rememberDatasetTitles(res.datasets);
        })
        .catch((err: unknown) => {
          if (controller.signal.aborted) return;
          setError(err instanceof Error ? err.message : "Search failed");
          setResults([]);
        })
        .finally(() => {
          if (!controller.signal.aborted) setLoading(false);
        });
    }, DEBOUNCE_MS);
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [open, query]);

  const toggle = (packageId: string) => {
    onChange(
      selected.includes(packageId)
        ? selected.filter((p) => p !== packageId)
        : [...selected, packageId],
    );
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Scope to datasets</DialogTitle>
          <DialogDescription>
            Narrow this block to the datasets you want it to use. The scope
            is saved with the block and re-sent every time it runs.
          </DialogDescription>
        </DialogHeader>

        <div className="flex items-center gap-2 rounded-md border border-hairline bg-white px-3 py-2">
          <Search className="h-4 w-4 shrink-0 text-muted" />
          <input
            autoFocus
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search datasets…"
            className="w-full bg-transparent text-sm text-ink placeholder:text-muted focus:outline-none"
          />
          {loading && <Loader2 className="h-4 w-4 animate-spin text-muted" />}
        </div>

        {selected.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-xs text-muted">Scoped to:</span>
            {selected.map((id) => (
              <button
                key={id}
                type="button"
                onClick={() => toggle(id)}
                title="Remove from scope"
                className="max-w-[280px] truncate rounded bg-surface-soft px-1.5 py-0.5 text-[11px] text-navy hover:bg-error/10 hover:text-error focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy"
              >
                {titles[id] ?? id} ×
              </button>
            ))}
          </div>
        )}

        <div className="max-h-[42vh] overflow-y-auto rounded-md border border-hairline bg-white">
          {error ? (
            <p className="px-4 py-6 text-center text-sm text-error">{error}</p>
          ) : results.length === 0 ? (
            <p className="px-4 py-6 text-center text-sm text-muted">
              {loading ? "Searching…" : "No datasets match that search."}
            </p>
          ) : (
            <ul className="divide-y divide-hairline">
              {results.map((d) => {
                const isSelected = selected.includes(d.package_id);
                return (
                  <li key={d.package_id}>
                    <button
                      type="button"
                      onClick={() => toggle(d.package_id)}
                      aria-pressed={isSelected}
                      className="flex w-full items-start gap-3 px-3 py-2.5 text-left transition-colors hover:bg-surface-soft focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-navy"
                    >
                      <span
                        className={`mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded border ${
                          isSelected
                            ? "border-navy bg-navy text-white"
                            : "border-hairline"
                        }`}
                      >
                        {isSelected && <Check className="h-3 w-3" />}
                      </span>
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-sm text-ink">
                          {d.title || d.package_id}
                        </span>
                        {d.summary && (
                          <span className="line-clamp-2 text-xs text-muted">
                            {d.summary}
                          </span>
                        )}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>

        <div className="flex justify-end">
          <button
            type="button"
            onClick={() => onOpenChange(false)}
            className="rounded-md bg-coral px-3 py-1.5 text-sm font-medium text-ink hover:bg-coral-active focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy"
          >
            Done
          </button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
