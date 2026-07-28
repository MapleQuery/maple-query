"use client";

import * as React from "react";
import { AlertTriangle, BarChart3 } from "lucide-react";
import { ResultChart } from "@/components/evidence/result-chart";
import { chartSource, chartableSources } from "@/lib/notebook";
import { truncate } from "@/lib/utils";
import type {
  StoredNotebookBlock,
  StoredNotebookBlockChart,
} from "@/lib/storage";
import type { ChartOverrides } from "@/lib/chart";

export interface ChartBlockProps {
  block: StoredNotebookBlockChart;
  /** Every block in the notebook — the chart resolves its source from
   * these on each render, so it never shows numbers its source no
   * longer has. */
  blocks: StoredNotebookBlock[];
  onChangeSource: (sourceBlockId: string) => void;
  onChangeOverrides: (overrides: ChartOverrides) => void;
}

/**
 * A chart of another block's result, as a block in its own right.
 *
 * It reads its source's rows live rather than holding a copy, so a
 * re-run updates the chart and there is no way for it to disagree with
 * the table above it. The cost is that the reference can dangle — the
 * source can be deleted, or moved below the chart — and every one of
 * those states is rendered plainly instead of collapsing to an empty
 * box.
 */
export function ChartBlock({
  block,
  blocks,
  onChangeSource,
  onChangeOverrides,
}: ChartBlockProps) {
  const sources = React.useMemo(() => chartableSources(blocks), [blocks]);
  const source = chartSource(blocks, block.sourceBlockId);
  const rows = source?.result?.rows ?? [];

  const label = (b: (typeof sources)[number], i: number): string =>
    truncate(b.question.trim() || `Untitled query ${i + 1}`, 60);

  // A chart placed above the query it reads renders before that query
  // has run in a fresh session. Worth saying, because the fix is to move
  // one of them rather than to re-run anything.
  const sourceIndex = blocks.findIndex((b) => b.id === block.sourceBlockId);
  const selfIndex = blocks.findIndex((b) => b.id === block.id);
  const sourceIsBelow = sourceIndex > selfIndex;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <p className="font-mono text-[10px] uppercase tracking-wider text-muted">
          Chart
        </p>
        <label className="inline-flex min-w-0 items-center gap-1 text-xs text-muted">
          of
          <select
            value={block.sourceBlockId}
            onChange={(e) => onChangeSource(e.target.value)}
            className="max-w-[320px] truncate rounded-md border border-hairline bg-white px-1.5 py-1 text-xs text-ink focus:border-navy focus:outline-none"
          >
            {/* A dangling reference stays selectable, so the control
                shows what is actually broken instead of silently
                snapping to another query's numbers. */}
            {!source && (
              <option value={block.sourceBlockId}>
                (source removed)
              </option>
            )}
            {sources.map((b, i) => (
              <option key={b.id} value={b.id}>
                {label(b, i)}
              </option>
            ))}
          </select>
        </label>
      </div>

      {!source ? (
        <Empty
          message={
            sources.length > 0
              ? "The query this chart read from was removed. Pick another above."
              : "The query this chart read from was removed."
          }
        />
      ) : rows.length === 0 ? (
        <Empty
          message={
            sourceIsBelow
              ? "This chart sits above the query it reads. Move it below, or run that query."
              : "That query has not returned any rows yet. Run it and the chart draws itself."
          }
        />
      ) : (
        <ResultChart
          rows={rows}
          overrides={block.overrides}
          onChange={onChangeOverrides}
        />
      )}
    </div>
  );
}

function Empty({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-2 rounded-xl border border-dashed border-hairline bg-surface-soft/40 px-4 py-6 text-sm text-muted">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber" />
      <span>{message}</span>
    </div>
  );
}

/** The icon the insert menu and the query block's action share. */
export const ChartIcon = BarChart3;
