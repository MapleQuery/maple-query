/**
 * Notebook-shape helpers: what a notebook exports, where a chart gets
 * its rows, and the one migration between block layouts.
 *
 * Kept apart from `storage.ts`, which is about persistence and knows
 * nothing about what a block means.
 */

import { uuid } from "./utils";
import type {
  StoredNotebook,
  StoredNotebookBlock,
  StoredNotebookBlockQuery,
} from "./storage";

/** Blocks that make it into the exported document. */
export function exportableBlocks(
  nb: StoredNotebook,
): StoredNotebookBlock[] {
  return nb.blocks.filter((b) => !b.hidden);
}

export function hiddenBlockCount(nb: StoredNotebook): number {
  return nb.blocks.filter((b) => b.hidden).length;
}

/** Query blocks that have run and returned rows — what a chart can read. */
export function chartableSources(
  blocks: StoredNotebookBlock[],
): StoredNotebookBlockQuery[] {
  // `rows?.length`, not `rows.length`: these blocks come back out of
  // localStorage, whose shape is versioned by hand, and a result written
  // by an older build (or half-written by an interrupted run) must not
  // take the whole page down on read.
  return blocks.filter(
    (b): b is StoredNotebookBlockQuery =>
      b.type === "query" && (b.result?.rows?.length ?? 0) > 0,
  );
}

/**
 * The rows a chart block draws, or null when its source is gone or has
 * not run. Null is a state the chart block renders, not an error — a
 * dangling reference has to be visible to be repairable.
 */
export function chartSource(
  blocks: StoredNotebookBlock[],
  sourceBlockId: string,
): StoredNotebookBlockQuery | null {
  const source = blocks.find((b) => b.id === sourceBlockId);
  return source && source.type === "query" ? source : null;
}

/**
 * Datasets a new block inserted at `atIndex` could reasonably be scoped
 * to: the ones the nearest finished query block *above* it reached.
 *
 * Nearest-above rather than anywhere in the notebook, because an insert
 * point sits in a line of argument — the relevant datasets are the ones
 * the reader just saw, not every dataset the document has ever touched.
 * Returns nothing when the block above has not run, which is also what
 * keeps the menu quiet on a fresh notebook.
 */
export function quickScopeCandidates(
  blocks: StoredNotebookBlock[],
  atIndex: number,
): { packageId: string; title?: string }[] {
  for (let i = Math.min(atIndex, blocks.length) - 1; i >= 0; i--) {
    const b = blocks[i];
    if (b.type !== "query") continue;
    const ids = b.result?.packageIds ?? [];
    if (ids.length === 0) return [];
    return ids.map((packageId) => ({
      packageId,
      title: b.result?.packageTitles?.[packageId],
    }));
  }
  return [];
}

/**
 * Convert the older inline chart — a `chart` field on a query block —
 * into a chart block sitting just after it.
 *
 * Charts began as a property of a query block, which made them the one
 * thing in a notebook that could not be moved, removed, or placed
 * against a different question. Promoting them to blocks costs this one
 * read-time conversion; it runs on load, is idempotent (it clears the
 * field it reads), and leaves a notebook that never had an inline chart
 * completely untouched, object identity included.
 */
export function migrateInlineCharts(nb: StoredNotebook): StoredNotebook {
  if (!nb.blocks.some((b) => b.type === "query" && b.chart)) return nb;

  const blocks: StoredNotebookBlock[] = [];
  for (const block of nb.blocks) {
    if (block.type !== "query" || !block.chart) {
      blocks.push(block);
      continue;
    }
    const { chart, ...rest } = block;
    blocks.push(rest);
    // The old shape's `hidden` meant "chart dismissed". A dismissed
    // chart becomes no chart block at all rather than a hidden one,
    // because hiding now means something else entirely.
    if (chart.hidden) continue;
    blocks.push({
      type: "chart",
      id: uuid(),
      sourceBlockId: block.id,
      ...(chart.type || chart.categoryColumn || chart.valueColumn
        ? {
            overrides: {
              ...(chart.type ? { type: chart.type } : {}),
              ...(chart.categoryColumn
                ? { categoryColumn: chart.categoryColumn }
                : {}),
              ...(chart.valueColumn
                ? { valueColumn: chart.valueColumn }
                : {}),
            },
          }
        : {}),
    });
  }
  return { ...nb, blocks };
}
