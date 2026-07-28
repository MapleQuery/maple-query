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
  return blocks.filter(
    (b): b is StoredNotebookBlockQuery =>
      b.type === "query" && (b.result?.rows.length ?? 0) > 0,
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
