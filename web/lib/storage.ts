/**
 * localStorage helpers with:
 *  - Namespaced keys (v1 schema)
 *  - Per-collection LRU indexes (updatedAt-based)
 *  - Quota-exceeded recovery via oldest-first eviction
 *
 * All reads are SSR-safe (return sensible defaults when window is undefined).
 */

const NS = "mq";
const V = "v1";
const MAX_ENTRIES = 50;

export interface IndexEntry {
  id: string;
  title: string;
  updatedAt: string;
}

type Collection = "conversations" | "notebooks" | "explorer";

function isBrowser(): boolean {
  return typeof window !== "undefined" && "localStorage" in window;
}

function indexKey(c: Collection): string {
  return `${NS}:${c}:${V}:index`;
}

function entryKey(c: Collection, id: string): string {
  return `${NS}:${c}:${V}:${id}`;
}

export function getIndex(c: Collection): IndexEntry[] {
  if (!isBrowser()) return [];
  try {
    const raw = window.localStorage.getItem(indexKey(c));
    if (!raw) return [];
    const parsed = JSON.parse(raw) as IndexEntry[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function writeIndex(c: Collection, entries: IndexEntry[]): void {
  if (!isBrowser()) return;
  window.localStorage.setItem(indexKey(c), JSON.stringify(entries));
}

export function loadEntry<T>(c: Collection, id: string): T | null {
  if (!isBrowser()) return null;
  try {
    const raw = window.localStorage.getItem(entryKey(c, id));
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    return null;
  }
}

/**
 * Persist an entry and refresh the index. If the write pushes quota,
 * evict oldest entries and retry until it fits or the index is empty.
 */
export function saveEntry<T extends { id: string; title: string; updatedAt: string }>(
  c: Collection,
  entry: T,
): void {
  if (!isBrowser()) return;
  const payload = JSON.stringify(entry);
  const stored = attemptWrite(c, entry.id, payload);
  if (!stored) return;

  let idx = getIndex(c).filter((e) => e.id !== entry.id);
  idx.unshift({ id: entry.id, title: entry.title, updatedAt: entry.updatedAt });
  idx = enforceCap(c, idx);
  writeIndex(c, idx);
}

function attemptWrite(c: Collection, id: string, payload: string): boolean {
  const key = entryKey(c, id);
  while (true) {
    try {
      window.localStorage.setItem(key, payload);
      return true;
    } catch (err) {
      if (!isQuotaError(err)) throw err;
      const idx = getIndex(c);
      if (idx.length === 0) return false;
      const oldest = idx[idx.length - 1];
      window.localStorage.removeItem(entryKey(c, oldest.id));
      writeIndex(c, idx.slice(0, -1));
    }
  }
}

function enforceCap(c: Collection, idx: IndexEntry[]): IndexEntry[] {
  while (idx.length > MAX_ENTRIES) {
    const dropped = idx.pop();
    if (dropped) {
      window.localStorage.removeItem(entryKey(c, dropped.id));
    }
  }
  return idx;
}

export function deleteEntry(c: Collection, id: string): void {
  if (!isBrowser()) return;
  window.localStorage.removeItem(entryKey(c, id));
  writeIndex(
    c,
    getIndex(c).filter((e) => e.id !== id),
  );
}

function isQuotaError(err: unknown): boolean {
  return (
    err instanceof DOMException &&
    (err.name === "QuotaExceededError" ||
      err.name === "NS_ERROR_DOM_QUOTA_REACHED" ||
      err.code === 22)
  );
}

// ---------------------------------------------------------------------------
// Typed façades per collection
// ---------------------------------------------------------------------------

import type { ChartOverrides } from "./chart";
import type { HistoryMessage, SuggestionT } from "./types";

export interface EvidenceCard {
  id: string;
  kind: string;
  payload: unknown;
  createdAt: string;
}

export interface StoredConversation {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  history: HistoryMessage[];
  evidenceByTurnId: Record<string, EvidenceCard[]>;
  /** Server-built turn records, echoed back as `turn_records` on the
   * next request. Optional: pre-record conversations load unchanged. */
  turnRecords?: Record<string, unknown>[];
}

export const conversations = {
  list: () => getIndex("conversations"),
  load: (id: string) => loadEntry<StoredConversation>("conversations", id),
  save: (c: StoredConversation) => saveEntry("conversations", c),
  remove: (id: string) => deleteEntry("conversations", id),
};

/**
 * Fields every block carries, whatever its type.
 *
 * A notebook is two things at once: the workspace a question was worked
 * out in, and the document that comes out the other end. `hidden` is the
 * seam between them — the block stays in the editor, editable and
 * runnable, and is left out of the export. Research that led somewhere
 * should not have to be deleted to keep it out of the finished piece.
 */
interface StoredNotebookBlockBase {
  id: string;
  /** Kept out of the exported document. Optional: absent means exported,
   * so notebooks written before this existed export unchanged. */
  hidden?: boolean;
}

export interface StoredNotebookBlockProse extends StoredNotebookBlockBase {
  type: "prose";
  markdown: string;
}
export interface StoredNotebookBlockQuery extends StoredNotebookBlockBase {
  type: "query";
  question: string;
  conversationId: string;
  state: "idle" | "running" | "done" | "error";
  /** Datasets this block is scoped to, pinned at insert time.
   *
   * Pinned to the *block* rather than chained through a parent's
   * conversation history, because blocks are reorderable and deletable
   * and a parent→child link breaks silently the moment one is dragged
   * or removed. A block that carries its own scope stays independently
   * re-runnable, which is the property that makes a notebook a notebook
   * rather than a transcript.
   *
   * Optional: pre-scope notebooks load unchanged. */
  scopePackageIds?: string[];
  /** Offers from the last run; drives the follow-up affordance.
   * Optional for the same reason. */
  suggestions?: SuggestionT[];
  /** Superseded by a chart *block*. Read once on load and converted;
   * see `migrateInlineCharts`. Never written. `hidden` here meant "chart
   * dismissed", which is not what `hidden` means on a block. */
  chart?: ChartOverrides & { hidden?: boolean };
  result?: {
    assistantText: string;
    sql: string;
    rows: Record<string, unknown>[];
    packageIds: string[];
    /** package_id → title, captured from the run that produced this
     * result. Held with the block so a saved notebook still reads as
     * names after the shared title cache is cleared. Optional: blocks
     * saved before it existed fall back to the id. */
    packageTitles?: Record<string, string>;
  };
  errorMessage?: string;
}
/**
 * A chart of some query block's result.
 *
 * It holds a *reference*, not a copy of the rows. A snapshot would be
 * reorder- and delete-proof, but it would also go stale the moment its
 * query is re-run — a chart showing last run's numbers directly above a
 * table showing this run's is wrong in the way that is hardest to catch,
 * because nothing on screen looks broken. Reading the source's rows live
 * means the chart is always consistent with the table it sits near, and
 * the one failure a reference has (its source deleted) is loud and
 * repairable: the block says so and offers the other queries.
 */
export interface StoredNotebookBlockChart extends StoredNotebookBlockBase {
  type: "chart";
  /** The query block whose result this charts. */
  sourceBlockId: string;
  /** Overrides only, never a resolved spec: absent means "whatever the
   * rows infer", so a re-run against different columns re-infers rather
   * than pointing at a column that no longer exists. */
  overrides?: ChartOverrides;
}

export type StoredNotebookBlock =
  | StoredNotebookBlockProse
  | StoredNotebookBlockQuery
  | StoredNotebookBlockChart;

export interface StoredNotebook {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  blocks: StoredNotebookBlock[];
}

export const notebooks = {
  list: () => getIndex("notebooks"),
  load: (id: string) => loadEntry<StoredNotebook>("notebooks", id),
  save: (n: StoredNotebook) => saveEntry("notebooks", n),
  remove: (id: string) => deleteEntry("notebooks", id),
};

export interface ExplorerStepPrompt {
  type: "prompt";
  id: string;
  text: string;
  producedStepId?: string;
}
export interface ExplorerStepSql {
  type: "sql";
  id: string;
  sql: string;
  rows: Record<string, unknown>[];
  rowCount: number;
  status: "ok" | "guard_rejected" | "execution_error" | "budget_exceeded" | "column_not_in_doc";
  reason?: string | null;
  sourceStepId?: string;
  bytesBilled?: number | null;
  elapsedMs?: number | null;
}
export type ExplorerStep = ExplorerStepPrompt | ExplorerStepSql;

export interface StoredExplorer {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  steps: ExplorerStep[];
  activeStepId: string | null;
}

export const explorer = {
  list: () => getIndex("explorer"),
  load: (id: string) => loadEntry<StoredExplorer>("explorer", id),
  save: (e: StoredExplorer) => saveEntry("explorer", e),
  remove: (id: string) => deleteEntry("explorer", id),
};
