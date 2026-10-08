/**
 * Moving work from chat into a notebook.
 *
 * A chat answer already holds everything a finished notebook block
 * needs: the question, the answer text, the SQL that ran, its rows, and
 * the sources it cited. Carrying that across as a *completed* block,
 * rather than a draft to re-run, is what makes "add to notebook" free:
 * no second turn, no second bill, and the numbers in the notebook are
 * exactly the numbers the user just read.
 */

import { inferChartSpec } from "./chart";
import {
  notebooks,
  type EvidenceCard,
  type StoredConversation,
  type StoredNotebook,
  type StoredNotebookBlock,
  type StoredNotebookBlockQuery,
  type StoredSource,
} from "./storage";
import { truncate, uuid } from "./utils";

export interface TurnSnapshot {
  question: string;
  assistantText: string;
  cards: EvidenceCard[];
}

const DATASET_LINK_RE = /\/datasets\/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/gi;

/** Package ids the answer actually cites, in citation order. */
export function citedPackageIds(text: string): string[] {
  const seen = new Set<string>();
  for (const m of text.matchAll(DATASET_LINK_RE)) seen.add(m[1].toLowerCase());
  return Array.from(seen);
}

/**
 * Long StatCan rows (`series, period, value, …`) as one row per period
 * with a column per series — the shape a chart and a reader both want.
 * Series labels lose the segments every series shares ("Canada; " on
 * each of five product groups) so the columns read as what differs.
 */
export function pivotSourceRows(
  rows: Record<string, unknown>[],
): Record<string, unknown>[] {
  if (rows.length === 0) return rows;
  const labels = Array.from(new Set(rows.map((r) => String(r.series ?? ""))));
  const short = shortenLabels(labels);
  const byPeriod = new Map<string, Record<string, unknown>>();
  for (const r of rows) {
    const period = String(r.period ?? "");
    const row = byPeriod.get(period) ?? { period };
    row[short.get(String(r.series ?? "")) ?? "value"] = r.value;
    byPeriod.set(period, row);
  }
  return Array.from(byPeriod.values());
}

function shortenLabels(labels: string[]): Map<string, string> {
  const out = new Map<string, string>();
  if (labels.length === 1) {
    out.set(labels[0], "value");
    return out;
  }
  const parts = labels.map((l) => l.split("; "));
  const width = Math.min(...parts.map((p) => p.length));
  const shared = new Set<number>();
  for (let i = 0; i < width; i++) {
    if (parts.every((p) => p[i] === parts[0][i])) shared.add(i);
  }
  labels.forEach((label, j) => {
    const kept = parts[j].filter((_, i) => !shared.has(i));
    out.set(label, kept.join("; ") || label);
  });
  return out;
}

interface CardPayload {
  kind?: string;
  candidates?: { package_id?: string; title?: string | null }[];
  guard?: { accepted?: boolean; sql_final?: string };
  sql?: string;
  executed?: { rows?: Record<string, unknown>[] };
  tableId?: string;
  source?: string;
  title?: string;
  url?: string;
  rows?: Record<string, unknown>[];
}

/** The last result the turn produced, and every source it read. */
export function resultFromCards(
  assistantText: string,
  cards: EvidenceCard[],
): NonNullable<StoredNotebookBlockQuery["result"]> {
  let sql = "";
  let rows: Record<string, unknown>[] = [];
  const titles: Record<string, string> = {};
  const sources: StoredSource[] = [];
  for (const card of cards) {
    const p = card.payload as CardPayload;
    if (p.kind === "datasets_ranked") {
      for (const c of p.candidates ?? []) {
        const t = c.title?.trim();
        if (c.package_id && t) titles[c.package_id] = t;
      }
    } else if (p.kind === "sql_generated" && p.guard?.accepted) {
      sql = p.guard.sql_final ?? p.sql ?? "";
      if (p.executed?.rows?.length) rows = p.executed.rows;
    } else if (p.kind === "source_data" && p.tableId && p.url) {
      if (!sources.some((s) => s.tableId === p.tableId)) {
        sources.push({
          tableId: p.tableId,
          title: p.title ?? p.tableId,
          url: p.url,
          source: p.source,
        });
      }
      // Whichever result arrived last is the block's table: the model
      // fetches what it answers from after it has looked around.
      if (p.rows?.length) {
        rows = p.source === "statcan" ? pivotSourceRows(p.rows) : p.rows;
      }
    }
  }
  const packageIds = citedPackageIds(assistantText);
  const packageTitles: Record<string, string> = {};
  for (const id of packageIds) if (titles[id]) packageTitles[id] = titles[id];
  return { assistantText, sql, rows, packageIds, packageTitles, sources };
}

export function blockFromTurn(turn: TurnSnapshot): StoredNotebookBlockQuery {
  return {
    type: "query",
    id: uuid(),
    question: turn.question,
    conversationId: uuid(),
    state: "done",
    result: resultFromCards(turn.assistantText, turn.cards),
  };
}

/** The query block, plus a chart of it when its rows have a shape. */
export function blocksFromTurn(turn: TurnSnapshot): StoredNotebookBlock[] {
  const block = blockFromTurn(turn);
  const rows = block.result?.rows ?? [];
  const out: StoredNotebookBlock[] = [block];
  if (rows.length >= 3 && inferChartSpec(rows)) {
    out.push({ type: "chart", id: uuid(), sourceBlockId: block.id });
  }
  return out;
}

/**
 * Append turns to a notebook — an existing one, or a new one titled
 * after the first question. Returns the notebook written.
 */
export function addTurnsToNotebook(
  turns: TurnSnapshot[],
  target: { notebookId: string } | { newTitle?: string },
  from?: { conversationId: string; title: string },
): StoredNotebook {
  const now = new Date().toISOString();
  const existing = "notebookId" in target ? notebooks.load(target.notebookId) : null;
  const nb: StoredNotebook = existing ?? {
    id: "notebookId" in target ? target.notebookId : uuid(),
    title:
      ("newTitle" in target && target.newTitle) ||
      truncate(turns[0]?.question ?? "Untitled notebook", 60),
    createdAt: now,
    updatedAt: now,
    blocks: [],
  };
  const next: StoredNotebook = {
    ...nb,
    // The first conversation a notebook was built from stays its origin.
    source: nb.source ?? from,
    updatedAt: now,
    blocks: [...nb.blocks, ...turns.flatMap(blocksFromTurn)],
  };
  notebooks.save(next);
  return next;
}

/**
 * A saved conversation's turns, paired with their evidence. History
 * holds the text; evidence is keyed by turn id in insertion order, which
 * is also turn order.
 */
export function turnsFromConversation(
  c: StoredConversation,
): (TurnSnapshot & { id: string })[] {
  const out: (TurnSnapshot & { id: string })[] = [];
  const evidence = c.evidenceByTurnId ?? {};
  const turnIds = Object.keys(evidence);
  let turnIdx = 0;
  const h = c.history;
  let i = 0;
  while (i < h.length) {
    const msg = h[i];
    if (msg.role !== "user") {
      i += 1;
      continue;
    }
    const nextAssistant = h.findIndex((m, j) => j > i && m.role === "assistant");
    const reply =
      nextAssistant >= 0
        ? ((h[nextAssistant] as { content: string | null }).content ?? "")
        : "";
    const id = turnIds[turnIdx++] ?? uuid();
    out.push({ id, question: msg.content, assistantText: reply, cards: evidence[id] ?? [] });
    i = nextAssistant >= 0 ? nextAssistant + 1 : i + 1;
  }
  return out;
}

/** "Statistics Canada, <title> (table 18-10-0004-01)" or "open.canada.ca, <title>". */
export function sourceCitation(s: StoredSource): string {
  if (s.source === "open.canada.ca") return `open.canada.ca, ${s.title}`;
  if (s.source === "parliament") return `openparliament.ca, ${s.title}`;
  return `Statistics Canada, ${s.title} (table ${s.tableId})`;
}
