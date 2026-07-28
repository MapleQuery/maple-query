"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Plus,
  FileText,
  MessageSquare,
  Trash2,
  ArrowUpCircle,
  ArrowDownCircle,
  Play,
  Loader2,
  Pencil,
  Check,
  Crosshair,
  Eye,
  EyeOff,
} from "lucide-react";
import {
  notebooks,
  type StoredNotebook,
  type StoredNotebookBlock,
  type StoredNotebookBlockProse,
  type StoredNotebookBlockQuery,
} from "@/lib/storage";
import { streamChat } from "@/lib/sse";
import {
  EMPTY_RESULT_ROWS,
  mergeRowsFrame,
  seedPreviewRows,
} from "@/lib/result-rows";
import type { SuggestionT } from "@/lib/types";
import {
  getCachedDatasetTitles,
  rememberDatasetTitles,
  useDatasetTitles,
} from "@/lib/dataset-titles";
import { uuid } from "@/lib/utils";
import { SqlBlock } from "@/components/evidence/sql-block";
import { RowsTable } from "@/components/evidence/rows-table";
import { DatasetChip } from "@/components/evidence/dataset-chip";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/components/ui/toast";
import { exportNotebookAsMarkdown } from "./export";
import { ExportMenu } from "./export-menu";
import { printMarkdownAsPdf } from "./print-pdf";
import { ScopePicker } from "./scope-picker";
import { ChartBlock, ChartIcon } from "./chart-block";
import {
  chartableSources,
  hiddenBlockCount,
  migrateInlineCharts,
} from "@/lib/notebook";

export interface NotebookContainerProps {
  notebookId: string;
}

export type BlockKind = "prose" | "query" | "chart";

/**
 * A blank block of `kind`, or null when one cannot exist yet.
 *
 * A chart needs something to chart, so inserting one from the menu binds
 * it to the nearest query above the insertion point — the one the author
 * was almost certainly looking at — and falls back to the last query
 * anywhere in the notebook. With no query at all, there is nothing to
 * make, and the menu does not offer it.
 */
function newBlock(
  kind: BlockKind,
  blocks: StoredNotebookBlock[],
  atIndex: number,
): StoredNotebookBlock | null {
  if (kind === "prose") return { type: "prose", id: uuid(), markdown: "" };
  if (kind === "query") {
    return {
      type: "query",
      id: uuid(),
      question: "",
      conversationId: uuid(),
      state: "idle",
    };
  }
  const sources = chartableSources(blocks);
  if (sources.length === 0) return null;
  const above = [...sources]
    .reverse()
    .find((s) => blocks.findIndex((b) => b.id === s.id) < atIndex);
  return {
    type: "chart",
    id: uuid(),
    sourceBlockId: (above ?? sources[sources.length - 1]).id,
  };
}

export function NotebookContainer({ notebookId }: NotebookContainerProps) {
  const router = useRouter();
  const toast = useToast();
  const [nb, setNb] = React.useState<StoredNotebook | null>(null);
  const [index, setIndex] = React.useState(notebooks.list());
  const [titleEditing, setTitleEditing] = React.useState(false);

  React.useEffect(() => {
    const stored = notebooks.load(notebookId);
    setIndex(notebooks.list());
    if (stored) {
      // Charts used to be a field on a query block. Converting on load
      // keeps every saved notebook readable without a stored version.
      const migrated = migrateInlineCharts(stored);
      // Written back when it actually changed something, so the legacy
      // field does not linger and the conversion does not re-run on
      // every open. `notebooks.save` rather than `persist`: opening a
      // notebook must not touch its "last edited" time.
      if (migrated !== stored) notebooks.save(migrated);
      setNb(migrated);
    } else {
      const now = new Date().toISOString();
      setNb({
        id: notebookId,
        title: "Untitled notebook",
        createdAt: now,
        updatedAt: now,
        blocks: [],
      });
    }
  }, [notebookId]);

  const persist = React.useCallback((next: StoredNotebook) => {
    const stamped = { ...next, updatedAt: new Date().toISOString() };
    setNb(stamped);
    notebooks.save(stamped);
    setIndex(notebooks.list());
  }, []);

  const updateBlock = React.useCallback(
    (blockId: string, mutator: (b: StoredNotebookBlock) => StoredNotebookBlock) => {
      setNb((prev) => {
        if (!prev) return prev;
        const next = {
          ...prev,
          blocks: prev.blocks.map((b) => (b.id === blockId ? mutator(b) : b)),
        };
        persist(next);
        return next;
      });
    },
    [persist],
  );

  const addBlock = (kind: BlockKind, atIndex?: number) => {
    if (!nb) return;
    const idx = atIndex ?? nb.blocks.length;
    const block = newBlock(kind, nb.blocks, idx);
    if (!block) return;
    const blocks = [...nb.blocks];
    blocks.splice(idx, 0, block);
    persist({ ...nb, blocks });
  };

  /** Insert a chart of `sourceBlockId` directly below it. */
  const insertChart = (sourceBlockId: string) => {
    if (!nb) return;
    const idx = nb.blocks.findIndex((b) => b.id === sourceBlockId);
    if (idx === -1) return;
    const blocks = [...nb.blocks];
    blocks.splice(idx + 1, 0, {
      type: "chart",
      id: uuid(),
      sourceBlockId,
    });
    persist({ ...nb, blocks });
  };

  const toggleHidden = (blockId: string) => {
    updateBlock(blockId, (b) => ({ ...b, hidden: !b.hidden }));
  };

  // Accepting an offer inserts a *draft*, not a result. Notebooks are
  // authored before they are executed, and a runnable draft keeps that
  // rhythm — the user can reword before spending a turn. The new block
  // gets its own fresh conversationId and empty history like every
  // other block; the only thing it inherits is the package scope.
  const insertFollowUp = (afterBlockId: string, s: SuggestionT) => {
    if (!nb) return;
    const idx = nb.blocks.findIndex((b) => b.id === afterBlockId);
    if (idx === -1) return;
    const block: StoredNotebookBlock = {
      type: "query",
      id: uuid(),
      question: s.question,
      conversationId: uuid(),
      state: "idle",
      scopePackageIds: s.package_ids,
    };
    const blocks = [...nb.blocks];
    blocks.splice(idx + 1, 0, block);
    persist({ ...nb, blocks });
  };

  const removeBlock = (blockId: string) => {
    if (!nb) return;
    persist({ ...nb, blocks: nb.blocks.filter((b) => b.id !== blockId) });
  };

  const moveBlock = (blockId: string, direction: -1 | 1) => {
    if (!nb) return;
    const idx = nb.blocks.findIndex((b) => b.id === blockId);
    const target = idx + direction;
    if (idx === -1 || target < 0 || target >= nb.blocks.length) return;
    const blocks = [...nb.blocks];
    [blocks[idx], blocks[target]] = [blocks[target], blocks[idx]];
    persist({ ...nb, blocks });
  };

  const handleNewNotebook = () => {
    router.push(`/notebook/${uuid()}`);
  };

  const handleDeleteNotebook = (id: string) => {
    notebooks.remove(id);
    setIndex(notebooks.list());
    if (id === notebookId) router.push(`/notebook/${uuid()}`);
  };

  const handleExportMarkdown = () => {
    if (!nb) return;
    const md = exportNotebookAsMarkdown(nb, getCachedDatasetTitles());
    const blob = new Blob([md], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const safe = (nb.title || "notebook").replace(/[^a-z0-9-]+/gi, "-").toLowerCase();
    a.href = url;
    a.download = `${safe}.md`;
    a.click();
    URL.revokeObjectURL(url);
    toast.show("Downloaded Markdown export", "success");
  };

  const handleExportPdf = async () => {
    if (!nb) return;
    const md = exportNotebookAsMarkdown(nb, getCachedDatasetTitles());
    try {
      await printMarkdownAsPdf(nb.title || "Untitled notebook", md);
      toast.show("Print view open — choose “Save as PDF”", "info");
    } catch {
      toast.show("Could not open the PDF view", "error");
    }
  };

  if (!nb) return null;

  const hiddenCount = hiddenBlockCount(nb);
  const exportableCount = nb.blocks.length - hiddenCount;
  const canChart = chartableSources(nb.blocks).length > 0;

  return (
    <div className="flex h-[calc(100vh-4rem)] min-h-0">
      <aside className="hidden w-64 shrink-0 flex-col border-r border-hairline bg-surface-soft/70 lg:flex">
        <div className="border-b border-hairline p-3">
          <button
            type="button"
            onClick={handleNewNotebook}
            className="flex w-full items-center gap-2 rounded-md bg-coral px-3 py-2 text-sm font-medium text-ink hover:bg-coral-active focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy"
          >
            <Plus className="h-4 w-4" /> New notebook
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-2">
          <p className="mb-1 px-2 text-[10px] font-semibold uppercase tracking-wider text-muted">
            Notebooks
          </p>
          {index.length === 0 ? (
            <p className="px-2 py-6 text-center text-xs text-muted">
              No saved notebooks.
            </p>
          ) : (
            <ul className="space-y-1">
              {index.map((entry) => (
                <li key={entry.id} className="group flex items-center gap-1 rounded-md px-2 py-1.5 transition-colors hover:bg-white/60">
                  <Link
                    href={`/notebook/${entry.id}`}
                    className="min-w-0 flex-1"
                  >
                    <span className={`line-clamp-1 text-sm ${entry.id === notebookId ? "font-medium text-ink" : "text-body"}`}>
                      {entry.title}
                    </span>
                    <span className="font-mono text-[10px] text-muted">
                      {new Date(entry.updatedAt).toLocaleDateString()}
                    </span>
                  </Link>
                  <button
                    type="button"
                    onClick={() => handleDeleteNotebook(entry.id)}
                    className="opacity-0 group-hover:opacity-100 p-1 text-muted hover:text-error"
                    aria-label="Delete notebook"
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col overflow-y-auto">
        <div className="mx-auto w-full max-w-4xl flex-1 px-4 py-10 md:px-6">
          <header className="mb-8 flex flex-wrap items-start justify-between gap-3 border-b border-hairline pb-6">
            <div className="min-w-0 flex-1">
              {titleEditing ? (
                <input
                  autoFocus
                  value={nb.title}
                  onChange={(e) => setNb({ ...nb, title: e.target.value })}
                  onBlur={() => {
                    setTitleEditing(false);
                    persist(nb);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      setTitleEditing(false);
                      persist(nb);
                    }
                  }}
                  className="w-full border-b border-hairline bg-transparent font-display text-3xl font-medium tracking-tight text-ink focus:border-navy focus:outline-none md:text-4xl"
                />
              ) : (
                <button
                  type="button"
                  onClick={() => setTitleEditing(true)}
                  className="group flex items-center gap-2 text-left"
                >
                  <h1 className="font-display text-3xl font-medium tracking-tight text-ink md:text-4xl">
                    {nb.title || "Untitled notebook"}
                  </h1>
                  <Pencil className="h-4 w-4 opacity-0 group-hover:opacity-100 text-muted" />
                </button>
              )}
              <p className="mt-2 font-mono text-xs text-muted">
                {nb.blocks.length} block{nb.blocks.length === 1 ? "" : "s"}
                {hiddenCount > 0 && ` · ${hiddenCount} not exported`} · last
                edited {new Date(nb.updatedAt).toLocaleString()}
              </p>
            </div>
            <ExportMenu
              onExportMarkdown={handleExportMarkdown}
              onExportPdf={() => void handleExportPdf()}
              // Every block hidden means an export with nothing in it,
              // which is worth refusing rather than delivering.
              disabled={nb.blocks.length === 0 || exportableCount === 0}
            />
          </header>

          {nb.blocks.length === 0 ? (
            <NotebookEmpty onAdd={(kind) => addBlock(kind)} />
          ) : (
            <div className="space-y-4">
              {nb.blocks.map((b, i) => (
                <React.Fragment key={b.id}>
                  <BlockInsert
                    onAdd={(kind) => addBlock(kind, i)}
                    canChart={canChart}
                  />
                  <NotebookBlock
                    block={b}
                    blocks={nb.blocks}
                    canMoveUp={i > 0}
                    canMoveDown={i < nb.blocks.length - 1}
                    onMove={(d) => moveBlock(b.id, d)}
                    onRemove={() => removeBlock(b.id)}
                    onToggleHidden={() => toggleHidden(b.id)}
                    onUpdate={updateBlock}
                    onInsertFollowUp={insertFollowUp}
                    onInsertChart={insertChart}
                  />
                </React.Fragment>
              ))}
              <BlockInsert
                onAdd={(kind) => addBlock(kind, nb.blocks.length)}
                canChart={canChart}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function NotebookEmpty({ onAdd }: { onAdd: (k: "prose" | "query") => void }) {
  return (
    <div className="rounded-2xl border border-dashed border-hairline bg-white/60 p-10 text-center">
      <FileText className="mx-auto mb-3 h-6 w-6 text-navy" />
      <h2 className="font-display text-xl font-medium text-ink">
        Start with a block
      </h2>
      <p className="mt-1 text-sm text-muted">
        Interleave Markdown prose with runnable questions. Export the finished
        piece as a report.
      </p>
      <div className="mt-4 inline-flex gap-2">
        <button
          type="button"
          onClick={() => onAdd("prose")}
          className="rounded-md border border-hairline bg-white px-3 py-1.5 text-sm font-medium text-ink hover:bg-surface-soft"
        >
          <FileText className="mr-1 inline h-4 w-4" /> Add prose
        </button>
        <button
          type="button"
          onClick={() => onAdd("query")}
          className="rounded-md bg-coral px-3 py-1.5 text-sm font-medium text-ink hover:bg-coral-active"
        >
          <MessageSquare className="mr-1 inline h-4 w-4" /> Add query
        </button>
      </div>
    </div>
  );
}

function BlockInsert({
  onAdd,
  canChart,
}: {
  onAdd: (k: BlockKind) => void;
  canChart: boolean;
}) {
  const [open, setOpen] = React.useState(false);
  return (
    <div className="relative py-1">
      <div className="absolute inset-x-0 top-1/2 h-px bg-hairline/60" />
      <div className="relative flex justify-center">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="inline-flex items-center gap-1 rounded-full border border-hairline bg-white px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-muted transition-colors hover:border-navy hover:text-navy"
        >
          <Plus className="h-3 w-3" /> Insert
        </button>
        {open && (
          <div className="absolute top-8 z-10 flex gap-1 rounded-lg border border-hairline bg-white p-1 shadow-lg">
            <button
              type="button"
              onClick={() => {
                onAdd("prose");
                setOpen(false);
              }}
              className="rounded-md px-3 py-1.5 text-xs font-medium text-ink hover:bg-surface-soft"
            >
              <FileText className="mr-1 inline h-3 w-3" /> Prose
            </button>
            <button
              type="button"
              onClick={() => {
                onAdd("query");
                setOpen(false);
              }}
              className="rounded-md px-3 py-1.5 text-xs font-medium text-ink hover:bg-surface-soft"
            >
              <MessageSquare className="mr-1 inline h-3 w-3" /> Query
            </button>
            {/* Offered only once there is a result to chart — an empty
                chart block would be a puzzle, not a starting point. */}
            {canChart && (
              <button
                type="button"
                onClick={() => {
                  onAdd("chart");
                  setOpen(false);
                }}
                className="rounded-md px-3 py-1.5 text-xs font-medium text-ink hover:bg-surface-soft"
              >
                <ChartIcon className="mr-1 inline h-3 w-3" /> Chart
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function NotebookBlock({
  block,
  blocks,
  canMoveUp,
  canMoveDown,
  onMove,
  onRemove,
  onToggleHidden,
  onUpdate,
  onInsertFollowUp,
  onInsertChart,
}: {
  block: StoredNotebookBlock;
  blocks: StoredNotebookBlock[];
  canMoveUp: boolean;
  canMoveDown: boolean;
  onMove: (d: -1 | 1) => void;
  onRemove: () => void;
  onToggleHidden: () => void;
  onUpdate: (
    id: string,
    mutator: (b: StoredNotebookBlock) => StoredNotebookBlock,
  ) => void;
  onInsertFollowUp: (afterBlockId: string, s: SuggestionT) => void;
  onInsertChart: (sourceBlockId: string) => void;
}) {
  return (
    <div
      className={`group relative rounded-xl border bg-white p-5 shadow-sm ${
        block.hidden
          ? "border-dashed border-hairline"
          : "border-hairline"
      }`}
    >
      <div className="absolute right-3 top-3 flex opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100">
        <BlockActions
          canMoveUp={canMoveUp}
          canMoveDown={canMoveDown}
          hidden={Boolean(block.hidden)}
          onMove={onMove}
          onRemove={onRemove}
          onToggleHidden={onToggleHidden}
        />
      </div>
      {/* Dimmed rather than collapsed: a research block is still being
          worked on, and hiding its contents would make the flag feel
          like a delete. The tag is what carries the meaning. */}
      <div className={block.hidden ? "opacity-45" : undefined}>
        {block.type === "prose" ? (
          <ProseBlock block={block} onUpdate={onUpdate} />
        ) : block.type === "chart" ? (
          <ChartBlock
            block={block}
            blocks={blocks}
            onChangeSource={(sourceBlockId) =>
              onUpdate(block.id, (b) =>
                b.type === "chart" ? { ...b, sourceBlockId } : b,
              )
            }
            onChangeOverrides={(overrides) =>
              onUpdate(block.id, (b) =>
                b.type === "chart" ? { ...b, overrides } : b,
              )
            }
          />
        ) : (
          <QueryBlock
            block={block}
            onUpdate={onUpdate}
            onInsertFollowUp={onInsertFollowUp}
            onInsertChart={onInsertChart}
          />
        )}
      </div>
      {block.hidden && (
        <p className="mt-3 inline-flex items-center gap-1.5 rounded border border-hairline bg-surface-soft px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-muted">
          <EyeOff className="h-3 w-3" /> Research only — not exported
        </p>
      )}
    </div>
  );
}

function BlockActions({
  canMoveUp,
  canMoveDown,
  hidden,
  onMove,
  onRemove,
  onToggleHidden,
}: {
  canMoveUp: boolean;
  canMoveDown: boolean;
  hidden: boolean;
  onMove: (d: -1 | 1) => void;
  onRemove: () => void;
  onToggleHidden: () => void;
}) {
  return (
    <div className="flex items-center gap-1 rounded-md border border-hairline bg-white p-0.5 shadow-sm">
      <button
        type="button"
        onClick={onToggleHidden}
        aria-pressed={hidden}
        title={
          hidden
            ? "Include this block in the export"
            : "Keep this block out of the export"
        }
        aria-label={
          hidden
            ? "Include this block in the export"
            : "Keep this block out of the export"
        }
        className={`rounded p-1 hover:bg-surface-soft ${
          hidden ? "text-navy" : "text-muted hover:text-ink"
        }`}
      >
        {hidden ? (
          <EyeOff className="h-3.5 w-3.5" />
        ) : (
          <Eye className="h-3.5 w-3.5" />
        )}
      </button>
      <button
        type="button"
        disabled={!canMoveUp}
        onClick={() => onMove(-1)}
        className="rounded p-1 text-muted hover:bg-surface-soft hover:text-ink disabled:opacity-30"
        aria-label="Move up"
      >
        <ArrowUpCircle className="h-3.5 w-3.5" />
      </button>
      <button
        type="button"
        disabled={!canMoveDown}
        onClick={() => onMove(1)}
        className="rounded p-1 text-muted hover:bg-surface-soft hover:text-ink disabled:opacity-30"
        aria-label="Move down"
      >
        <ArrowDownCircle className="h-3.5 w-3.5" />
      </button>
      <button
        type="button"
        onClick={onRemove}
        className="rounded p-1 text-muted hover:bg-error/10 hover:text-error"
        aria-label="Remove block"
      >
        <Trash2 className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}

function ProseBlock({
  block,
  onUpdate,
}: {
  block: StoredNotebookBlockProse;
  onUpdate: (
    id: string,
    m: (b: StoredNotebookBlock) => StoredNotebookBlock,
  ) => void;
}) {
  const [editing, setEditing] = React.useState(block.markdown === "");
  return editing ? (
    <div>
      <p className="mb-2 font-mono text-[10px] uppercase tracking-wider text-muted">
        Prose · Markdown
      </p>
      <Textarea
        autoFocus
        value={block.markdown}
        onChange={(e) =>
          onUpdate(block.id, (b) =>
            b.type === "prose" ? { ...b, markdown: e.target.value } : b,
          )
        }
        onBlur={() => setEditing(false)}
        rows={5}
        placeholder="Write in Markdown…"
        className="min-h-[110px] resize-y"
      />
    </div>
  ) : (
    <button
      type="button"
      onClick={() => setEditing(true)}
      className="prose-body block w-full text-left text-body"
    >
      {block.markdown.trim() ? (
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{block.markdown}</ReactMarkdown>
      ) : (
        <span className="text-muted">Empty prose block. Click to edit.</span>
      )}
    </button>
  );
}

function QueryBlock({
  block,
  onUpdate,
  onInsertFollowUp,
  onInsertChart,
}: {
  block: StoredNotebookBlockQuery;
  onUpdate: (
    id: string,
    m: (b: StoredNotebookBlock) => StoredNotebookBlock,
  ) => void;
  onInsertFollowUp: (afterBlockId: string, s: SuggestionT) => void;
  onInsertChart: (sourceBlockId: string) => void;
}) {
  const [draft, setDraft] = React.useState(block.question);
  const [assistantText, setAssistantText] = React.useState("");
  const [scopeOpen, setScopeOpen] = React.useState(false);

  React.useEffect(() => setDraft(block.question), [block.question]);

  const scoped = (block.scopePackageIds?.length ?? 0) > 0;

  // Offered whenever there is a shape to plot. The chart block's own
  // controls handle the rest, including rows the inference declines.
  const rows = block.result?.rows ?? [];
  const chartable =
    rows.length >= 2 && Object.keys(rows[0] ?? {}).length >= 2;

  // An empty scope is stored as absent, not as `[]`, so the run path's
  // "is there a scope" check stays a single truthiness test.
  const setScope = (ids: string[]) =>
    onUpdate(block.id, (b) =>
      b.type === "query"
        ? { ...b, scopePackageIds: ids.length > 0 ? ids : undefined }
        : b,
    );

  // Chips name their dataset rather than print its UUID. A block that
  // has run carries its own titles; a scope pinned from a suggestion,
  // or a notebook saved before titles were recorded, is backfilled.
  const scopeIds = block.scopePackageIds;
  const resultIds = block.result?.packageIds;
  const referencedIds = React.useMemo(
    () => Array.from(new Set([...(scopeIds ?? []), ...(resultIds ?? [])])),
    [scopeIds, resultIds],
  );
  const titles = useDatasetTitles(referencedIds, block.result?.packageTitles);

  const run = async () => {
    const question = draft.trim();
    if (!question) return;
    setAssistantText("");
    let sql = "";
    let result = EMPTY_RESULT_ROWS;
    const pkgIds = new Set<string>();
    const pkgTitles: Record<string, string> = {};
    let localAssistantText = "";
    let offers: SuggestionT[] = [];

    onUpdate(block.id, (b) =>
      b.type === "query"
        ? {
            ...b,
            question,
            state: "running",
            result: undefined,
            suggestions: undefined,
            errorMessage: undefined,
          }
        : b,
    );

    const controller = new AbortController();

    try {
      await streamChat(
        {
          conversation_id: block.conversationId,
          question,
          history: [],
          // Re-sending the pinned scope on every run is what makes a
          // saved block reproducible. Scope is a preference rather than
          // a filter server-side, so a block whose datasets were
          // re-ingested still produces a sensible turn rather than an
          // empty one.
          ...(block.scopePackageIds && block.scopePackageIds.length > 0
            ? { scope_package_ids: block.scopePackageIds }
            : {}),
        },
        {
          onEvent: (event) => {
            switch (event.name) {
              case "datasets_ranked":
                // Titles ride along on this frame, so the chips below
                // never have to ask the API what a package is called.
                rememberDatasetTitles(event.payload.candidates);
                for (const c of event.payload.candidates) {
                  pkgIds.add(c.package_id);
                  const t = c.title?.trim();
                  if (t) pkgTitles[c.package_id] = t;
                }
                break;
              case "sql_guarded":
                if (event.payload.accepted) sql = event.payload.sql_final;
                break;
              case "sql_executed":
                result = seedPreviewRows(event.payload.sample_rows);
                break;
              case "rows":
                result = mergeRowsFrame(result, event.payload);
                break;
              case "message_delta":
                localAssistantText += event.payload.delta;
                setAssistantText(localAssistantText);
                break;
              case "suggestions":
                offers = event.payload.items;
                break;
            }
          },
          onDone: () => {
            onUpdate(block.id, (b) =>
              b.type === "query"
                ? {
                    ...b,
                    state: "done",
                    suggestions: offers,
                    result: {
                      assistantText: localAssistantText,
                      sql,
                      rows: result.rows,
                      packageIds: Array.from(pkgIds),
                      packageTitles: { ...pkgTitles },
                    },
                  }
                : b,
            );
          },
          onError: (err) => {
            onUpdate(block.id, (b) =>
              b.type === "query"
                ? { ...b, state: "error", errorMessage: err.message }
                : b,
            );
          },
        },
        controller.signal,
      );
    } catch (err) {
      onUpdate(block.id, (b) =>
        b.type === "query"
          ? { ...b, state: "error", errorMessage: (err as Error).message }
          : b,
      );
    }
  };

  return (
    <div className="space-y-3">
      <p className="font-mono text-[10px] uppercase tracking-wider text-muted">
        Query · single-turn
      </p>
      {/* A pinned scope the user cannot see or escape would make a saved
          notebook mysteriously narrow months later — and one they cannot
          *set* leaves a question that found the wrong dataset with no way
          to correct it. */}
      <p className="flex flex-wrap items-center gap-1.5 text-xs text-muted">
        {scoped && (
          <>
            <span>Scoped to:</span>
            {block.scopePackageIds?.map((p) => (
              <DatasetChip key={p} packageId={p} title={titles[p]} />
            ))}
          </>
        )}
        <button
          type="button"
          onClick={() => setScopeOpen(true)}
          className="inline-flex items-center gap-1 rounded px-1 text-[11px] text-muted underline hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy"
        >
          <Crosshair className="h-3 w-3" />
          {scoped ? "edit scope" : "scope to datasets"}
        </button>
        {scoped && (
          <button
            type="button"
            onClick={() => setScope([])}
            className="rounded px-1 text-[11px] text-muted underline hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy"
          >
            drop scope
          </button>
        )}
      </p>
      <ScopePicker
        open={scopeOpen}
        onOpenChange={setScopeOpen}
        selected={block.scopePackageIds ?? []}
        onChange={setScope}
      />
      <div className="flex items-end gap-2">
        <Textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          rows={2}
          placeholder="What do you want to ask?"
          className="min-h-[70px] flex-1 resize-y font-sans text-[15px]"
        />
        <button
          type="button"
          onClick={run}
          disabled={block.state === "running" || draft.trim().length === 0}
          className="flex shrink-0 items-center gap-1.5 rounded-md bg-coral px-3 py-2 text-sm font-medium text-ink hover:bg-coral-active disabled:cursor-not-allowed disabled:opacity-40"
        >
          {block.state === "running" ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Play className="h-4 w-4" />
          )}
          {block.state === "done" || block.result ? "Re-run" : "Run"}
        </button>
      </div>

      {block.state === "running" && assistantText && (
        <div className="prose-body rounded-lg border border-hairline bg-surface-soft/60 px-4 py-3 text-[15px] leading-relaxed">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{assistantText}</ReactMarkdown>
        </div>
      )}

      {block.state === "done" && block.result && (
        <div className="space-y-3">
          {block.result.assistantText && (
            <div className="prose-body rounded-lg border border-hairline bg-surface-soft/60 px-4 py-3 text-[15px] leading-relaxed">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {block.result.assistantText}
              </ReactMarkdown>
            </div>
          )}
          {block.result.sql && <SqlBlock sql={block.result.sql} status="accepted" />}
          {block.result.rows.length > 0 && (
            <RowsTable rows={block.result.rows} maxRows={20} />
          )}
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
            {block.result.packageIds.length > 0 && (
              <p className="flex flex-wrap items-center gap-1.5 text-xs text-muted">
                <Check className="h-3 w-3 text-success" />
                Datasets:{" "}
                {block.result.packageIds.map((p) => (
                  <DatasetChip key={p} packageId={p} title={titles[p]} />
                ))}
              </p>
            )}
            {chartable && (
              <button
                type="button"
                onClick={() => onInsertChart(block.id)}
                className="ml-auto inline-flex items-center gap-1.5 rounded-md border border-hairline bg-white px-2.5 py-1 text-xs text-muted transition-colors hover:border-navy hover:text-navy focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy"
              >
                <ChartIcon className="h-3.5 w-3.5" /> Chart this result
              </button>
            )}
          </div>
        </div>
      )}

      {block.state === "done" &&
        block.suggestions &&
        block.suggestions.length > 0 && (
          // Uncapped by design, unlike the chat chips. Each step here
          // costs an insert *and* a run, and the chain stays on the page
          // as a document being built. A cap would also have to count
          // preceding scoped blocks, making the affordance
          // order-dependent — drag a block up and the row reappears.
          <div className="space-y-2 border-t border-hairline pt-3">
            <p className="text-xs font-medium uppercase tracking-wide text-muted">
              Next steps
            </p>
            <ul className="flex flex-wrap gap-2">
              {block.suggestions.slice(0, 3).map((s, i) => (
                <li key={`${s.kind}-${s.package_ids.join(",")}-${i}`}>
                  <button
                    type="button"
                    onClick={() => onInsertFollowUp(block.id, s)}
                    aria-label={`Insert follow-up block: ${s.question}`}
                    className="flex max-w-full items-center gap-1.5 rounded-md border border-hairline bg-white px-3 py-1.5 text-left text-sm text-ink transition-colors hover:bg-surface-soft focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy focus-visible:ring-offset-2 focus-visible:ring-offset-canvas"
                  >
                    <Plus className="h-3.5 w-3.5 shrink-0 text-muted" />
                    {s.label}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}

      {block.state === "error" && (
        <div className="rounded-lg border border-error/30 bg-error/10 px-4 py-3 text-sm text-error">
          {block.errorMessage ?? "Query failed."}
        </div>
      )}
    </div>
  );
}
