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
  Download,
  Play,
  Loader2,
  Pencil,
  Check,
} from "lucide-react";
import {
  notebooks,
  type StoredNotebook,
  type StoredNotebookBlock,
  type StoredNotebookBlockProse,
  type StoredNotebookBlockQuery,
} from "@/lib/storage";
import { streamChat } from "@/lib/sse";
import { truncate, uuid } from "@/lib/utils";
import { SqlBlock } from "@/components/evidence/sql-block";
import { RowsTable } from "@/components/evidence/rows-table";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/components/ui/toast";
import { exportNotebookAsMarkdown } from "./export";

export interface NotebookContainerProps {
  notebookId: string;
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
      setNb(stored);
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

  const addBlock = (kind: "prose" | "query", atIndex?: number) => {
    if (!nb) return;
    const block: StoredNotebookBlock =
      kind === "prose"
        ? { type: "prose", id: uuid(), markdown: "" }
        : {
            type: "query",
            id: uuid(),
            question: "",
            conversationId: uuid(),
            state: "idle",
          };
    const idx = atIndex ?? nb.blocks.length;
    const blocks = [...nb.blocks];
    blocks.splice(idx, 0, block);
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

  const handleExport = () => {
    if (!nb) return;
    const md = exportNotebookAsMarkdown(nb);
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

  if (!nb) return null;

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
                {nb.blocks.length} block{nb.blocks.length === 1 ? "" : "s"} · last
                edited {new Date(nb.updatedAt).toLocaleString()}
              </p>
            </div>
            <button
              type="button"
              onClick={handleExport}
              className="inline-flex items-center gap-1.5 rounded-md border border-hairline bg-white px-3 py-2 text-sm font-medium text-ink hover:bg-surface-soft"
            >
              <Download className="h-4 w-4" /> Export as Markdown
            </button>
          </header>

          {nb.blocks.length === 0 ? (
            <NotebookEmpty onAdd={(kind) => addBlock(kind)} />
          ) : (
            <div className="space-y-4">
              {nb.blocks.map((b, i) => (
                <React.Fragment key={b.id}>
                  <BlockInsert onAdd={(kind) => addBlock(kind, i)} />
                  <NotebookBlock
                    block={b}
                    canMoveUp={i > 0}
                    canMoveDown={i < nb.blocks.length - 1}
                    onMove={(d) => moveBlock(b.id, d)}
                    onRemove={() => removeBlock(b.id)}
                    onUpdate={updateBlock}
                  />
                </React.Fragment>
              ))}
              <BlockInsert onAdd={(kind) => addBlock(kind, nb.blocks.length)} />
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

function BlockInsert({ onAdd }: { onAdd: (k: "prose" | "query") => void }) {
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
          </div>
        )}
      </div>
    </div>
  );
}

function NotebookBlock({
  block,
  canMoveUp,
  canMoveDown,
  onMove,
  onRemove,
  onUpdate,
}: {
  block: StoredNotebookBlock;
  canMoveUp: boolean;
  canMoveDown: boolean;
  onMove: (d: -1 | 1) => void;
  onRemove: () => void;
  onUpdate: (
    id: string,
    mutator: (b: StoredNotebookBlock) => StoredNotebookBlock,
  ) => void;
}) {
  return (
    <div className="group relative rounded-xl border border-hairline bg-white p-5 shadow-sm">
      <div className="absolute right-3 top-3 flex opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
        <BlockActions
          canMoveUp={canMoveUp}
          canMoveDown={canMoveDown}
          onMove={onMove}
          onRemove={onRemove}
        />
      </div>
      {block.type === "prose" ? (
        <ProseBlock block={block} onUpdate={onUpdate} />
      ) : (
        <QueryBlock block={block} onUpdate={onUpdate} />
      )}
    </div>
  );
}

function BlockActions({
  canMoveUp,
  canMoveDown,
  onMove,
  onRemove,
}: {
  canMoveUp: boolean;
  canMoveDown: boolean;
  onMove: (d: -1 | 1) => void;
  onRemove: () => void;
}) {
  return (
    <div className="flex items-center gap-1 rounded-md border border-hairline bg-white p-0.5 shadow-sm">
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
}: {
  block: StoredNotebookBlockQuery;
  onUpdate: (
    id: string,
    m: (b: StoredNotebookBlock) => StoredNotebookBlock,
  ) => void;
}) {
  const [draft, setDraft] = React.useState(block.question);
  const [assistantText, setAssistantText] = React.useState("");

  React.useEffect(() => setDraft(block.question), [block.question]);

  const run = async () => {
    const question = draft.trim();
    if (!question) return;
    setAssistantText("");
    let sql = "";
    let rows: Record<string, unknown>[] = [];
    const pkgIds = new Set<string>();
    let localAssistantText = "";

    onUpdate(block.id, (b) =>
      b.type === "query"
        ? { ...b, question, state: "running", result: undefined, errorMessage: undefined }
        : b,
    );

    const controller = new AbortController();

    try {
      await streamChat(
        {
          conversation_id: block.conversationId,
          question,
          history: [],
        },
        {
          onEvent: (event) => {
            switch (event.name) {
              case "datasets_ranked":
                for (const c of event.payload.candidates) pkgIds.add(c.package_id);
                break;
              case "sql_guarded":
                if (event.payload.accepted) sql = event.payload.sql_final;
                break;
              case "sql_executed":
                rows = event.payload.sample_rows ?? [];
                break;
              case "rows":
                rows = [...rows, ...event.payload.rows];
                break;
              case "message_delta":
                localAssistantText += event.payload.delta;
                setAssistantText(localAssistantText);
                break;
            }
          },
          onDone: () => {
            onUpdate(block.id, (b) =>
              b.type === "query"
                ? {
                    ...b,
                    state: "done",
                    result: {
                      assistantText: localAssistantText,
                      sql,
                      rows,
                      packageIds: Array.from(pkgIds),
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
          {block.result.packageIds.length > 0 && (
            <p className="flex flex-wrap items-center gap-1.5 text-xs text-muted">
              <Check className="h-3 w-3 text-success" />
              Datasets:{" "}
              {block.result.packageIds.map((p) => (
                <Link
                  key={p}
                  href={`/datasets/${p}`}
                  className="rounded bg-surface-soft px-1.5 py-0.5 font-mono text-[10px] text-navy hover:underline"
                >
                  {truncate(p, 24)}
                </Link>
              ))}
            </p>
          )}
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
