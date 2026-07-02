"use client";

import * as React from "react";
import {
  ArrowUp,
  Loader2,
  Pencil,
  Play,
  RefreshCw,
  Sparkles,
} from "lucide-react";
import { streamChat } from "@/lib/sse";
import { runSql } from "@/lib/api";
import {
  explorer as explorerStore,
  type ExplorerStep,
  type StoredExplorer,
} from "@/lib/storage";
import { SqlBlock } from "@/components/evidence/sql-block";
import { RowsTable } from "@/components/evidence/rows-table";
import { Textarea } from "@/components/ui/textarea";
import { cn, truncate, uuid } from "@/lib/utils";
import { useToast } from "@/components/ui/toast";

const SESSION_KEY = "explorer:current-v1";

export function ExplorerContainer() {
  const toast = useToast();
  const [session, setSession] = React.useState<StoredExplorer | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [prompt, setPrompt] = React.useState("");

  React.useEffect(() => {
    const id = typeof window !== "undefined"
      ? window.localStorage.getItem(SESSION_KEY)
      : null;
    if (id) {
      const stored = explorerStore.load(id);
      if (stored) {
        setSession(stored);
        return;
      }
    }
    createSession();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const createSession = React.useCallback(() => {
    const now = new Date().toISOString();
    const fresh: StoredExplorer = {
      id: uuid(),
      title: "Explorer session",
      createdAt: now,
      updatedAt: now,
      steps: [],
      activeStepId: null,
    };
    setSession(fresh);
    explorerStore.save(fresh);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(SESSION_KEY, fresh.id);
    }
  }, []);

  const persist = React.useCallback((next: StoredExplorer) => {
    const stamped = { ...next, updatedAt: new Date().toISOString() };
    setSession(stamped);
    explorerStore.save(stamped);
  }, []);

  const activeStep = React.useMemo(() => {
    if (!session) return null;
    return session.steps.find((s) => s.id === session.activeStepId) ?? null;
  }, [session]);

  const handleAsk = async () => {
    if (!session) return;
    const q = prompt.trim();
    if (!q) return;
    setPrompt("");
    setBusy(true);

    const promptStep: Extract<ExplorerStep, { type: "prompt" }> = {
      type: "prompt",
      id: uuid(),
      text: q,
    };
    const sqlStepId = uuid();
    const sqlStep: Extract<ExplorerStep, { type: "sql" }> = {
      type: "sql",
      id: sqlStepId,
      sql: "",
      rows: [],
      rowCount: 0,
      status: "ok",
      sourceStepId: promptStep.id,
    };

    let workingSession: StoredExplorer = {
      ...session,
      steps: [...session.steps, promptStep, sqlStep],
      activeStepId: sqlStepId,
    };
    persist(workingSession);

    const controller = new AbortController();

    try {
      let sql = "";
      let rows: Record<string, unknown>[] = [];
      let status: Extract<ExplorerStep, { type: "sql" }>["status"] = "ok";
      let reason: string | null = null;
      let elapsedMs: number | null = null;
      let bytesBilled: number | null = null;

      await streamChat(
        {
          conversation_id: uuid(),
          question: q,
          history: [],
        },
        {
          onEvent: (event) => {
            switch (event.name) {
              case "sql_guarded":
                sql = event.payload.sql_final;
                if (!event.payload.accepted) {
                  status = "guard_rejected";
                  reason = event.payload.reason;
                }
                break;
              case "sql_executed":
                rows = event.payload.sample_rows ?? [];
                elapsedMs = event.payload.elapsed_ms;
                bytesBilled = event.payload.bytes_billed;
                break;
              case "rows":
                rows = [...rows, ...event.payload.rows];
                break;
              case "tool_error":
                status = "execution_error";
                reason = event.payload.message;
                break;
              case "budget_exceeded":
                status = "budget_exceeded";
                reason = `Cap ${event.payload.cap} reached for ${event.payload.which}.`;
                break;
            }
          },
          onDone: () => {
            workingSession = {
              ...workingSession,
              steps: workingSession.steps.map((s) =>
                s.id === sqlStepId && s.type === "sql"
                  ? {
                      ...s,
                      sql,
                      rows,
                      rowCount: rows.length,
                      status,
                      reason,
                      elapsedMs,
                      bytesBilled,
                    }
                  : s,
              ),
            };
            persist(workingSession);
          },
          onError: (err) => {
            workingSession = {
              ...workingSession,
              steps: workingSession.steps.map((s) =>
                s.id === sqlStepId && s.type === "sql"
                  ? {
                      ...s,
                      sql,
                      rows,
                      rowCount: rows.length,
                      status: "execution_error",
                      reason: err.message,
                    }
                  : s,
              ),
            };
            persist(workingSession);
          },
        },
        controller.signal,
      );
    } catch (err) {
      toast.show((err as Error).message ?? "Stream failed", "error");
    } finally {
      setBusy(false);
    }
  };

  const handleEditRun = async (stepId: string, newSql: string) => {
    if (!session) return;
    setBusy(true);
    try {
      const res = await runSql(newSql);
      persist({
        ...session,
        steps: session.steps.map((s) =>
          s.id === stepId && s.type === "sql"
            ? {
                ...s,
                sql: res.sql_final || newSql,
                rows: res.rows,
                rowCount: res.row_count,
                status: res.status,
                reason: res.reason,
                elapsedMs: res.elapsed_ms,
                bytesBilled: res.bytes_billed,
              }
            : s,
        ),
        activeStepId: stepId,
      });
    } catch (err) {
      toast.show((err as Error).message ?? "SQL run failed", "error");
    } finally {
      setBusy(false);
    }
  };

  const setActive = (stepId: string) => {
    if (!session) return;
    persist({ ...session, activeStepId: stepId });
  };

  if (!session) return null;

  return (
    <div className="flex h-[calc(100vh-4rem)] min-h-0">
      <section
        aria-label="Prompt + step chain"
        className="flex w-[420px] shrink-0 flex-col border-r border-hairline bg-surface-soft/40"
      >
        <div className="border-b border-hairline p-4">
          <h2 className="font-display text-lg font-medium text-ink">
            Step chain
          </h2>
          <p className="text-xs text-muted">
            Ask a question, review the SQL, then edit it directly. Every run
            goes through the guard.
          </p>
        </div>

        <div className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
          {session.steps.length === 0 ? (
            <div className="rounded-xl border border-dashed border-hairline bg-white/60 p-6 text-center">
              <Sparkles className="mx-auto mb-2 h-4 w-4 text-navy" />
              <p className="text-sm text-ink">
                Ask a question to seed the chain.
              </p>
            </div>
          ) : (
            <ol className="space-y-3">
              {session.steps.map((step, i) => (
                <li key={step.id}>
                  {step.type === "prompt" ? (
                    <PromptCard
                      index={i}
                      text={step.text}
                      isActive={session.activeStepId === step.id}
                      onSelect={() => setActive(step.id)}
                    />
                  ) : (
                    <SqlStepCard
                      index={i}
                      step={step}
                      isActive={session.activeStepId === step.id}
                      onSelect={() => setActive(step.id)}
                      onRun={(sql) => handleEditRun(step.id, sql)}
                      running={busy && session.activeStepId === step.id}
                    />
                  )}
                </li>
              ))}
            </ol>
          )}
        </div>

        <div className="border-t border-hairline p-3">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleAsk();
            }}
            className="flex items-end gap-2 rounded-xl border border-hairline bg-white p-2 shadow-sm focus-within:ring-2 focus-within:ring-navy"
          >
            <Textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleAsk();
                }
              }}
              rows={1}
              placeholder="Ask a question. MapleQuery will pick datasets and generate SQL."
              className="max-h-40 min-h-[42px] flex-1 resize-none border-0 shadow-none focus-visible:ring-0"
              disabled={busy}
            />
            <button
              type="submit"
              disabled={busy || !prompt.trim()}
              className="rounded-md bg-coral p-2 text-ink hover:bg-coral-active disabled:cursor-not-allowed disabled:opacity-40"
              aria-label="Send prompt"
            >
              {busy ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <ArrowUp className="h-4 w-4" />
              )}
            </button>
          </form>
          <button
            type="button"
            onClick={createSession}
            className="mt-2 inline-flex items-center gap-1 text-xs text-muted hover:text-ink"
          >
            <RefreshCw className="h-3 w-3" /> Reset session
          </button>
        </div>
      </section>

      <section
        aria-label="Active step result"
        className="flex min-w-0 flex-1 flex-col overflow-y-auto p-6"
      >
        {activeStep && activeStep.type === "sql" ? (
          <ActiveStepView
            step={activeStep}
            onRun={(sql) => handleEditRun(activeStep.id, sql)}
            busy={busy}
          />
        ) : (
          <div className="grid flex-1 place-items-center">
            <div className="max-w-md rounded-2xl border border-dashed border-hairline bg-white/50 p-8 text-center">
              <Sparkles className="mx-auto mb-2 h-5 w-5 text-navy" />
              <p className="text-sm text-ink">
                Ask a question or pick a step to see its rows here.
              </p>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

function PromptCard({
  index,
  text,
  isActive,
  onSelect,
}: {
  index: number;
  text: string;
  isActive: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        "block w-full rounded-xl border p-3 text-left transition-all",
        isActive
          ? "border-navy bg-white shadow-md"
          : "border-hairline bg-white/70 hover:border-navy/60 hover:bg-white",
      )}
    >
      <div className="mb-1 flex items-center gap-2">
        <span className="font-mono text-[10px] text-muted">
          {String(index + 1).padStart(2, "0")}
        </span>
        <span className="rounded-full bg-coral/15 px-2 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-wider text-navy">
          Prompt
        </span>
      </div>
      <p className="line-clamp-3 text-sm text-body">{text}</p>
    </button>
  );
}

function SqlStepCard({
  index,
  step,
  isActive,
  onSelect,
  onRun,
  running,
}: {
  index: number;
  step: Extract<ExplorerStep, { type: "sql" }>;
  isActive: boolean;
  onSelect: () => void;
  onRun: (sql: string) => void | Promise<void>;
  running: boolean;
}) {
  const statusChip: Record<typeof step.status, { label: string; className: string }> = {
    ok: { label: "ok", className: "bg-success/15 text-success" },
    guard_rejected: {
      label: "guard rejected",
      className: "bg-error/15 text-error",
    },
    execution_error: {
      label: "error",
      className: "bg-error/15 text-error",
    },
    budget_exceeded: {
      label: "budget",
      className: "bg-amber/25 text-[#b7791f]",
    },
    column_not_in_doc: {
      label: "col not found",
      className: "bg-amber/25 text-[#b7791f]",
    },
  };
  const chip = statusChip[step.status] ?? statusChip.ok;

  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        "block w-full rounded-xl border p-3 text-left transition-all",
        isActive
          ? "border-navy bg-white shadow-md"
          : "border-hairline bg-white/70 hover:border-navy/60 hover:bg-white",
      )}
    >
      <div className="mb-2 flex items-center gap-2">
        <span className="font-mono text-[10px] text-muted">
          {String(index + 1).padStart(2, "0")}
        </span>
        <span className="rounded-full bg-navy/10 px-2 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-wider text-navy">
          SQL
        </span>
        <span
          className={cn(
            "rounded-full px-2 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-wider",
            chip.className,
          )}
        >
          {chip.label}
        </span>
        {running && <Loader2 className="ml-auto h-3.5 w-3.5 animate-spin text-muted" />}
      </div>
      <pre className="line-clamp-3 whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed text-ink">
        {step.sql || "(streaming…)"}
      </pre>
      <p className="mt-2 flex items-center gap-2 font-mono text-[10px] text-muted">
        <span>{step.rowCount.toLocaleString()} rows</span>
        <span aria-hidden="true">·</span>
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onSelect();
          }}
          className="inline-flex items-center gap-1 hover:text-ink"
        >
          <Pencil className="h-3 w-3" /> edit
        </button>
      </p>
    </button>
  );
}

function ActiveStepView({
  step,
  onRun,
  busy,
}: {
  step: Extract<ExplorerStep, { type: "sql" }>;
  onRun: (sql: string) => void | Promise<void>;
  busy: boolean;
}) {
  const [draft, setDraft] = React.useState(step.sql);
  const [editing, setEditing] = React.useState(false);

  React.useEffect(() => {
    setDraft(step.sql);
    setEditing(false);
  }, [step.id, step.sql]);

  const run = async () => {
    await onRun(draft);
    setEditing(false);
  };
  const pending = busy;

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-4">
      <header className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-wider text-muted">
            Active step
          </p>
          <h2 className="font-display text-2xl font-medium text-ink">
            {step.status === "ok"
              ? `${step.rowCount.toLocaleString()} rows`
              : truncate(step.reason ?? step.status, 90)}
          </h2>
          {step.elapsedMs != null && (
            <p className="font-mono text-[11px] text-muted">
              {step.elapsedMs} ms
              {step.bytesBilled != null && ` · ${step.bytesBilled} bytes billed`}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          {editing ? (
            <>
              <button
                type="button"
                onClick={() => {
                  setDraft(step.sql);
                  setEditing(false);
                }}
                className="rounded-md border border-hairline bg-white px-3 py-1.5 text-sm text-ink hover:bg-surface-soft"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={run}
                disabled={pending || draft.trim().length === 0}
                className="inline-flex items-center gap-1.5 rounded-md bg-coral px-3 py-1.5 text-sm font-medium text-ink hover:bg-coral-active disabled:cursor-not-allowed disabled:opacity-50"
              >
                {pending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Play className="h-4 w-4" />
                )}
                Run
              </button>
            </>
          ) : (
            <button
              type="button"
              onClick={() => setEditing(true)}
              className="inline-flex items-center gap-1.5 rounded-md border border-hairline bg-white px-3 py-1.5 text-sm text-ink hover:bg-surface-soft"
            >
              <Pencil className="h-4 w-4" /> Edit SQL
            </button>
          )}
        </div>
      </header>

      {editing ? (
        <Textarea
          autoFocus
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          rows={12}
          spellCheck={false}
          className="min-h-[280px] resize-y font-mono text-[13px] leading-relaxed"
        />
      ) : (
        <SqlBlock
          sql={step.sql || "-- awaiting SQL"}
          status={
            step.status === "ok"
              ? "accepted"
              : step.status === "guard_rejected"
                ? "rejected"
                : "pending"
          }
          reason={step.reason ?? null}
        />
      )}

      <RowsTable
        rows={step.rows}
        maxRows={100}
        caption={
          step.status !== "ok" && step.reason ? (
            <span className="text-error">{step.reason}</span>
          ) : (
            <span>
              First {Math.min(step.rows.length, 100)} of{" "}
              {step.rowCount.toLocaleString()} rows
            </span>
          )
        }
      />
    </div>
  );
}
