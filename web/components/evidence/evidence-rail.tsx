"use client";

import * as React from "react";
import {
  Database,
  Columns3,
  Table,
  FileCode,
  ShieldCheck,
  ShieldAlert,
  Sparkles,
  Sigma,
  AlertTriangle,
  Clock,
  CheckCircle2,
} from "lucide-react";
import { DatasetCard } from "./dataset-card";
import { ColumnList } from "./column-list";
import { SqlBlock } from "./sql-block";
import { RowsTable } from "./rows-table";
import { Skeleton } from "@/components/ui/skeleton";
import { cn, formatElapsed } from "@/lib/utils";
import type {
  ColumnCandidateT,
  DatasetCandidateT,
  DerivationT,
} from "@/lib/types";

export type RailCard =
  | {
      id: string;
      kind: "retrieval_started";
      query: string;
      k: number;
    }
  | {
      id: string;
      kind: "datasets_ranked";
      candidates: DatasetCandidateT[];
    }
  | {
      id: string;
      kind: "columns_ranked";
      packageIds: string[];
      candidates: ColumnCandidateT[];
    }
  | {
      id: string;
      kind: "sample_rows";
      packageId: string;
      rows: Record<string, unknown>[];
    }
  | {
      id: string;
      kind: "sql_generated";
      sql: string;
      rationale: string;
      guard?: {
        accepted: boolean;
        reason: string | null;
        sql_final: string;
      };
      executed?: {
        row_count: number;
        elapsed_ms: number | null;
        rows: Record<string, unknown>[];
      };
    }
  | {
      id: string;
      kind: "tool_error";
      tool: string;
      message: string;
    }
  | {
      id: string;
      kind: "budget_exceeded";
      which: "tool_calls" | "sql_executions";
      value: number;
      cap: number;
    }
  | {
      id: string;
      kind: "turn_timeout";
      elapsed_ms: number;
      cap_ms: number;
    }
  | {
      id: string;
      kind: "derivation";
      derivation: DerivationT;
    };

export interface EvidenceRailProps {
  cards: RailCard[];
  isStreaming?: boolean;
  cached?: boolean;
  className?: string;
}

export function EvidenceRail({
  cards,
  isStreaming,
  cached,
  className,
}: EvidenceRailProps) {
  return (
    <div className={cn("flex h-full flex-col", className)}>
      <div className="border-b border-hairline bg-canvas/70 px-5 py-4">
        <h2 className="font-display text-lg font-medium text-ink">Evidence</h2>
        <p className="text-xs text-muted">
          Live trace of retrieval, guardrails, and SQL behind the answer.
        </p>
      </div>
      <div className="flex-1 overflow-y-auto px-5 py-5">
        {cards.length === 0 && !isStreaming && (
          <EmptyRail />
        )}
        {cards.length === 0 && isStreaming && <SearchingSkeleton />}
        <ol className="space-y-3">
          {cards.map((card, i) => (
            <RailItem key={card.id} card={card} index={i + 1} />
          ))}
        </ol>
        {cached && cards.length > 0 && (
          <div className="mt-4 flex items-center gap-1.5 rounded-lg border border-dashed border-hairline bg-white/50 px-3 py-2 text-[11px] text-muted">
            <CheckCircle2 className="h-3 w-3 text-teal" />
            Replayed from cache. Identical question, warm result.
          </div>
        )}
      </div>
    </div>
  );
}

function EmptyRail() {
  return (
    <div className="rounded-xl border border-dashed border-hairline bg-white/50 p-6 text-center">
      <Sparkles className="mx-auto mb-2 h-5 w-5 text-navy" />
      <p className="text-sm font-medium text-ink">
        Ask a question to see the evidence trace
      </p>
      <p className="mt-1 text-xs text-muted">
        MapleQuery only answers from cited datasets. Every step lands here.
      </p>
    </div>
  );
}

function SearchingSkeleton() {
  return (
    <ol className="space-y-3">
      {[0, 1].map((i) => (
        <li
          key={i}
          className="space-y-2 rounded-xl border border-hairline bg-white p-4"
        >
          <Skeleton className="h-3 w-2/3" />
          <Skeleton className="h-3 w-full" />
          <Skeleton className="h-3 w-1/2" />
        </li>
      ))}
    </ol>
  );
}

function RailItem({ card, index }: { card: RailCard; index: number }) {
  switch (card.kind) {
    case "retrieval_started":
      return (
        <RailShell
          index={index}
          icon={<Database className="h-4 w-4" />}
          title="Searching datasets"
          meta={`k=${card.k}`}
        >
          <p className="text-xs text-body">
            Query: <span className="font-mono text-ink">{card.query}</span>
          </p>
        </RailShell>
      );

    case "datasets_ranked":
      return (
        <RailShell
          index={index}
          icon={<Database className="h-4 w-4" />}
          title="Candidate datasets"
          meta={`${card.candidates.length} ranked`}
        >
          <ol className="mt-2 space-y-2">
            {card.candidates.slice(0, 5).map((c, i) => (
              <DatasetCard key={c.package_id} index={i + 1} candidate={c} />
            ))}
          </ol>
        </RailShell>
      );

    case "columns_ranked":
      return (
        <RailShell
          index={index}
          icon={<Columns3 className="h-4 w-4" />}
          title="Candidate columns"
          meta={`${card.candidates.length} ranked`}
        >
          <div className="mt-2">
            <ColumnList candidates={card.candidates} />
          </div>
        </RailShell>
      );

    case "sample_rows":
      return (
        <RailShell
          index={index}
          icon={<Table className="h-4 w-4" />}
          title="Sample rows"
          meta={card.packageId}
        >
          <div className="mt-2">
            <RowsTable rows={card.rows} maxRows={5} />
          </div>
        </RailShell>
      );

    case "sql_generated": {
      const status = card.guard
        ? card.guard.accepted
          ? "accepted"
          : "rejected"
        : "pending";
      return (
        <RailShell
          index={index}
          icon={
            card.guard?.accepted === false ? (
              <ShieldAlert className="h-4 w-4 text-error" />
            ) : (
              <ShieldCheck className="h-4 w-4 text-navy" />
            )
          }
          title={card.guard ? "SQL + guardrails" : "SQL generated"}
        >
          <div className="mt-2 space-y-2">
            <SqlBlock
              sql={card.guard?.sql_final ?? card.sql}
              rationale={card.rationale}
              status={status}
              reason={card.guard?.reason ?? null}
            />
            {card.executed && (
              <RowsTable
                rows={card.executed.rows}
                caption={
                  <>
                    <span>
                      {card.executed.row_count.toLocaleString()} rows ·{" "}
                      {formatElapsed(card.executed.elapsed_ms)}
                    </span>
                  </>
                }
              />
            )}
          </div>
        </RailShell>
      );
    }

    case "tool_error":
      return (
        <RailShell
          index={index}
          icon={<AlertTriangle className="h-4 w-4 text-error" />}
          title="Tool error"
          meta={card.tool}
        >
          <p className="mt-1 text-xs text-error">{card.message}</p>
        </RailShell>
      );

    case "budget_exceeded":
      return (
        <RailShell
          index={index}
          icon={<AlertTriangle className="h-4 w-4 text-amber" />}
          title="Budget exceeded"
          meta={card.which.replace("_", " ")}
        >
          <p className="mt-1 text-xs text-body">
            Cap {card.cap} reached at value {card.value}. The model was asked to
            produce a best-effort answer with what it has.
          </p>
        </RailShell>
      );

    case "turn_timeout":
      return (
        <RailShell
          index={index}
          icon={<Clock className="h-4 w-4 text-error" />}
          title="Turn timeout"
          meta={formatElapsed(card.elapsed_ms)}
        >
          <p className="mt-1 text-xs text-body">
            Wall-clock cap {formatElapsed(card.cap_ms)} exceeded. Please
            re-ask.
          </p>
        </RailShell>
      );

    case "derivation":
      return <DerivationCard index={index} derivation={card.derivation} />;

    default:
      // exhaustiveness for TS
      return null;
  }
}

const FLAG_LABELS: Record<string, string> = {
  cross_source_sum: "summed across multiple datasets",
  unknown_units: "units unverified",
  ungrounded: "figure not tied to a computed total",
};

function formatDerivedValue(d: DerivationT): string {
  if (d.result_value === null) return "computed figure";
  const scaleNote =
    d.unit_scale === "unknown"
      ? " (units unverified)"
      : d.unit_scale !== "dollars" && d.unit_scale !== "count"
        ? ` (column reported in ${d.unit_scale})`
        : "";
  const abs = Math.abs(d.result_value);
  const magnitude =
    abs >= 1e9
      ? `${(d.result_value / 1e9).toFixed(1)}B`
      : abs >= 1e6
        ? `${(d.result_value / 1e6).toFixed(1)}M`
        : abs >= 1e3
          ? `${(d.result_value / 1e3).toFixed(1)}K`
          : `${d.result_value}`;
  const prefix = d.unit_scale === "count" ? "" : "$";
  return `${prefix}${magnitude}${scaleNote}`;
}

function DerivationCard({
  index,
  derivation: d,
}: {
  index: number;
  derivation: DerivationT;
}) {
  const how = `${d.aggregation}${
    d.value_columns.length ? ` of ${d.value_columns.join(", ")}` : ""
  }`;
  return (
    <RailShell
      index={index}
      icon={<Sigma className="h-4 w-4 text-body" />}
      title="How I got this number"
      meta={formatDerivedValue(d)}
    >
      <dl className="mt-1 space-y-1 text-xs text-body">
        {d.dataset_titles.length > 0 && (
          <div>
            <dt className="inline font-medium text-muted">From: </dt>
            <dd className="inline">{d.dataset_titles.join(", ")}</dd>
          </div>
        )}
        <div>
          <dt className="inline font-medium text-muted">How: </dt>
          <dd className="inline">{how}</dd>
        </div>
        <div>
          <dt className="inline font-medium text-muted">Over: </dt>
          <dd className="inline">
            ~{d.source_row_estimate.toLocaleString()} rows
          </dd>
        </div>
      </dl>
      {d.flags.length > 0 && (
        <ul className="mt-2 flex flex-wrap gap-1">
          {d.flags.map((flag) => (
            <li
              key={flag}
              className="rounded-full border border-hairline bg-amber-50 px-2 py-0.5 text-[10px] font-medium text-amber-700"
            >
              {FLAG_LABELS[flag] ?? flag}
            </li>
          ))}
        </ul>
      )}
    </RailShell>
  );
}

function RailShell({
  index,
  icon,
  title,
  meta,
  children,
}: {
  index: number;
  icon: React.ReactNode;
  title: string;
  meta?: React.ReactNode;
  children?: React.ReactNode;
}) {
  return (
    <li className="animate-rise rounded-xl border border-hairline bg-white p-4 shadow-sm">
      <div className="flex items-center gap-2">
        <span className="grid h-6 w-6 place-items-center rounded-full bg-coral/15 text-navy">
          {icon}
        </span>
        <p className="text-sm font-semibold text-ink">{title}</p>
        <span className="ml-auto font-mono text-[10px] uppercase tracking-wider text-muted">
          {String(index).padStart(2, "0")}
          {meta ? ` · ${meta}` : ""}
        </span>
      </div>
      {children && <div className="mt-2">{children}</div>}
    </li>
  );
}

// Wrap the icon-container in a coloured background based on the card kind
// via wrapper elements above; RailShell can be extended later if we want
// per-kind color chips.
export function railShellExports() {
  return { FileCode };
}
