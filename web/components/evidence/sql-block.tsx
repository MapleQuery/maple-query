"use client";

import * as React from "react";
import { Copy, Check, AlertTriangle } from "lucide-react";
import { highlightSql } from "@/lib/highlight";
import { cn } from "@/lib/utils";

export interface SqlBlockProps {
  sql: string;
  rationale?: string;
  status?: "pending" | "accepted" | "rejected";
  reason?: string | null;
  className?: string;
}

export function SqlBlock({
  sql,
  rationale,
  status = "pending",
  reason,
  className,
}: SqlBlockProps) {
  const [html, setHtml] = React.useState<string | null>(null);
  const [copied, setCopied] = React.useState(false);

  React.useEffect(() => {
    let cancelled = false;
    highlightSql(sql).then((h) => {
      if (!cancelled) setHtml(h);
    }).catch(() => {
      if (!cancelled) setHtml(null);
    });
    return () => {
      cancelled = true;
    };
  }, [sql]);

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(sql);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {
      /* ignore */
    }
  };

  return (
    <div
      className={cn(
        "overflow-hidden rounded-xl border border-hairline bg-white",
        className,
      )}
    >
      <div className="flex items-center gap-2 border-b border-hairline bg-surface-soft px-4 py-2">
        <span className="font-mono text-[11px] font-medium uppercase tracking-wider text-muted">
          SQL
        </span>
        {status === "accepted" && (
          <span className="inline-flex items-center gap-1 rounded-full bg-success/15 px-2 py-0.5 text-[10px] font-semibold text-success">
            <Check className="h-3 w-3" /> guard accepted
          </span>
        )}
        {status === "rejected" && (
          <span className="inline-flex items-center gap-1 rounded-full bg-error/15 px-2 py-0.5 text-[10px] font-semibold text-error">
            <AlertTriangle className="h-3 w-3" /> guard rejected
          </span>
        )}
        <button
          type="button"
          onClick={onCopy}
          className="ml-auto inline-flex items-center gap-1 rounded-md border border-hairline bg-white px-2 py-1 text-[11px] font-medium text-muted transition-colors hover:text-ink"
        >
          {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <div className="shiki-block">
        {html ? (
          <div dangerouslySetInnerHTML={{ __html: html }} />
        ) : (
          <pre className="whitespace-pre-wrap font-mono text-[12.5px] leading-[1.55]">
            {sql}
          </pre>
        )}
      </div>
      {rationale && (
        <p className="border-t border-hairline bg-surface-soft/60 px-4 py-2 text-xs text-body">
          <span className="font-mono text-[10px] uppercase tracking-wider text-muted">
            Rationale ·{" "}
          </span>
          {rationale}
        </p>
      )}
      {status === "rejected" && reason && (
        <p className="border-t border-hairline bg-error/5 px-4 py-2 text-xs text-error">
          {reason}
        </p>
      )}
    </div>
  );
}
