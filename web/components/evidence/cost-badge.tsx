import { cn, formatDollars } from "@/lib/utils";

export interface CostBadgeProps {
  dollars: number;
  toolCalls?: number;
  elapsedMs?: number | null;
  cached?: boolean;
  className?: string;
}

export function CostBadge({
  dollars,
  toolCalls,
  elapsedMs,
  cached,
  className,
}: CostBadgeProps) {
  return (
    <div
      className={cn(
        "flex flex-wrap items-center gap-3 rounded-lg border border-hairline bg-white px-3 py-2 text-[11px] text-muted",
        className,
      )}
    >
      <div className="flex items-center gap-1.5">
        <span className="font-mono uppercase tracking-wider">Cost</span>
        <span className="font-mono font-semibold text-ink">
          {formatDollars(dollars)}
        </span>
      </div>
      {toolCalls != null && (
        <div className="flex items-center gap-1.5">
          <span className="font-mono uppercase tracking-wider">Tools</span>
          <span className="font-mono font-semibold text-ink">{toolCalls}</span>
        </div>
      )}
      {elapsedMs != null && (
        <div className="flex items-center gap-1.5">
          <span className="font-mono uppercase tracking-wider">Elapsed</span>
          <span className="font-mono font-semibold text-ink">
            {elapsedMs < 1000
              ? `${elapsedMs} ms`
              : `${(elapsedMs / 1000).toFixed(1)} s`}
          </span>
        </div>
      )}
      {cached && (
        <span className="rounded-full bg-teal/15 px-2 py-0.5 font-mono text-[10px] font-semibold text-teal">
          Cached
        </span>
      )}
    </div>
  );
}
