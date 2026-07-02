import { cn } from "@/lib/utils";
import type { ColumnCandidateT } from "@/lib/types";

export interface ColumnListProps {
  candidates: ColumnCandidateT[];
  className?: string;
}

export function ColumnList({ candidates, className }: ColumnListProps) {
  if (candidates.length === 0) return null;
  const byPkg = new Map<string, ColumnCandidateT[]>();
  for (const c of candidates) {
    const arr = byPkg.get(c.package_id) ?? [];
    arr.push(c);
    byPkg.set(c.package_id, arr);
  }

  return (
    <div className={cn("space-y-3", className)}>
      {Array.from(byPkg.entries()).map(([pkgId, cols]) => (
        <div
          key={pkgId}
          className="rounded-xl border border-hairline bg-white p-4"
        >
          <p className="mb-2 font-mono text-[11px] text-muted">{pkgId}</p>
          <ul className="space-y-2">
            {cols.slice(0, 8).map((c, i) => (
              <li
                key={`${c.column_name}-${i}`}
                className="flex items-start justify-between gap-3 border-t border-hairline/60 pt-2 first:border-t-0 first:pt-0"
              >
                <div className="min-w-0">
                  <p className="font-mono text-[12px] font-medium text-ink">
                    {c.column_name}
                  </p>
                  {c.description && (
                    <p className="mt-0.5 line-clamp-2 text-[11px] leading-relaxed text-body">
                      {c.description}
                    </p>
                  )}
                </div>
                {typeof c.distance === "number" && (
                  <span className="shrink-0 font-mono text-[10px] text-muted">
                    d={c.distance.toFixed(3)}
                  </span>
                )}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
