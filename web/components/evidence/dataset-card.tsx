"use client";

import Link from "next/link";
import { ArrowUpRight } from "lucide-react";
import { cn } from "@/lib/utils";
import type { DatasetCandidateT } from "@/lib/types";

export interface DatasetCardProps {
  index: number;
  candidate: DatasetCandidateT;
  isActive?: boolean;
  onHighlight?: () => void;
}

export function DatasetCard({
  index,
  candidate,
  isActive,
  onHighlight,
}: DatasetCardProps) {
  // Prefer the real title. Don't slice the summary into a pseudo-title —
  // it truncated mid-word ("…across multiple o") and duplicated the
  // summary shown just below. The summary still renders as its own line.
  const title = candidate.title?.trim() || "Untitled dataset";

  return (
    <li
      id={`ds-${candidate.package_id}`}
      onMouseEnter={onHighlight}
      className={cn(
        "scroll-mt-20 rounded-xl border border-hairline bg-white transition-shadow duration-200 hover:shadow-sm",
        isActive && "src-flash",
      )}
    >
      <Link
        href={`/datasets/${candidate.package_id}`}
        className="group block rounded-xl p-4 focus:outline-none focus-visible:ring-2 focus-visible:ring-coral"
      >
        <div className="mb-2 flex items-start justify-between gap-2">
          <span className="grid h-6 w-6 shrink-0 place-items-center rounded-full bg-coral/15 text-[11px] font-semibold text-navy">
            {index}
          </span>
          {typeof candidate.distance === "number" && (
            <span className="rounded-full bg-navy/10 px-2 py-0.5 font-mono text-[10px] font-medium text-navy">
              d={candidate.distance.toFixed(3)}
            </span>
          )}
        </div>
        <h3 className="text-sm font-semibold text-ink group-hover:underline">
          {title}
        </h3>
        {candidate.summary && (
          <p className="mt-1 line-clamp-3 text-xs leading-relaxed text-body">
            {candidate.summary}
          </p>
        )}
        <div className="mt-3 flex items-center justify-between">
          <span className="font-mono text-[11px] text-muted">
            {candidate.package_id}
          </span>
          <span className="inline-flex items-center gap-1 text-xs font-medium text-navy group-hover:underline">
            Open
            <ArrowUpRight className="h-3 w-3 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
          </span>
        </div>
      </Link>
    </li>
  );
}
