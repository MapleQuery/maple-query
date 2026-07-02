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
  const title =
    candidate.title?.trim() ||
    candidate.summary?.split(/\.\s+/)[0]?.slice(0, 90) ||
    candidate.package_id;

  return (
    <li
      id={`ds-${candidate.package_id}`}
      onMouseEnter={onHighlight}
      className={cn(
        "scroll-mt-20 rounded-xl border border-hairline bg-white p-4 transition-shadow duration-200 hover:shadow-sm",
        isActive && "src-flash",
      )}
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
      <h3 className="text-sm font-semibold text-ink">{title}</h3>
      {candidate.summary && (
        <p className="mt-1 line-clamp-3 text-xs leading-relaxed text-body">
          {candidate.summary}
        </p>
      )}
      <div className="mt-3 flex items-center justify-between">
        <span className="font-mono text-[11px] text-muted">
          {candidate.package_id}
        </span>
        <Link
          href={`/datasets/${candidate.package_id}`}
          className="inline-flex items-center gap-1 text-xs font-medium text-navy hover:underline"
        >
          Open
          <ArrowUpRight className="h-3 w-3" />
        </Link>
      </div>
    </li>
  );
}
