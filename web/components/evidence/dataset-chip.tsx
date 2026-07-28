"use client";

import Link from "next/link";
import { cn, truncate } from "@/lib/utils";

export interface DatasetChipProps {
  packageId: string;
  /** Resolved title. Absent while a backfill is in flight, or when the
   * package has no title at all — the chip then shows the id, which is
   * still a working link. */
  title?: string;
}

/**
 * A dataset reference rendered as its name. The id stays reachable in
 * the tooltip and the href, so a chip is still traceable back to the
 * row it came from.
 */
export function DatasetChip({ packageId, title }: DatasetChipProps) {
  const label = title?.trim();
  return (
    <Link
      href={`/datasets/${packageId}`}
      title={label ? `${label} · ${packageId}` : packageId}
      className={cn(
        "max-w-full rounded bg-surface-soft px-1.5 py-0.5 text-navy hover:underline",
        label ? "text-[11px]" : "font-mono text-[10px]",
      )}
    >
      {label ? truncate(label, 52) : truncate(packageId, 24)}
    </Link>
  );
}
