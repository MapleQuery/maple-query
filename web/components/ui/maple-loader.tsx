"use client";

import * as React from "react";
import { MapleLeaf } from "./maple-leaf";
import { cn } from "@/lib/utils";

/**
 * Verbs Canadians use for "thinking". Rotated through so the loading
 * state has some texture instead of a bare spinner.
 */
export const CANADIAN_VERBS = [
  "Portaging",
  "Zambonieing",
  "Pondering",
  "Snowshoeing",
  "Tobogganing",
  "Percolating",
  "Deliberating",
  "Ruminating",
  "Sledding",
  "Musing",
  "Chinwagging",
  "Sugarshacking",
] as const;

export interface MapleLoaderProps {
  className?: string;
  /** Size of the maple leaf in px. */
  size?: number;
  /** Show a verb next to the leaf. */
  withVerb?: boolean;
  /** How the verb sits relative to the leaf. */
  layout?: "row" | "stack";
  /** Rotation interval for the verb, ms. */
  intervalMs?: number;
}

/**
 * Pulsing maple leaf + rotating Canadian verb. The one loading indicator
 * for the app — inline in messages, in `loading.tsx` files, and inside
 * per-page transient screens.
 */
export function MapleLoader({
  className,
  size = 40,
  withVerb = true,
  layout = "stack",
  intervalMs = 1600,
}: MapleLoaderProps) {
  const [verbIdx, setVerbIdx] = React.useState(() =>
    Math.floor(Math.random() * CANADIAN_VERBS.length),
  );

  React.useEffect(() => {
    if (!withVerb) return;
    const id = window.setInterval(() => {
      setVerbIdx((i) => (i + 1) % CANADIAN_VERBS.length);
    }, intervalMs);
    return () => window.clearInterval(id);
  }, [withVerb, intervalMs]);

  return (
    <span
      role="status"
      aria-live="polite"
      className={cn(
        "inline-flex items-center",
        layout === "stack" ? "flex-col gap-3" : "gap-2.5",
        className,
      )}
    >
      <MapleLeaf pulse size={size} />
      {withVerb && (
        <span className="font-mono text-[10px] font-medium uppercase tracking-[0.14em] text-navy">
          {CANADIAN_VERBS[verbIdx]}
        </span>
      )}
    </span>
  );
}

/**
 * Full-viewport centered loader for `loading.tsx` route boundaries.
 */
export function PageLoader({ label }: { label?: string }) {
  return (
    <div className="grid min-h-[calc(100vh-4rem)] place-items-center">
      <div className="flex flex-col items-center gap-4">
        <MapleLoader size={56} />
        {label && (
          <span className="text-sm text-muted">{label}</span>
        )}
      </div>
    </div>
  );
}
