import Image from "next/image";
import { cn } from "@/lib/utils";

export interface MapleLeafProps {
  className?: string;
  /** Add the growing/shrinking pulse used by loaders. */
  pulse?: boolean;
  /** Rendered size in px (drives both width and height). */
  size?: number;
  priority?: boolean;
}

/**
 * The brand mark. A single <Image> so Next can serve modern formats and
 * so every surface (header, favicon-adjacent contexts, loaders) shares
 * one source of truth.
 */
export function MapleLeaf({
  className,
  pulse = false,
  size = 32,
  priority = false,
}: MapleLeafProps) {
  return (
    <Image
      src="/brand/maple-leaf.webp"
      alt=""
      aria-hidden="true"
      width={size}
      height={size}
      priority={priority}
      className={cn(
        "select-none",
        pulse && "origin-center animate-leaf-pulse",
        className,
      )}
    />
  );
}
