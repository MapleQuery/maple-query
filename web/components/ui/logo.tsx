import { cn } from "@/lib/utils";

export function LogoMark({ className }: { className?: string }) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "grid h-8 w-8 place-items-center rounded-md bg-navy text-white",
        className,
      )}
    >
      <svg viewBox="0 0 80 80" className="h-5 w-5" fill="none">
        <circle cx="34" cy="34" r="20" stroke="currentColor" strokeWidth="4" />
        <line
          x1="48"
          y1="48"
          x2="66"
          y2="66"
          stroke="currentColor"
          strokeWidth="5"
          strokeLinecap="round"
        />
        <rect x="24" y="26" width="20" height="4" rx="2" fill="currentColor" />
        <rect x="24" y="33" width="14" height="4" rx="2" fill="currentColor" />
        <rect x="24" y="40" width="18" height="4" rx="2" fill="currentColor" />
      </svg>
    </span>
  );
}
