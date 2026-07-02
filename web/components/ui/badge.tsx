import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-[11px] font-semibold uppercase tracking-wider",
  {
    variants: {
      variant: {
        default: "border-hairline bg-white text-muted",
        navy: "border-transparent bg-navy/10 text-navy",
        coral: "border-transparent bg-coral/15 text-navy",
        success: "border-transparent bg-success/15 text-success",
        amber: "border-transparent bg-amber/25 text-[#b7791f]",
        error: "border-transparent bg-error/15 text-error",
      },
    },
    defaultVariants: { variant: "default" },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <span className={cn(badgeVariants({ variant, className }))} {...props} />
  );
}
