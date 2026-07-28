"use client";

import * as React from "react";
import { ChevronDown, Download, FileDown, FileText } from "lucide-react";
import { cn } from "@/lib/utils";

export interface ExportMenuProps {
  onExportMarkdown: () => void;
  onExportPdf: () => void;
  disabled?: boolean;
}

/** One Export control; the format is a choice inside it rather than a
 * button per format, so adding a third format costs no header space. */
export function ExportMenu({
  onExportMarkdown,
  onExportPdf,
  disabled,
}: ExportMenuProps) {
  const [open, setOpen] = React.useState(false);
  const wrapRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: PointerEvent) => {
      if (!wrapRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  const choose = (run: () => void) => {
    setOpen(false);
    run();
  };

  return (
    <div ref={wrapRef} className="relative">
      <button
        type="button"
        disabled={disabled}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1.5 rounded-md border border-hairline bg-white px-3 py-2 text-sm font-medium text-ink hover:bg-surface-soft focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy disabled:cursor-not-allowed disabled:opacity-40"
      >
        <Download className="h-4 w-4" /> Export
        <ChevronDown
          className={cn(
            "h-3.5 w-3.5 text-muted transition-transform",
            open && "rotate-180",
          )}
        />
      </button>
      {open && (
        <div
          role="menu"
          aria-label="Export format"
          className="absolute right-0 top-full z-20 mt-1 w-52 rounded-lg border border-hairline bg-white p-1 shadow-lg"
        >
          <ExportMenuItem
            icon={<FileText className="h-3.5 w-3.5 text-muted" />}
            label="Markdown"
            hint=".md"
            onSelect={() => choose(onExportMarkdown)}
          />
          <ExportMenuItem
            icon={<FileDown className="h-3.5 w-3.5 text-muted" />}
            label="PDF"
            hint=".pdf"
            onSelect={() => choose(onExportPdf)}
          />
        </div>
      )}
    </div>
  );
}

function ExportMenuItem({
  icon,
  label,
  hint,
  onSelect,
}: {
  icon: React.ReactNode;
  label: string;
  hint: string;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      role="menuitem"
      onClick={onSelect}
      className="flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-sm text-ink hover:bg-surface-soft focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy"
    >
      {icon}
      <span className="flex-1">{label}</span>
      <span className="font-mono text-[10px] text-muted">{hint}</span>
    </button>
  );
}
