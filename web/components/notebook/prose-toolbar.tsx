"use client";

import * as React from "react";
import { Bold, Italic, Palette, Type } from "lucide-react";
import {
  PROSE_COLORS,
  PROSE_SIZES,
  applySpanStyle,
  setHeading,
  toggleWrap,
  type EditResult,
  type Selection,
} from "@/lib/prose";

export interface ProseToolbarProps {
  markdown: string;
  /** Live selection in the textarea this toolbar drives. */
  selection: Selection;
  onApply: (result: EditResult) => void;
}

const HEADINGS = [
  { level: 0, label: "Body" },
  { level: 1, label: "H1" },
  { level: 2, label: "H2" },
  { level: 3, label: "H3" },
];

/**
 * Formatting controls over the Markdown source.
 *
 * Every button is a pure string transform (`lib/prose.ts`) applied to
 * the text and the current selection — the textarea stays the editor and
 * Markdown stays what is stored. A WYSIWYG surface would have to own the
 * document model, and there are three renderers downstream (screen,
 * print, Word) that all read the same string today.
 *
 * `onMouseDown` + `preventDefault` on every control: clicking a button
 * must not blur the textarea, or the selection the transform operates on
 * is gone before the click lands.
 */
export function ProseToolbar({
  markdown,
  selection,
  onApply,
}: ProseToolbarProps) {
  const hasSelection = selection.end > selection.start;

  return (
    <div
      className="mb-2 flex flex-wrap items-center gap-1 rounded-md border border-hairline bg-surface-soft/60 p-1"
      onMouseDown={(e) => e.preventDefault()}
    >
      <div className="flex items-center">
        {HEADINGS.map((h) => (
          <button
            key={h.level}
            type="button"
            title={h.level === 0 ? "Body text" : `Heading ${h.level}`}
            onClick={() => onApply(setHeading(markdown, selection, h.level))}
            className="rounded px-2 py-1 text-xs font-medium text-muted hover:bg-white hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy"
          >
            {h.label}
          </button>
        ))}
      </div>

      <Divider />

      <ToolButton
        title="Bold"
        onClick={() => onApply(toggleWrap(markdown, selection, "**"))}
      >
        <Bold className="h-3.5 w-3.5" />
      </ToolButton>
      <ToolButton
        title="Italic"
        onClick={() => onApply(toggleWrap(markdown, selection, "_"))}
      >
        <Italic className="h-3.5 w-3.5" />
      </ToolButton>

      <Divider />

      {/* Size and colour need something to attach to, so they stay
          disabled until there is a selection rather than silently
          doing nothing. */}
      <Menu
        title="Font size"
        icon={<Type className="h-3.5 w-3.5" />}
        disabled={!hasSelection}
      >
        {(close) =>
          PROSE_SIZES.map((s) => (
            <button
              key={s.value}
              type="button"
              onClick={() => {
                onApply(
                  applySpanStyle(markdown, selection, { size: s.value }),
                );
                close();
              }}
              className="flex w-full items-center justify-between gap-4 rounded px-2 py-1.5 text-left text-xs text-ink hover:bg-surface-soft"
            >
              <span style={{ fontSize: s.value }}>{s.label}</span>
            </button>
          ))
        }
      </Menu>

      <Menu
        title="Text colour"
        icon={<Palette className="h-3.5 w-3.5" />}
        disabled={!hasSelection}
      >
        {(close) => (
          <div className="flex gap-1 p-0.5">
            {PROSE_COLORS.map((c) => (
              <button
                key={c.hex}
                type="button"
                title={c.label}
                aria-label={c.label}
                onClick={() => {
                  onApply(
                    applySpanStyle(markdown, selection, { color: c.hex }),
                  );
                  close();
                }}
                className="h-5 w-5 rounded border border-hairline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy"
                style={{ backgroundColor: c.hex }}
              />
            ))}
          </div>
        )}
      </Menu>

      <span className="ml-auto pr-1 font-mono text-[10px] uppercase tracking-wider text-muted">
        Markdown
      </span>
    </div>
  );
}

function Divider() {
  return <span className="mx-0.5 h-4 w-px bg-hairline" />;
}

function ToolButton({
  title,
  onClick,
  children,
}: {
  title: string;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      title={title}
      aria-label={title}
      onClick={onClick}
      className="rounded p-1.5 text-muted hover:bg-white hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy"
    >
      {children}
    </button>
  );
}

function Menu({
  title,
  icon,
  disabled,
  children,
}: {
  title: string;
  icon: React.ReactNode;
  disabled?: boolean;
  children: (close: () => void) => React.ReactNode;
}) {
  const [open, setOpen] = React.useState(false);
  const ref = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", onDown);
    return () => document.removeEventListener("pointerdown", onDown);
  }, [open]);

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        title={title}
        aria-label={title}
        aria-haspopup="menu"
        aria-expanded={open}
        disabled={disabled}
        onClick={() => setOpen((v) => !v)}
        className="rounded p-1.5 text-muted hover:bg-white hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy disabled:cursor-not-allowed disabled:opacity-40"
      >
        {icon}
      </button>
      {open && (
        <div className="absolute left-0 top-8 z-20 min-w-[7rem] rounded-lg border border-hairline bg-white p-1 shadow-lg">
          {children(() => setOpen(false))}
        </div>
      )}
    </div>
  );
}
