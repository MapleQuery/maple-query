"use client";

import * as React from "react";
import Link from "next/link";
import { BookPlus, Check, ChevronDown, Copy, FilePlus2 } from "lucide-react";
import { notebooks } from "@/lib/storage";
import { addTurnsToNotebook, type TurnSnapshot } from "@/lib/turn-to-block";
import { useToast } from "@/components/ui/toast";
import { track } from "@/lib/analytics";
import { cn } from "@/lib/utils";

/**
 * Under every finished answer: send it to a notebook, or copy it.
 *
 * The answer moves as a finished block — text, SQL, rows, sources and a
 * chart when the rows have a shape — so nothing re-runs. After a move
 * the row says where it went and links there, instead of a toast that
 * vanishes before the user decides to follow it.
 */
export function TurnActions({
  turn,
  linked,
  conversationId,
  conversationTitle,
  onAdded,
}: {
  turn: TurnSnapshot;
  /** The notebook this conversation feeds, once it feeds one. The main
   * button appends there, so a notebook builds up answer by answer
   * instead of each click minting a new one. */
  linked: { id: string; title: string } | null;
  conversationId: string;
  conversationTitle: string;
  onAdded: (nb: { id: string; title: string }) => void;
}) {
  const toast = useToast();
  const [open, setOpen] = React.useState(false);
  const [added, setAdded] = React.useState<{ id: string; title: string } | null>(null);
  const [recent, setRecent] = React.useState<{ id: string; title: string }[]>([]);
  const ref = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const add = (target: { notebookId: string } | { newTitle?: string }) => {
    const nb = addTurnsToNotebook([turn], target, {
      conversationId,
      title: conversationTitle,
    });
    setAdded({ id: nb.id, title: nb.title });
    onAdded({ id: nb.id, title: nb.title });
    setOpen(false);
    track("chat_turn_added_to_notebook", {
      new_notebook: !("notebookId" in target),
    });
  };

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(turn.assistantText);
      toast.show("Answer copied", "success");
    } catch {
      toast.show("Could not copy to the clipboard", "error");
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-2">
      <div ref={ref} className="relative">
        <div className="inline-flex overflow-hidden rounded-md border border-hairline bg-white text-xs shadow-sm">
          <button
            type="button"
            disabled={added !== null && added.id === linked?.id}
            onClick={() =>
              add(linked ? { notebookId: linked.id } : { newTitle: conversationTitle })
            }
            title={linked ? `Append to “${linked.title}”` : "Start a notebook from this answer"}
            className="inline-flex max-w-[16rem] items-center gap-1.5 px-2.5 py-1 text-ink hover:bg-surface-soft disabled:cursor-default disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-navy"
          >
            <BookPlus className="h-3.5 w-3.5 shrink-0 text-navy" />
            <span className="truncate">
              {linked ? `Add to ${linked.title}` : "Add to notebook"}
            </span>
          </button>
          <button
            type="button"
            aria-label="Choose a notebook"
            aria-expanded={open}
            onClick={() => {
              if (!open) setRecent(notebooks.list().slice(0, 6));
              setOpen(!open);
            }}
            className="border-l border-hairline px-1.5 text-muted hover:bg-surface-soft hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-navy"
          >
            <ChevronDown className="h-3.5 w-3.5" />
          </button>
        </div>
        {open && (
          <div className="absolute left-0 z-20 mt-1 w-72 rounded-lg border border-hairline bg-white p-1 shadow-lg">
            <button
              type="button"
              onClick={() => add({ newTitle: conversationTitle })}
              className="flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-left text-sm text-ink hover:bg-surface-soft"
            >
              <FilePlus2 className="h-4 w-4 text-navy" />
              New notebook
            </button>
            {recent.length > 0 && (
              <>
                <p className="px-2.5 pb-1 pt-2 text-[10px] font-semibold uppercase tracking-wider text-muted">
                  Add to existing
                </p>
                <ul>
                  {recent.map((n) => (
                    <li key={n.id}>
                      <button
                        type="button"
                        onClick={() => add({ notebookId: n.id })}
                        className="w-full truncate rounded-md px-2.5 py-1.5 text-left text-sm text-body hover:bg-surface-soft hover:text-ink"
                      >
                        {n.title || "Untitled notebook"}
                      </button>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </div>
        )}
      </div>
      <button
        type="button"
        onClick={() => void copy()}
        className="inline-flex items-center gap-1.5 rounded-md border border-hairline bg-white px-2.5 py-1 text-xs text-muted shadow-sm hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy"
      >
        <Copy className="h-3.5 w-3.5" />
        Copy
      </button>
      {added && (
        <span
          className={cn(
            "inline-flex animate-rise items-center gap-1.5 text-xs text-muted",
          )}
        >
          <Check className="h-3.5 w-3.5 text-success" />
          Added to{" "}
          <Link
            href={`/notebook/${added.id}`}
            className="font-medium text-navy underline decoration-coral/40 underline-offset-2 hover:decoration-coral"
          >
            {added.title}
          </Link>
        </span>
      )}
    </div>
  );
}
