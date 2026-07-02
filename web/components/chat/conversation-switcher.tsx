"use client";

import Link from "next/link";
import { Plus, Trash2 } from "lucide-react";
import { conversations } from "@/lib/storage";
import { cn } from "@/lib/utils";

export interface ConversationSwitcherProps {
  currentId?: string;
  onNew: () => void;
  onDelete: (id: string) => void;
  index: { id: string; title: string; updatedAt: string }[];
}

export function ConversationSwitcher({
  currentId,
  onNew,
  onDelete,
  index,
}: ConversationSwitcherProps) {
  return (
    <aside className="hidden w-64 shrink-0 border-r border-hairline bg-surface-soft/70 lg:flex lg:flex-col">
      <div className="border-b border-hairline p-3">
        <button
          type="button"
          onClick={onNew}
          className="flex w-full items-center gap-2 rounded-md bg-coral px-3 py-2 text-sm font-medium text-ink transition-colors hover:bg-coral-active focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy"
        >
          <Plus className="h-4 w-4" /> New conversation
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-2">
        <p className="mb-1 px-2 text-[10px] font-semibold uppercase tracking-wider text-muted">
          Recent
        </p>
        {index.length === 0 ? (
          <p className="px-2 py-6 text-center text-xs text-muted">
            No conversations yet. Ask something to get started.
          </p>
        ) : (
          <ul className="space-y-1">
            {index.map((c) => (
              <li key={c.id}>
                <div
                  className={cn(
                    "group flex items-center gap-2 rounded-md px-2 py-1.5 transition-colors",
                    currentId === c.id
                      ? "bg-white shadow-sm"
                      : "hover:bg-white/70",
                  )}
                >
                  <Link
                    href={`/chat/${c.id}`}
                    className="flex min-w-0 flex-1 flex-col text-left"
                  >
                    <span
                      className={cn(
                        "line-clamp-1 text-sm",
                        currentId === c.id
                          ? "font-medium text-ink"
                          : "text-body",
                      )}
                    >
                      {c.title || "Untitled"}
                    </span>
                    <span className="font-mono text-[10px] text-muted">
                      {new Date(c.updatedAt).toLocaleDateString()}
                    </span>
                  </Link>
                  <button
                    type="button"
                    onClick={() => onDelete(c.id)}
                    className="opacity-0 group-hover:opacity-100 focus-visible:opacity-100 rounded p-1 text-muted transition-colors hover:text-error focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy"
                    aria-label="Delete conversation"
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </aside>
  );
}
