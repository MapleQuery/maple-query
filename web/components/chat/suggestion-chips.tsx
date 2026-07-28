"use client";

import * as React from "react";
import type { SuggestionT } from "@/lib/types";

export interface SuggestionChipsProps {
  items: SuggestionT[];
  /** Disabled while any turn is streaming — an offer accepted mid-turn
   * would abort the answer the user is still reading. */
  disabled?: boolean;
  onAccept: (suggestion: SuggestionT) => void;
}

/**
 * Next-step offers under the latest assistant message.
 *
 * Deliberately subordinate to the answer: bordered and surface-toned
 * rather than coral, because coral is this app's primary action (the
 * composer's send) and an offer must not compete with the content it
 * sits beneath.
 *
 * Nothing here fires on its own. No chip runs on render, on a timer, or
 * on the answer completing — each accepted offer is a priced turn, and
 * the click is the user's cost decision to make.
 */
export function SuggestionChips({
  items,
  disabled = false,
  onAccept,
}: SuggestionChipsProps) {
  if (items.length === 0) return null;

  return (
    <div className="space-y-2">
      {/* Without this the row reads as an error-recovery bar rather
          than as an offer. */}
      <p className="text-xs font-medium uppercase tracking-wide text-muted">
        Next steps
      </p>
      <ul className="flex flex-wrap gap-2">
        {items.map((s, i) => (
          <li key={`${s.kind}-${s.package_ids.join(",")}-${i}`}>
            <button
              type="button"
              disabled={disabled}
              onClick={() => onAccept(s)}
              // `label` is an abbreviation of `question`; a screen-reader
              // user has to know what will actually be sent before
              // activating it.
              aria-label={s.question}
              className="max-w-full rounded-md border border-hairline bg-white px-3 py-1.5 text-left text-sm text-ink transition-colors hover:bg-surface-soft focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy focus-visible:ring-offset-2 focus-visible:ring-offset-canvas disabled:pointer-events-none disabled:opacity-50"
            >
              {s.label}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
