"use client";

import * as React from "react";
import { ArrowUp, Square } from "lucide-react";
import { cn } from "@/lib/utils";

export interface ChatComposerProps {
  onSubmit: (text: string) => void;
  onAbort?: () => void;
  disabled?: boolean;
  streaming?: boolean;
  suggestions?: string[];
  placeholder?: string;
}

export function ChatComposer({
  onSubmit,
  onAbort,
  disabled,
  streaming,
  suggestions = [],
  placeholder = "Ask in plain language. MapleQuery will show its work.",
}: ChatComposerProps) {
  const [text, setText] = React.useState("");
  const taRef = React.useRef<HTMLTextAreaElement>(null);

  const autosize = React.useCallback(() => {
    const el = taRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
  }, []);

  React.useEffect(() => {
    autosize();
  }, [text, autosize]);

  const submit = (value: string) => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSubmit(trimmed);
    setText("");
  };

  return (
    <div className="border-t border-hairline bg-canvas/95 px-4 py-4 backdrop-blur md:px-6 lg:px-8">
      {suggestions.length > 0 && !streaming && (
        <div className="mb-3 flex flex-wrap gap-2">
          {suggestions.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => submit(s)}
              className="rounded-full border border-hairline bg-white px-3 py-1.5 text-xs font-medium text-body transition-colors hover:border-navy hover:text-navy focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy"
            >
              {s}
            </button>
          ))}
        </div>
      )}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit(text);
        }}
        className={cn(
          "flex items-end gap-2 rounded-xl border border-hairline bg-white p-2 shadow-sm transition-colors focus-within:ring-2 focus-within:ring-navy",
          disabled && "opacity-60",
        )}
      >
        <label htmlFor="composer-input" className="sr-only">
          Ask a question about Canadian government data
        </label>
        <textarea
          id="composer-input"
          ref={taRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit(text);
            }
          }}
          rows={1}
          disabled={disabled}
          placeholder={placeholder}
          className="max-h-56 flex-1 resize-none bg-transparent px-2 py-2 text-[15px] text-ink placeholder:text-muted focus:outline-none disabled:cursor-not-allowed"
        />
        {streaming ? (
          <button
            type="button"
            onClick={onAbort}
            className="rounded-lg border border-hairline bg-white p-2.5 text-ink transition-colors hover:bg-surface-soft focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy"
            aria-label="Stop"
          >
            <Square className="h-4 w-4" />
          </button>
        ) : (
          <button
            type="submit"
            disabled={disabled || !text.trim()}
            className="rounded-lg bg-coral p-2.5 text-ink transition-colors hover:bg-coral-active focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-40"
            aria-label="Send"
          >
            <ArrowUp className="h-4 w-4" />
          </button>
        )}
      </form>
      <p className="mt-2 px-1 text-xs text-muted">
        MapleQuery only answers from cited datasets. Press{" "}
        <kbd className="rounded border border-hairline bg-white px-1 font-mono text-[10px]">
          Enter
        </kbd>{" "}
        to send,{" "}
        <kbd className="rounded border border-hairline bg-white px-1 font-mono text-[10px]">
          Shift + Enter
        </kbd>{" "}
        for a new line.
      </p>
    </div>
  );
}
