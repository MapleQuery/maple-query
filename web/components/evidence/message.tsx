"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { LogoMark } from "@/components/ui/logo";
import { cn } from "@/lib/utils";

export interface MessageProps {
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
  meta?: React.ReactNode;
}

export function Message({ role, content, streaming, meta }: MessageProps) {
  if (role === "user") {
    return (
      <div className="flex animate-rise justify-end">
        <div className="max-w-[80%] rounded-2xl rounded-br-md border border-hairline bg-white px-4 py-2.5 text-[15px] leading-relaxed text-ink shadow-sm">
          {content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex animate-rise gap-3">
      <span
        aria-hidden="true"
        className="mt-1 grid h-8 w-8 shrink-0 place-items-center rounded-full bg-coral/15 text-navy"
      >
        <LogoMark className="h-6 w-6" />
      </span>
      <div className="min-w-0 flex-1">
        <div className="mb-1.5 flex items-center gap-2">
          <span className="text-sm font-semibold text-ink">MapleQuery</span>
          <span className="rounded bg-coral/10 px-1.5 py-0.5 font-mono text-[10px] font-medium uppercase tracking-wider text-navy">
            AI
          </span>
        </div>
        <div
          className={cn(
            "prose-body max-w-none text-[15px] leading-relaxed text-body",
            streaming && "after:ml-0.5 after:inline-block after:h-4 after:w-[2px] after:animate-dot-blink after:bg-coral after:align-middle",
          )}
        >
          {content ? (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
          ) : streaming ? (
            <TypingDots />
          ) : (
            <span className="text-muted">…</span>
          )}
        </div>
        {meta && <div className="mt-2">{meta}</div>}
      </div>
    </div>
  );
}

function TypingDots() {
  return (
    <span className="inline-flex items-center gap-2.5 rounded-2xl rounded-tl-md border border-hairline bg-white px-4 py-3">
      <span className="font-mono text-[10px] font-medium uppercase tracking-[0.14em] text-navy">
        Thinking
      </span>
      <span className="flex gap-1">
        <span className="h-1.5 w-1.5 animate-dot-blink rounded-full bg-coral" />
        <span
          className="h-1.5 w-1.5 animate-dot-blink rounded-full bg-coral"
          style={{ animationDelay: "0.2s" }}
        />
        <span
          className="h-1.5 w-1.5 animate-dot-blink rounded-full bg-coral"
          style={{ animationDelay: "0.4s" }}
        />
      </span>
    </span>
  );
}
