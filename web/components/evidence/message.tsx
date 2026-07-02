"use client";

import * as React from "react";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { LogoMark } from "@/components/ui/logo";
import { MapleLoader } from "@/components/ui/maple-loader";
import { cn } from "@/lib/utils";

export interface MessageProps {
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
  meta?: React.ReactNode;
}

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

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
        </div>
        <div className="prose-body max-w-none text-[15px] leading-relaxed text-body">
          {content ? (
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                code({ className, children, ...props }) {
                  const text = String(children ?? "");
                  const isBlock = className?.startsWith("language-");
                  if (!isBlock && UUID_RE.test(text.trim())) {
                    return (
                      <Link
                        href={`/datasets/${text.trim()}`}
                        className="rounded bg-coral/10 px-1.5 py-0.5 font-mono text-[0.92em] text-navy hover:bg-coral/20 hover:underline"
                      >
                        {text}
                      </Link>
                    );
                  }
                  return (
                    <code className={className} {...props}>
                      {children}
                    </code>
                  );
                },
              }}
            >
              {content}
            </ReactMarkdown>
          ) : streaming ? (
            <span className="inline-flex items-center gap-2.5 rounded-2xl rounded-tl-md border border-hairline bg-white px-4 py-3">
              <MapleLoader size={20} layout="row" />
            </span>
          ) : (
            <span className="text-muted">…</span>
          )}
          {content && streaming && <StreamingCursor />}
        </div>
        {meta && <div className="mt-2">{meta}</div>}
      </div>
    </div>
  );
}

function StreamingCursor() {
  return (
    <span
      aria-hidden="true"
      className="ml-0.5 inline-block h-4 w-[2px] animate-dot-blink bg-coral align-middle"
    />
  );
}
