"use client";

import * as React from "react";
import Link from "next/link";
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
            <MapleLoader />
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

const CANADIAN_VERBS = [
  "Portaging",
  "Zambonieing",
  "Pondering",
  "Snowshoeing",
  "Tobogganing",
  "Percolating",
  "Deliberating",
  "Ruminating",
  "Sledding",
  "Musing",
  "Chinwagging",
  "Sugarshacking",
];

function MapleLoader() {
  const [verbIdx, setVerbIdx] = React.useState(() =>
    Math.floor(Math.random() * CANADIAN_VERBS.length),
  );

  React.useEffect(() => {
    const id = window.setInterval(() => {
      setVerbIdx((i) => (i + 1) % CANADIAN_VERBS.length);
    }, 1600);
    return () => window.clearInterval(id);
  }, []);

  return (
    <span className="inline-flex items-center gap-2.5 rounded-2xl rounded-tl-md border border-hairline bg-white px-4 py-3">
      <MapleLeafSpinner />
      <span className="font-mono text-[10px] font-medium uppercase tracking-[0.14em] text-navy">
        {CANADIAN_VERBS[verbIdx]}
      </span>
    </span>
  );
}

function MapleLeafSpinner() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 100 100"
      className="h-4 w-4 origin-center animate-leaf-pulse text-coral"
      fill="currentColor"
    >
      <path d="M50 6 L54 26 L58 22 L64 30 L74 24 L70 38 L86 36 L78 46 L94 52 L74 60 L78 68 L64 66 L66 78 L58 74 L54 82 L50 74 L46 82 L42 74 L34 78 L36 66 L22 68 L26 60 L6 52 L22 46 L14 36 L30 38 L26 24 L36 30 L42 22 L46 26 Z" />
    </svg>
  );
}
