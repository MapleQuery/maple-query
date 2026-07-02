"use client";

import { createHighlighter, type Highlighter } from "shiki";

let cache: Promise<Highlighter> | null = null;

async function get(): Promise<Highlighter> {
  if (!cache) {
    cache = createHighlighter({
      themes: ["github-dark-default"],
      langs: ["sql"],
    });
  }
  return cache;
}

export async function highlightSql(sql: string): Promise<string> {
  const hl = await get();
  return hl.codeToHtml(sql, {
    lang: "sql",
    theme: "github-dark-default",
  });
}
