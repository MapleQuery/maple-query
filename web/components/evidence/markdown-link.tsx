"use client";

import * as React from "react";
import Link from "next/link";

const LINK_CLASS =
  "text-navy underline decoration-coral/40 underline-offset-2 hover:decoration-coral";

/** In-app citations route client-side; sources open in a new tab. */
export function MarkdownLink({
  href,
  children,
}: {
  href?: string;
  children?: React.ReactNode;
}) {
  const url = href ?? "";
  if (url.startsWith("/")) {
    return (
      <Link href={url} className={LINK_CLASS}>
        {children}
      </Link>
    );
  }
  return (
    <a href={url} target="_blank" rel="noopener noreferrer" className={LINK_CLASS}>
      {children}
    </a>
  );
}
