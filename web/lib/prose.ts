/**
 * The prose vocabulary: what a text block may contain, and the one place
 * that decides how it renders.
 *
 * Markdown carries headings, bold and italics natively. It has no
 * spelling for size or colour, so those ride on a single inline element
 * — `<span style="color:…">`, `<span style="font-size:…">` — which is
 * ordinary Markdown-with-inline-HTML rather than a private syntax. That
 * choice is what lets one stored string render on screen, through the
 * print pipeline, and into Word without three dialects of formatting.
 *
 * Raw HTML in a document the user typed is a real hazard, so it passes
 * two independent filters. `rehype-sanitize` runs first against an
 * allowlist that admits `span[style]` and nothing else new — every
 * default block on scripts, event handlers and embedded frames stays.
 * Then the `span` renderer below re-emits only `color` and `font-size`,
 * and only when the value matches the patterns here. A declaration that
 * clears the allowlist but fails the pattern is dropped rather than
 * trusted, so neither filter is load-bearing on its own.
 *
 * Colours are a fixed palette rather than a picker. A report stays
 * coherent when its emphasis comes from a small set, the values are
 * already the app's own tokens, and a closed set is what makes the DOCX
 * mapping exact instead of approximate.
 */

import type { Components, Options } from "react-markdown";
import rehypeRaw from "rehype-raw";
import rehypeSanitize, { defaultSchema } from "rehype-sanitize";
import remarkGfm from "remark-gfm";
import * as React from "react";

export interface ProseColor {
  /** Stored in the document, and the DOCX mapping key. */
  hex: string;
  label: string;
}

/** Design tokens, not free colour. `ink` is the default and is written
 * as no span at all, so unstyled text stays unstyled in the source. */
export const PROSE_COLORS: ProseColor[] = [
  { hex: "#13181B", label: "Ink" },
  { hex: "#5C6B75", label: "Muted" },
  { hex: "#003A6C", label: "Navy" },
  { hex: "#ED7259", label: "Coral" },
  { hex: "#2F7D5B", label: "Green" },
  { hex: "#C2503B", label: "Red" },
];

export interface ProseSize {
  /** `em`, so it scales with the surrounding context — including the
   * print stylesheet, whose base is points rather than pixels. */
  value: string;
  label: string;
}

export const PROSE_SIZES: ProseSize[] = [
  { value: "0.85em", label: "Small" },
  { value: "1em", label: "Normal" },
  { value: "1.25em", label: "Large" },
  { value: "1.5em", label: "Huge" },
];

const COLOR_RE = /^#[0-9a-f]{6}$/i;
const SIZE_RE = /^\d+(\.\d+)?em$/;

/** The sanitize allowlist: defaults, plus `span` carrying `style`. */
export const proseSanitizeSchema = {
  ...defaultSchema,
  tagNames: [...(defaultSchema.tagNames ?? []), "span"],
  attributes: {
    ...defaultSchema.attributes,
    span: [...(defaultSchema.attributes?.span ?? []), "style"],
  },
};

export const proseRemarkPlugins: Options["remarkPlugins"] = [remarkGfm];
/** Order matters: parse the raw HTML, then sanitize what parsing found. */
export const proseRehypePlugins: Options["rehypePlugins"] = [
  rehypeRaw,
  [rehypeSanitize, proseSanitizeSchema],
];

/**
 * Re-emit a span's style from scratch, keeping only the two properties
 * this vocabulary defines and only when their values parse.
 */
export function safeSpanStyle(
  style: React.CSSProperties | undefined,
): React.CSSProperties | undefined {
  if (!style) return undefined;
  const out: React.CSSProperties = {};
  const color = typeof style.color === "string" ? style.color.trim() : "";
  if (COLOR_RE.test(color)) out.color = color;
  const size =
    typeof style.fontSize === "string" ? style.fontSize.trim() : "";
  if (SIZE_RE.test(size)) out.fontSize = size;
  return Object.keys(out).length > 0 ? out : undefined;
}

export const proseComponents: Components = {
  span({ style, children, ...rest }) {
    const safe = safeSpanStyle(style as React.CSSProperties | undefined);
    return React.createElement("span", { ...rest, style: safe }, children);
  },
};

// ---------------------------------------------------------------------------
// Editing helpers — pure string transforms over the Markdown source, so
// the toolbar has no opinion about how the text is stored.
// ---------------------------------------------------------------------------

export interface Selection {
  start: number;
  end: number;
}

export interface EditResult {
  markdown: string;
  /** Where the caret/selection should land afterwards. */
  selection: Selection;
}

/** Wrap the selection in `marker`, or unwrap it when already wrapped. */
export function toggleWrap(
  markdown: string,
  sel: Selection,
  marker: string,
): EditResult {
  const before = markdown.slice(0, sel.start);
  const selected = markdown.slice(sel.start, sel.end);
  const after = markdown.slice(sel.end);

  if (before.endsWith(marker) && after.startsWith(marker)) {
    return {
      markdown:
        before.slice(0, -marker.length) + selected + after.slice(marker.length),
      selection: {
        start: sel.start - marker.length,
        end: sel.end - marker.length,
      },
    };
  }
  if (
    selected.length >= marker.length * 2 &&
    selected.startsWith(marker) &&
    selected.endsWith(marker)
  ) {
    const inner = selected.slice(marker.length, -marker.length);
    return {
      markdown: before + inner + after,
      selection: { start: sel.start, end: sel.start + inner.length },
    };
  }
  return {
    markdown: `${before}${marker}${selected}${marker}${after}`,
    selection: {
      start: sel.start + marker.length,
      end: sel.end + marker.length,
    },
  };
}

/**
 * Set the heading level of every line the selection touches. Level 0
 * strips the heading. Applied per line so selecting a paragraph and a
 * heading together produces a consistent result rather than a nested one.
 */
export function setHeading(
  markdown: string,
  sel: Selection,
  level: number,
): EditResult {
  const lineStart = markdown.lastIndexOf("\n", Math.max(0, sel.start - 1)) + 1;
  const lineEndIdx = markdown.indexOf("\n", sel.end);
  const lineEnd = lineEndIdx === -1 ? markdown.length : lineEndIdx;

  const block = markdown.slice(lineStart, lineEnd);
  const rewritten = block
    .split("\n")
    .map((line) => {
      const bare = line.replace(/^#{1,6}\s+/, "");
      return level > 0 ? `${"#".repeat(level)} ${bare}` : bare;
    })
    .join("\n");

  const markdownNext =
    markdown.slice(0, lineStart) + rewritten + markdown.slice(lineEnd);
  const delta = rewritten.length - block.length;
  return {
    markdown: markdownNext,
    selection: { start: sel.start, end: sel.end + delta },
  };
}

/**
 * Wrap the selection in a styled span, or restyle one it is already
 * inside. Passing neither colour nor size removes the span, so the
 * control that applied the style is also the one that clears it.
 */
export function applySpanStyle(
  markdown: string,
  sel: Selection,
  style: { color?: string; size?: string },
): EditResult {
  const selected = markdown.slice(sel.start, sel.end);
  if (!selected) return { markdown, selection: sel };

  // Restyling: if the selection is exactly an existing span, replace it
  // rather than nesting a second one inside the first.
  const existing = /^<span style="[^"]*">([\s\S]*)<\/span>$/.exec(selected);
  const inner = existing ? existing[1] : selected;

  const declarations: string[] = [];
  if (style.color && COLOR_RE.test(style.color)) {
    declarations.push(`color:${style.color}`);
  }
  if (style.size && SIZE_RE.test(style.size) && style.size !== "1em") {
    declarations.push(`font-size:${style.size}`);
  }

  const replacement =
    declarations.length === 0
      ? inner
      : `<span style="${declarations.join(";")}">${inner}</span>`;

  return {
    markdown:
      markdown.slice(0, sel.start) + replacement + markdown.slice(sel.end),
    selection: { start: sel.start, end: sel.start + replacement.length },
  };
}
