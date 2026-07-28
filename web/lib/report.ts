/**
 * The notebook as a document, independent of what it will be written to.
 *
 * PDF and Word are both *built*, not printed — a page laid out by a
 * generator rather than a browser screenshot of the app. They want the
 * same things (a heading is a heading, a table has columns, a chart is an
 * image) and they want them in different APIs, so the traversal happens
 * once here and each writer only translates.
 *
 * That split is the point. The previous PDF was the Markdown export fed
 * through the browser's print dialog, which meant the output carried the
 * browser's own header and footer, paginated like a web page, and — the
 * failure that made it unusable — printed before its images had decoded,
 * so every chart came out as a broken-image icon. None of those are
 * fixable inside a print stylesheet. They are all consequences of not
 * building the document.
 */

import {
  chartData,
  renderChartSvg,
  resolveChartSpec,
} from "./chart";
import { chartSource, exportableBlocks } from "./notebook";
import type { DatasetTitleMap } from "./dataset-titles";
import type { StoredNotebook } from "./storage";

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

export interface ReportRun {
  text: string;
  bold?: boolean;
  italic?: boolean;
  mono?: boolean;
  /** `#rrggbb`. */
  color?: string;
  /** Multiplier on the body size, from the prose vocabulary. */
  scale?: number;
}

export type ReportNode =
  | { kind: "title"; text: string }
  | { kind: "heading"; level: 1 | 2 | 3; runs: ReportRun[] }
  | { kind: "paragraph"; runs: ReportRun[] }
  | { kind: "bullet"; runs: ReportRun[] }
  | { kind: "code"; lines: string[] }
  | { kind: "note"; text: string }
  | {
      kind: "table";
      columns: string[];
      rows: string[][];
      /** e.g. "Showing first 20 of 103 rows." */
      note?: string;
    }
  | { kind: "chart"; svg: string; width: number; height: number };

export interface Report {
  title: string;
  nodes: ReportNode[];
}

// ---------------------------------------------------------------------------
// Inline parsing
// ---------------------------------------------------------------------------

const SPAN_RE = /<span style="([^"]*)">([\s\S]*?)<\/span>/g;
const COLOR_DECL_RE = /color:\s*(#[0-9a-f]{6})/i;
const SIZE_DECL_RE = /font-size:\s*(\d+(?:\.\d+)?)em/i;

/**
 * Markdown inline → runs.
 *
 * Deliberately small: bold, italics, inline code, and the styled span
 * `lib/prose.ts` writes. Anything else stays literal text, which is the
 * honest outcome for a converter that does not claim to be a Markdown
 * engine — a half-supported construct rendering as mangled text is worse
 * than one rendering as itself.
 */
export function parseInline(source: string): ReportRun[] {
  const runs: ReportRun[] = [];

  const pushStyled = (text: string, base: Partial<ReportRun>): void => {
    // Bold before italics: `**` must not be read as two `*`.
    const pattern = /(\*\*|__)(.+?)\1|(\*|_)(.+?)\3|`([^`]+)`/g;
    let last = 0;
    let m: RegExpExecArray | null;
    while ((m = pattern.exec(text)) !== null) {
      if (m.index > last) {
        runs.push({ ...base, text: text.slice(last, m.index) });
      }
      if (m[2] !== undefined) {
        runs.push({ ...base, text: m[2], bold: true });
      } else if (m[4] !== undefined) {
        runs.push({ ...base, text: m[4], italic: true });
      } else if (m[5] !== undefined) {
        runs.push({ ...base, text: m[5], mono: true });
      }
      last = m.index + m[0].length;
    }
    if (last < text.length) runs.push({ ...base, text: text.slice(last) });
  };

  let cursor = 0;
  let span: RegExpExecArray | null;
  SPAN_RE.lastIndex = 0;
  while ((span = SPAN_RE.exec(source)) !== null) {
    if (span.index > cursor) pushStyled(source.slice(cursor, span.index), {});
    const base: Partial<ReportRun> = {};
    const color = COLOR_DECL_RE.exec(span[1]);
    if (color) base.color = color[1];
    const size = SIZE_DECL_RE.exec(span[1]);
    if (size) base.scale = Number(size[1]);
    pushStyled(span[2], base);
    cursor = span.index + span[0].length;
  }
  if (cursor < source.length) pushStyled(source.slice(cursor), {});

  return runs.filter((r) => r.text !== "");
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

const SVG_SIZE_RE = /width="(\d+)"\s+height="(\d+)"/;

function cellText(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return value.toLocaleString("en-CA");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

/**
 * Walk the notebook into a document.
 *
 * Hidden blocks are already gone by the time anything is emitted, and no
 * export carries an "exported at" line: a timestamp is metadata about the
 * act of exporting, not part of the piece, and it dates a document that
 * is about to be edited anyway.
 */
export function buildReport(
  nb: StoredNotebook,
  titles?: DatasetTitleMap,
): Report {
  const nodes: ReportNode[] = [];
  const push = (n: ReportNode) => nodes.push(n);

  for (const b of exportableBlocks(nb)) {
    if (b.type === "prose") {
      for (const line of b.markdown.split("\n")) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        const heading = /^(#{1,6})\s+(.*)$/.exec(trimmed);
        if (heading) {
          const level = Math.min(3, heading[1].length) as 1 | 2 | 3;
          push({ kind: "heading", level, runs: parseInline(heading[2]) });
          continue;
        }
        const bullet = /^[-*]\s+(.*)$/.exec(trimmed);
        if (bullet) {
          push({ kind: "bullet", runs: parseInline(bullet[1]) });
          continue;
        }
        push({ kind: "paragraph", runs: parseInline(trimmed) });
      }
      continue;
    }

    if (b.type === "chart") {
      const source = chartSource(nb.blocks, b.sourceBlockId);
      const rows = source?.result?.rows ?? [];
      const spec = resolveChartSpec(rows, b.overrides);
      if (!spec) continue;
      const svg = renderChartSvg(chartData(rows, spec), spec, {
        valueLabel: spec.valueColumn,
      });
      const size = svg ? SVG_SIZE_RE.exec(svg) : null;
      if (svg && size) {
        push({
          kind: "chart",
          svg,
          width: Number(size[1]),
          height: Number(size[2]),
        });
      }
      continue;
    }

    const name = (id: string): string =>
      b.result?.packageTitles?.[id] ?? titles?.[id] ?? id;

    push({
      kind: "heading",
      level: 2,
      runs: parseInline(b.question || "(empty query)"),
    });
    if (b.scopePackageIds && b.scopePackageIds.length > 0) {
      push({
        kind: "note",
        text: `Scoped to ${b.scopePackageIds.map(name).join(", ")}`,
      });
    }
    if (b.result?.assistantText) {
      for (const line of b.result.assistantText.trim().split("\n")) {
        if (line.trim()) push({ kind: "paragraph", runs: parseInline(line) });
      }
    }
    if (b.result?.sql) {
      push({ kind: "code", lines: b.result.sql.trim().split("\n") });
    }
    if (b.result?.rows && b.result.rows.length > 0) {
      const rows = b.result.rows.slice(0, 20);
      const columns = Object.keys(rows[0] ?? {});
      push({
        kind: "table",
        columns,
        rows: rows.map((row) => columns.map((c) => cellText(row[c]))),
        note:
          b.result.rows.length > rows.length
            ? `Showing first ${rows.length} of ${b.result.rows.length} rows.`
            : undefined,
      });
    }
    if (b.result?.packageIds && b.result.packageIds.length > 0) {
      push({
        kind: "note",
        text: `Sources: ${b.result.packageIds.map(name).join(", ")}`,
      });
    }
  }

  return { title: nb.title || "Untitled notebook", nodes };
}

// ---------------------------------------------------------------------------
// Shared typography
// ---------------------------------------------------------------------------

/**
 * One type scale for both writers, in points, so a Word document and a
 * PDF of the same notebook are recognisably the same piece.
 */
export const REPORT_TYPE = {
  body: 11,
  title: 24,
  h1: 17,
  h2: 14,
  h3: 12,
  code: 9,
  note: 9,
} as const;

export const REPORT_COLOR = {
  ink: "#13181B",
  body: "#2D3942",
  muted: "#5C6B75",
  hairline: "#D3DADD",
  soft: "#F4F6F8",
} as const;

/** `docx` wants `RRGGBB`; the model and pdfmake both use `#rrggbb`. */
export function bareHex(hex: string): string {
  return hex.replace("#", "").toUpperCase();
}

/**
 * Rasterise a chart SVG to PNG bytes.
 *
 * Both writers embed raster images: `docx` accepts SVG only alongside a
 * fallback and Word's own support is uneven, and pdfmake takes data URIs.
 * The chart SVG is self-contained — no external fonts or images — which
 * is what makes it safe to draw through a canvas without tainting it.
 *
 * Waits for `decode()` rather than `onload`, because a decoded image is
 * the actual precondition for drawing one. Getting that wrong is what
 * put broken-image icons in the printed PDF.
 */
export async function rasterizeSvg(
  svg: string,
  width: number,
  height: number,
  scale = 2,
): Promise<{ dataUri: string; bytes: Uint8Array } | null> {
  const blob = new Blob([svg], { type: "image/svg+xml" });
  const url = URL.createObjectURL(blob);
  try {
    const image = new Image();
    image.width = width;
    image.height = height;
    image.src = url;
    await image.decode();

    const canvas = document.createElement("canvas");
    canvas.width = width * scale;
    canvas.height = height * scale;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

    const dataUri = canvas.toDataURL("image/png");
    const base64 = dataUri.slice(dataUri.indexOf(",") + 1);
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return { dataUri, bytes };
  } catch {
    // A chart that will not rasterise must not take the export with it.
    return null;
  } finally {
    URL.revokeObjectURL(url);
  }
}
