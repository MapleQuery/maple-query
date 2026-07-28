/**
 * The notebook as a Word document.
 *
 * Journalists work in Word, and a PDF is where editing stops. This is
 * the one export that produces a document someone can keep writing in.
 *
 * It builds from the same `StoredNotebook` the other two exports read,
 * not from their Markdown. Markdown is a lossy target for this: a `.docx`
 * wants real heading styles, real table cells and embedded image bytes,
 * and re-parsing a rendered string to recover structure we already have
 * would be a second, worse document model. The *inline* vocabulary is
 * still shared — `lib/prose.ts` defines the same bold / italic / size /
 * colour set the screen and the PDF use, and `parseInline` below is the
 * one place it becomes Word runs.
 *
 * `docx` is imported dynamically. It is a few hundred kilobytes of OOXML
 * writer that only matters the moment someone picks this format, and
 * every other page in the app would otherwise pay for it.
 */

import type {
  IParagraphOptions,
  IRunOptions,
  Paragraph as ParagraphT,
  Table as TableT,
} from "docx";
import {
  chartData,
  renderChartSvg,
  resolveChartSpec,
  type ChartSpec,
} from "@/lib/chart";
import { chartSource, exportableBlocks } from "@/lib/notebook";
import type { DatasetTitleMap } from "@/lib/dataset-titles";
import type { StoredNotebook } from "@/lib/storage";

/** Word's default body size is 22 half-points (11pt); `em` values from
 * the prose vocabulary scale against that. */
const BODY_HALF_POINTS = 22;

interface InlineRun {
  text: string;
  bold?: boolean;
  italics?: boolean;
  /** `RRGGBB`, no leading hash — what `docx` expects. */
  color?: string;
  /** Half-points. */
  size?: number;
}

const SPAN_RE = /<span style="([^"]*)">([\s\S]*?)<\/span>/g;
const COLOR_DECL_RE = /color:\s*(#[0-9a-f]{6})/i;
const SIZE_DECL_RE = /font-size:\s*(\d+(?:\.\d+)?)em/i;

/**
 * Markdown inline → Word runs.
 *
 * Deliberately small: bold, italics, inline code, and the styled span
 * this app writes. Anything else stays literal text, which is the honest
 * outcome for a converter that does not claim to be a Markdown engine —
 * a half-supported construct that renders as mangled text would be worse
 * than one that renders as itself.
 */
export function parseInline(source: string): InlineRun[] {
  const runs: InlineRun[] = [];

  const pushStyled = (text: string, base: Partial<InlineRun>): void => {
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
        runs.push({ ...base, text: m[4], italics: true });
      } else if (m[5] !== undefined) {
        runs.push({ ...base, text: m[5] });
      }
      last = m.index + m[0].length;
    }
    if (last < text.length) runs.push({ ...base, text: text.slice(last) });
  };

  let cursor = 0;
  let span: RegExpExecArray | null;
  SPAN_RE.lastIndex = 0;
  while ((span = SPAN_RE.exec(source)) !== null) {
    if (span.index > cursor) {
      pushStyled(source.slice(cursor, span.index), {});
    }
    const declarations = span[1];
    const base: Partial<InlineRun> = {};
    const color = COLOR_DECL_RE.exec(declarations);
    if (color) base.color = color[1].slice(1).toUpperCase();
    const size = SIZE_DECL_RE.exec(declarations);
    if (size) {
      base.size = Math.round(BODY_HALF_POINTS * Number(size[1]));
    }
    pushStyled(span[2], base);
    cursor = span.index + span[0].length;
  }
  if (cursor < source.length) pushStyled(source.slice(cursor), {});

  return runs.filter((r) => r.text !== "");
}

/**
 * Rasterise our SVG to PNG bytes.
 *
 * `docx` can embed SVG only alongside a raster fallback, and Word's own
 * SVG support is uneven across versions, so the chart goes in as PNG.
 * The SVG is self-contained — no external fonts or images — which is
 * what makes it safe to draw through a canvas without tainting it.
 *
 * At 2× for a chart that stays legible when the reader zooms.
 */
async function svgToPng(
  svg: string,
  scale = 2,
): Promise<{ data: Uint8Array; width: number; height: number } | null> {
  const sized = /width="(\d+)"\s+height="(\d+)"/.exec(svg);
  if (!sized) return null;
  const width = Number(sized[1]);
  const height = Number(sized[2]);

  const blob = new Blob([svg], { type: "image/svg+xml" });
  const url = URL.createObjectURL(blob);
  try {
    const image = new Image();
    image.width = width;
    image.height = height;
    await new Promise<void>((resolve, reject) => {
      image.onload = () => resolve();
      image.onerror = () => reject(new Error("svg_decode_failed"));
      image.src = url;
    });
    const canvas = document.createElement("canvas");
    canvas.width = width * scale;
    canvas.height = height * scale;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    const png = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, "image/png"),
    );
    if (!png) return null;
    return {
      data: new Uint8Array(await png.arrayBuffer()),
      width,
      height,
    };
  } catch {
    // A chart that will not rasterise must not take the export with it;
    // the caller carries on and the document is short one image.
    return null;
  } finally {
    URL.revokeObjectURL(url);
  }
}

/**
 * Build the `.docx` and hand back a Blob.
 *
 * `titles` resolves package ids to names, exactly as the Markdown export
 * does — a reader of the finished document cannot look a UUID up.
 */
export async function exportNotebookAsDocx(
  nb: StoredNotebook,
  titles?: DatasetTitleMap,
): Promise<Blob> {
  const {
    AlignmentType,
    Document,
    HeadingLevel,
    ImageRun,
    Packer,
    Paragraph,
    Table,
    TableCell,
    TableRow,
    TextRun,
    WidthType,
  } = await import("docx");

  const HEADINGS = [
    HeadingLevel.HEADING_1,
    HeadingLevel.HEADING_2,
    HeadingLevel.HEADING_3,
    HeadingLevel.HEADING_4,
    HeadingLevel.HEADING_5,
    HeadingLevel.HEADING_6,
  ];

  const runs = (source: string, extra: IRunOptions = {}): InstanceType<
    typeof TextRun
  >[] =>
    parseInline(source).map(
      (r) =>
        new TextRun({
          ...extra,
          text: r.text,
          bold: r.bold ?? (extra.bold as boolean | undefined),
          italics: r.italics ?? (extra.italics as boolean | undefined),
          ...(r.color ? { color: r.color } : {}),
          ...(r.size ? { size: r.size } : {}),
        }),
    );

  const para = (
    source: string,
    options: IParagraphOptions = {},
  ): ParagraphT =>
    new Paragraph({ ...options, children: runs(source) });

  const children: (ParagraphT | TableT)[] = [];
  const push = (node: ParagraphT | TableT) => children.push(node);

  push(
    new Paragraph({
      text: nb.title || "Untitled notebook",
      heading: HeadingLevel.TITLE,
    }),
  );
  const blocks = exportableBlocks(nb);
  push(
    new Paragraph({
      children: [
        new TextRun({
          text: `Exported ${new Date().toLocaleString()} · ${blocks.length} block${blocks.length === 1 ? "" : "s"}`,
          italics: true,
          color: "5C6B75",
        }),
      ],
    }),
  );

  for (const b of blocks) {
    if (b.type === "prose") {
      // Markdown is line-oriented; a `#` prefix becomes a real Word
      // heading style rather than a bigger run, so the document keeps a
      // navigable outline.
      for (const line of b.markdown.split("\n")) {
        const heading = /^(#{1,6})\s+(.*)$/.exec(line);
        if (heading) {
          push(
            new Paragraph({
              heading: HEADINGS[heading[1].length - 1],
              children: runs(heading[2]),
            }),
          );
          continue;
        }
        const bullet = /^\s*[-*]\s+(.*)$/.exec(line);
        if (bullet) {
          push(
            new Paragraph({ children: runs(bullet[1]), bullet: { level: 0 } }),
          );
          continue;
        }
        push(para(line));
      }
      continue;
    }

    if (b.type === "chart") {
      const source = chartSource(nb.blocks, b.sourceBlockId);
      const rows = source?.result?.rows ?? [];
      const spec: ChartSpec | null = resolveChartSpec(rows, b.overrides);
      if (!spec) continue;
      const svg = renderChartSvg(chartData(rows, spec), spec, {
        valueLabel: spec.valueColumn,
      });
      const png = svg ? await svgToPng(svg) : null;
      if (png) {
        push(
          new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [
              new ImageRun({
                type: "png",
                data: png.data,
                transformation: { width: png.width, height: png.height },
              }),
            ],
          }),
        );
      }
      continue;
    }

    const name = (id: string): string =>
      b.result?.packageTitles?.[id] ?? titles?.[id] ?? id;

    push(
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: runs(b.question || "(empty query)"),
      }),
    );
    if (b.scopePackageIds && b.scopePackageIds.length > 0) {
      push(
        new Paragraph({
          children: [
            new TextRun({
              text: `Scoped to: ${b.scopePackageIds.map(name).join(", ")}`,
              italics: true,
              color: "5C6B75",
            }),
          ],
        }),
      );
    }
    if (b.result?.assistantText) {
      for (const line of b.result.assistantText.trim().split("\n")) {
        push(para(line));
      }
    }
    if (b.result?.sql) {
      // Monospaced and grey rather than a code style Word may not have.
      for (const line of b.result.sql.trim().split("\n")) {
        push(
          new Paragraph({
            children: [
              new TextRun({
                text: line,
                font: "Consolas",
                size: 18,
                color: "2D3942",
              }),
            ],
          }),
        );
      }
    }
    if (b.result?.rows && b.result.rows.length > 0) {
      const rows = b.result.rows.slice(0, 20);
      const columns = Object.keys(rows[0] ?? {});
      push(
        new Table({
          width: { size: 100, type: WidthType.PERCENTAGE },
          rows: [
            new TableRow({
              tableHeader: true,
              children: columns.map(
                (c) =>
                  new TableCell({
                    children: [
                      new Paragraph({
                        children: [new TextRun({ text: c, bold: true })],
                      }),
                    ],
                  }),
              ),
            }),
            ...rows.map(
              (row) =>
                new TableRow({
                  children: columns.map(
                    (c) =>
                      new TableCell({
                        children: [
                          new Paragraph({
                            children: [
                              new TextRun({ text: cellText(row[c]) }),
                            ],
                          }),
                        ],
                      }),
                  ),
                }),
            ),
          ],
        }),
      );
      if (b.result.rows.length > rows.length) {
        push(
          new Paragraph({
            children: [
              new TextRun({
                text: `Showing first ${rows.length} of ${b.result.rows.length} rows.`,
                italics: true,
                color: "5C6B75",
                size: 18,
              }),
            ],
          }),
        );
      }
    }
    if (b.result?.packageIds && b.result.packageIds.length > 0) {
      push(
        new Paragraph({
          children: [
            new TextRun({
              text: `Sources: ${b.result.packageIds.map(name).join(", ")}`,
              italics: true,
              color: "5C6B75",
            }),
          ],
        }),
      );
    }
  }

  const doc = new Document({
    creator: "MapleQuery",
    title: nb.title || "Untitled notebook",
    sections: [{ children }],
  });
  return Packer.toBlob(doc);
}

function cellText(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return value.toLocaleString("en-CA");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}
