/**
 * The notebook as a generated PDF.
 *
 * This replaces printing the app through the browser's print dialog. That
 * approach produced a document with the browser's own date/URL furniture
 * in the margins, web-page pagination, and — because `print()` fired
 * before the chart data URIs had decoded — a broken-image icon where
 * every chart should have been. Those are not print-stylesheet problems;
 * they follow from not building the document.
 *
 * pdfmake takes a declarative definition and does the parts that are
 * genuinely hard: line breaking, page breaking, keeping a table header
 * with its rows, measuring column widths. What is left is a translation
 * from `lib/report.ts`'s model, which is the same model the Word writer
 * consumes, so the two outputs cannot drift.
 */

import {
  REPORT_COLOR,
  REPORT_TYPE,
  rasterizeSvg,
  type Report,
  type ReportNode,
  type ReportRun,
} from "@/lib/report";

/** A4 minus generous margins, in points — the usable text column. */
const CONTENT_WIDTH = 595.28 - 64 * 2;

type PdfContent = Record<string, unknown>;

function runsToText(runs: ReportRun[]): PdfContent[] {
  return runs.map((r) => ({
    text: r.text,
    ...(r.bold ? { bold: true } : {}),
    ...(r.italic ? { italics: true } : {}),
    ...(r.color ? { color: r.color } : {}),
    ...(r.mono ? { font: "Courier" } : {}),
    ...(r.scale
      ? { fontSize: Math.round(REPORT_TYPE.body * r.scale * 10) / 10 }
      : {}),
  }));
}

export async function exportNotebookAsPdf(report: Report): Promise<Blob> {
  // Both families come from `build/*` bundles that export the same
  // `{ vfs, fonts }` pair — the metrics to register and the descriptor
  // naming the four faces. `build/vfs_fonts` is the other way in, but it
  // is a bare filename→data map whose shape has changed across releases
  // and which carries no descriptor, so this is the one to depend on.
  //
  // Courier is one of PDF's 14 standard faces — metrics only, no
  // embedded outlines — so monospaced SQL costs a few kilobytes of
  // `.afm` rather than a second TTF family.
  const [pdfMakeModule, robotoModule, courierModule] = await Promise.all([
    import("pdfmake/build/pdfmake"),
    import("pdfmake/build/fonts/Roboto.js"),
    import("pdfmake/build/standard-fonts/Courier.js"),
  ]);
  const pdfMake = (pdfMakeModule.default ?? pdfMakeModule) as {
    addVirtualFileSystem: (vfs: Record<string, string>) => void;
    addFonts: (fonts: Record<string, Record<string, string>>) => void;
    // `getBlob` resolves a promise; it does not take a callback. The
    // callback form silently never settles, which reads as a click that
    // did nothing rather than as an error.
    createPdf: (def: unknown) => { getBlob: () => Promise<Blob> };
  };

  type FontBundle = {
    vfs: Record<string, string>;
    fonts: Record<string, Record<string, string>>;
  };
  const unwrap = (m: unknown): FontBundle =>
    ((m as { default?: unknown }).default ?? m) as FontBundle;
  const roboto = unwrap(robotoModule);
  const courier = unwrap(courierModule);

  // Registered on the module, not passed in the definition: a `fonts`
  // key on the document definition is silently ignored, which surfaces
  // only as "font not defined" at layout time.
  pdfMake.addVirtualFileSystem({ ...roboto.vfs, ...courier.vfs });
  pdfMake.addFonts({ ...roboto.fonts, ...courier.fonts });

  // Charts rasterise before the definition is built: pdfmake resolves
  // images synchronously, so every one has to be bytes by then.
  const images = new Map<ReportNode, { dataUri: string; width: number }>();
  await Promise.all(
    report.nodes.map(async (node) => {
      if (node.kind !== "chart") return;
      const raster = await rasterizeSvg(node.svg, node.width, node.height);
      if (raster) {
        images.set(node, {
          dataUri: raster.dataUri,
          width: Math.min(CONTENT_WIDTH, node.width),
        });
      }
    }),
  );

  const content: PdfContent[] = [
    { text: report.title, style: "title" },
  ];

  for (const node of report.nodes) {
    switch (node.kind) {
      case "title":
        content.push({ text: node.text, style: "title" });
        break;
      case "heading":
        content.push({
          text: runsToText(node.runs),
          style: `h${node.level}`,
        });
        break;
      case "paragraph":
        content.push({ text: runsToText(node.runs), style: "body" });
        break;
      case "bullet":
        content.push({
          ul: [{ text: runsToText(node.runs) }],
          style: "body",
        });
        break;
      case "note":
        content.push({ text: node.text, style: "note" });
        break;
      case "code":
        content.push({
          table: {
            widths: ["*"],
            body: [
              [
                {
                  text: node.lines.join("\n"),
                  style: "code",
                  margin: [6, 5, 6, 5],
                },
              ],
            ],
          },
          layout: {
            hLineWidth: () => 0.5,
            vLineWidth: () => 0.5,
            hLineColor: () => REPORT_COLOR.hairline,
            vLineColor: () => REPORT_COLOR.hairline,
            fillColor: () => REPORT_COLOR.soft,
          },
          margin: [0, 0, 0, 10],
        });
        break;
      case "table": {
        content.push({
          table: {
            headerRows: 1,
            // Even columns: a data table reads as a grid, and pdfmake's
            // auto sizing collapses a long text column against a short
            // numeric one.
            widths: node.columns.map(() => "*"),
            body: [
              node.columns.map((c) => ({
                text: c,
                style: "th",
              })),
              ...node.rows.map((row) =>
                row.map((cell) => ({ text: cell, style: "td" })),
              ),
            ],
          },
          layout: {
            hLineWidth: (i: number) => (i === 1 ? 1 : 0.5),
            vLineWidth: () => 0,
            hLineColor: () => REPORT_COLOR.hairline,
            paddingTop: () => 4,
            paddingBottom: () => 4,
            paddingLeft: () => 0,
            paddingRight: () => 8,
            fillColor: (i: number) =>
              i === 0 ? REPORT_COLOR.soft : null,
          },
          margin: [0, 0, 0, node.note ? 3 : 12],
        });
        if (node.note) content.push({ text: node.note, style: "note" });
        break;
      }
      case "chart": {
        const image = images.get(node);
        if (image) {
          content.push({
            image: image.dataUri,
            width: image.width,
            margin: [0, 4, 0, 12],
          });
        }
        break;
      }
    }
  }

  const definition = {
    info: { title: report.title, creator: "MapleQuery" },
    pageSize: "A4",
    pageMargins: [64, 56, 64, 56],
    defaultStyle: {
      font: "Roboto",
      fontSize: REPORT_TYPE.body,
      color: REPORT_COLOR.body,
      lineHeight: 1.35,
    },
    styles: {
      title: {
        fontSize: REPORT_TYPE.title,
        bold: true,
        color: REPORT_COLOR.ink,
        margin: [0, 0, 0, 16],
      },
      h1: {
        fontSize: REPORT_TYPE.h1,
        bold: true,
        color: REPORT_COLOR.ink,
        margin: [0, 14, 0, 6],
      },
      h2: {
        fontSize: REPORT_TYPE.h2,
        bold: true,
        color: REPORT_COLOR.ink,
        margin: [0, 12, 0, 5],
      },
      h3: {
        fontSize: REPORT_TYPE.h3,
        bold: true,
        color: REPORT_COLOR.ink,
        margin: [0, 10, 0, 4],
      },
      body: { margin: [0, 0, 0, 8] },
      note: {
        fontSize: REPORT_TYPE.note,
        color: REPORT_COLOR.muted,
        italics: true,
        margin: [0, 0, 0, 10],
      },
      code: {
        font: "Courier",
        fontSize: REPORT_TYPE.code,
        color: REPORT_COLOR.body,
        lineHeight: 1.25,
      },
      th: {
        fontSize: REPORT_TYPE.note,
        bold: true,
        color: REPORT_COLOR.muted,
        margin: [6, 0, 0, 0],
      },
      td: {
        fontSize: REPORT_TYPE.note + 0.5,
        color: REPORT_COLOR.body,
        margin: [6, 0, 0, 0],
      },
    },
    // Page numbers only, and only past page one — the document's own
    // furniture rather than the browser's.
    footer: (currentPage: number, pageCount: number) =>
      pageCount > 1
        ? {
            text: `${currentPage} / ${pageCount}`,
            alignment: "center",
            fontSize: REPORT_TYPE.note,
            color: REPORT_COLOR.muted,
            margin: [0, 20, 0, 0],
          }
        : undefined,
    content,
  };

  return pdfMake.createPdf(definition).getBlob();
}
