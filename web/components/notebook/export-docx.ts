/**
 * The notebook as a Word document.
 *
 * Journalists work in Word, and a PDF is where editing stops. This is the
 * one export that produces something someone can keep writing in.
 *
 * It consumes `lib/report.ts`'s model, the same one the PDF writer takes,
 * so the two outputs stay recognisably the same piece. What is specific
 * here is Word's own vocabulary: real named styles rather than direct
 * formatting, so a heading is navigable, restyleable, and picked up by a
 * table of contents. Word's stock Title and Heading styles are serif and
 * blue; left alone they make every export look like a school essay, so
 * the ones this document uses are declared outright.
 *
 * `docx` is imported dynamically — a few hundred kilobytes of OOXML
 * writer that only matters once someone picks this format.
 */

import type { Paragraph as ParagraphT, Table as TableT } from "docx";
import {
  REPORT_COLOR,
  REPORT_TYPE,
  bareHex,
  rasterizeSvg,
  type Report,
  type ReportRun,
} from "@/lib/report";

/** Word sizes are half-points. */
const hp = (points: number): number => Math.round(points * 2);

export async function exportNotebookAsDocx(report: Report): Promise<Blob> {
  const {
    AlignmentType,
    BorderStyle,
    Document,
    HeadingLevel,
    ImageRun,
    Packer,
    Paragraph,
    ShadingType,
    Table,
    TableCell,
    TableRow,
    TextRun,
    WidthType,
  } = await import("docx");

  // Rasterise first: the document tree is built synchronously below.
  const charts = new Map<
    number,
    { bytes: Uint8Array; width: number; height: number }
  >();
  await Promise.all(
    report.nodes.map(async (node, i) => {
      if (node.kind !== "chart") return;
      const raster = await rasterizeSvg(node.svg, node.width, node.height);
      if (raster) {
        // Scaled to the text column so a wide chart does not run into
        // the margin.
        const width = Math.min(node.width, 460);
        charts.set(i, {
          bytes: raster.bytes,
          width,
          height: Math.round((node.height / node.width) * width),
        });
      }
    }),
  );

  const runs = (source: ReportRun[]): InstanceType<typeof TextRun>[] =>
    source.map(
      (r) =>
        new TextRun({
          text: r.text,
          ...(r.bold ? { bold: true } : {}),
          ...(r.italic ? { italics: true } : {}),
          ...(r.color ? { color: bareHex(r.color) } : {}),
          ...(r.mono ? { font: "Consolas" } : {}),
          ...(r.scale ? { size: hp(REPORT_TYPE.body * r.scale) } : {}),
        }),
    );

  const children: (ParagraphT | TableT)[] = [];
  const push = (n: ParagraphT | TableT) => children.push(n);

  push(
    new Paragraph({
      text: report.title,
      heading: HeadingLevel.TITLE,
    }),
  );

  report.nodes.forEach((node, i) => {
    switch (node.kind) {
      case "title":
        push(new Paragraph({ text: node.text, heading: HeadingLevel.TITLE }));
        break;
      case "heading":
        push(
          new Paragraph({
            heading:
              node.level === 1
                ? HeadingLevel.HEADING_1
                : node.level === 2
                  ? HeadingLevel.HEADING_2
                  : HeadingLevel.HEADING_3,
            children: runs(node.runs),
          }),
        );
        break;
      case "paragraph":
        push(new Paragraph({ children: runs(node.runs) }));
        break;
      case "bullet":
        push(
          new Paragraph({ children: runs(node.runs), bullet: { level: 0 } }),
        );
        break;
      case "note":
        push(new Paragraph({ style: "Note", children: [new TextRun(node.text)] }));
        break;
      case "code":
        node.lines.forEach((line, idx) =>
          push(
            new Paragraph({
              style: "Code",
              // Shading on every line so the block reads as one panel.
              shading: {
                type: ShadingType.CLEAR,
                fill: bareHex(REPORT_COLOR.soft),
              },
              spacing: {
                before: idx === 0 ? 80 : 0,
                after: idx === node.lines.length - 1 ? 160 : 0,
              },
              children: [new TextRun(line || " ")],
            }),
          ),
        );
        break;
      case "table": {
        const border = {
          style: BorderStyle.SINGLE,
          size: 4,
          color: bareHex(REPORT_COLOR.hairline),
        };
        push(
          new Table({
            width: { size: 100, type: WidthType.PERCENTAGE },
            borders: {
              top: border,
              bottom: border,
              left: { style: BorderStyle.NONE, size: 0, color: "auto" },
              right: { style: BorderStyle.NONE, size: 0, color: "auto" },
              insideHorizontal: border,
              insideVertical: {
                style: BorderStyle.NONE,
                size: 0,
                color: "auto",
              },
            },
            rows: [
              new TableRow({
                tableHeader: true,
                children: node.columns.map(
                  (c) =>
                    new TableCell({
                      shading: {
                        type: ShadingType.CLEAR,
                        fill: bareHex(REPORT_COLOR.soft),
                      },
                      children: [
                        new Paragraph({
                          style: "TableText",
                          children: [
                            new TextRun({
                              text: c,
                              bold: true,
                              color: bareHex(REPORT_COLOR.muted),
                            }),
                          ],
                        }),
                      ],
                    }),
                ),
              }),
              ...node.rows.map(
                (row) =>
                  new TableRow({
                    children: row.map(
                      (cell) =>
                        new TableCell({
                          children: [
                            new Paragraph({
                              style: "TableText",
                              children: [new TextRun(cell)],
                            }),
                          ],
                        }),
                    ),
                  }),
              ),
            ],
          }),
        );
        if (node.note) {
          push(
            new Paragraph({
              style: "Note",
              children: [new TextRun(node.note)],
            }),
          );
        }
        break;
      }
      case "chart": {
        const chart = charts.get(i);
        if (chart) {
          push(
            new Paragraph({
              alignment: AlignmentType.CENTER,
              spacing: { before: 120, after: 200 },
              children: [
                new ImageRun({
                  type: "png",
                  data: chart.bytes,
                  transformation: {
                    width: chart.width,
                    height: chart.height,
                  },
                }),
              ],
            }),
          );
        }
        break;
      }
    }
  });

  const doc = new Document({
    creator: "MapleQuery",
    title: report.title,
    styles: {
      default: {
        document: {
          run: {
            font: "Calibri",
            size: hp(REPORT_TYPE.body),
            color: bareHex(REPORT_COLOR.body),
          },
          paragraph: { spacing: { line: 300, after: 140 } },
        },
        // Word's stock heading styles are blue Calibri Light and its
        // Title is a serif with a rule under it. Overridden so an export
        // looks like this product rather than like Word's defaults.
        title: {
          run: {
            font: "Calibri",
            size: hp(REPORT_TYPE.title),
            bold: true,
            color: bareHex(REPORT_COLOR.ink),
          },
          paragraph: { spacing: { after: 260 } },
        },
        heading1: {
          run: {
            font: "Calibri",
            size: hp(REPORT_TYPE.h1),
            bold: true,
            color: bareHex(REPORT_COLOR.ink),
          },
          paragraph: { spacing: { before: 280, after: 120 } },
        },
        heading2: {
          run: {
            font: "Calibri",
            size: hp(REPORT_TYPE.h2),
            bold: true,
            color: bareHex(REPORT_COLOR.ink),
          },
          paragraph: { spacing: { before: 240, after: 100 } },
        },
        heading3: {
          run: {
            font: "Calibri",
            size: hp(REPORT_TYPE.h3),
            bold: true,
            color: bareHex(REPORT_COLOR.ink),
          },
          paragraph: { spacing: { before: 200, after: 80 } },
        },
      },
      paragraphStyles: [
        {
          id: "Note",
          name: "Note",
          basedOn: "Normal",
          run: {
            size: hp(REPORT_TYPE.note),
            italics: true,
            color: bareHex(REPORT_COLOR.muted),
          },
          paragraph: { spacing: { after: 180 } },
        },
        {
          id: "Code",
          name: "Code",
          basedOn: "Normal",
          run: {
            font: "Consolas",
            size: hp(REPORT_TYPE.code),
            color: bareHex(REPORT_COLOR.body),
          },
          paragraph: { spacing: { line: 240, after: 0 } },
        },
        {
          id: "TableText",
          name: "Table Text",
          basedOn: "Normal",
          run: { size: hp(REPORT_TYPE.note + 0.5) },
          paragraph: { spacing: { before: 40, after: 40, line: 240 } },
        },
      ],
    },
    sections: [
      {
        properties: {
          page: { margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } },
        },
        children,
      },
    ],
  });
  return Packer.toBlob(doc);
}
