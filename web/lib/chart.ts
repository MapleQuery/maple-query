/**
 * Result rows → an SVG chart, as a pure function of the data.
 *
 * Two properties drive every decision here.
 *
 * **It renders to a string, not to a component tree.** The notebook's
 * Markdown and PDF exports run the blocks through `react-markdown` in a
 * detached iframe, so anything that exists only as mounted React is
 * absent from the exported report. A function that returns SVG markup
 * can be drawn on screen *and* embedded in the Markdown as an image, so
 * one definition serves both and neither can drift from the other.
 *
 * **The spec is inferred, not asked for.** Which column is the category
 * and which is the value is a structural question about the result set,
 * and the loop already paid for a model call to produce those rows. So
 * it is computed here — deterministically, offline, for free — and the
 * user can override it. The same reasoning the header detector uses.
 *
 * Colour, marks and labelling follow a validated single-hue spec:
 * `#005B9F` clears the lightness band, chroma floor and 3:1 contrast on
 * both the white card and white paper. Only one series is ever plotted,
 * so there is no legend — the block's question and the axis caption
 * already say what is drawn.
 */

export type ChartType = "bar" | "line";

export interface ChartSpec {
  type: ChartType;
  categoryColumn: string;
  valueColumn: string;
}

export interface ChartPoint {
  label: string;
  value: number;
}

export interface ChartData {
  points: ChartPoint[];
  /** Rows dropped by the cap below. Rendered, never swallowed. */
  omitted: number;
  /** Rows whose value or label would not parse. Also rendered. */
  unusable: number;
}

/** Bars past this stop being a comparison and start being a table. */
const MAX_BARS = 12;
/** A line can carry more, but not an unbounded number. */
const MAX_LINE_POINTS = 60;
/** Share of a column's values that must parse as numbers to plot it. */
const NUMERIC_SHARE = 0.8;

const SERIES = "#005B9F";
const HAIRLINE = "#D3DADD";
const MUTED = "#5C6B75";
const INK = "#13181B";
const SURFACE = "#FFFFFF";
const FONT =
  "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif";

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/**
 * Tolerant numeric read. Values reach the client both as real numbers
 * (`SAFE_CAST(... AS FLOAT64)`) and as strings straight out of
 * `JSON_VALUE`, and the corpus writes them with thousands separators,
 * currency marks, trailing percents and accounting parens for negatives.
 */
export function toNumber(value: unknown): number | null {
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : null;
  }
  if (typeof value !== "string") return null;
  const raw = value.trim();
  if (!raw) return null;
  const negated = /^\(.*\)$/.test(raw);
  // `\s` already covers the two no-break spaces, but this corpus writes
  // thousands with them often enough to be worth naming.
  const cleaned = raw.replace(/[()$,\s\u00a0\u202f]/g, "").replace(/%$/, "");
  if (!/^[-+]?(?:\d+\.?\d*|\.\d+)$/.test(cleaned)) return null;
  const n = Number(cleaned);
  if (!Number.isFinite(n)) return null;
  return negated ? -Math.abs(n) : n;
}

const YEAR_RE = /^(1[89]\d{2}|2[01]\d{2})$/;
/** `2023-24`, `2023-2024`, `2023/24` — how this corpus writes a fiscal year. */
const FISCAL_RE = /^(1[89]\d{2}|2[01]\d{2})\s*[-/]\s*(\d{2}|\d{4})$/;
const ISO_DATE_RE = /^\d{4}-\d{2}(-\d{2})?/;

/** A sortable key for a temporal label, or null if it is not temporal. */
function temporalKey(label: string): number | null {
  const s = label.trim();
  const year = YEAR_RE.exec(s);
  if (year) return Number(year[1]);
  const fiscal = FISCAL_RE.exec(s);
  if (fiscal) return Number(fiscal[1]);
  if (ISO_DATE_RE.test(s)) {
    const t = Date.parse(s);
    if (Number.isFinite(t)) return t;
  }
  return null;
}

function cellToLabel(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value).trim();
}

// ---------------------------------------------------------------------------
// Spec inference
// ---------------------------------------------------------------------------

function numericShare(
  rows: Record<string, unknown>[],
  column: string,
): number {
  if (rows.length === 0) return 0;
  let n = 0;
  for (const row of rows) {
    if (toNumber(row[column]) !== null) n += 1;
  }
  return n / rows.length;
}

function temporalShare(
  rows: Record<string, unknown>[],
  column: string,
): number {
  if (rows.length === 0) return 0;
  let n = 0;
  for (const row of rows) {
    if (temporalKey(cellToLabel(row[column])) !== null) n += 1;
  }
  return n / rows.length;
}

/**
 * Choose a category column, a value column and a form, or return null
 * when the rows are not chartable (a single column, no numeric column,
 * nothing but one row).
 *
 * The category is the first column that is *not* mostly numeric — the
 * shape `GROUP BY dimension` produces. When every column is numeric,
 * which is what a `GROUP BY year` looks like, the first column is taken
 * as the category and the next numeric one as the value.
 */
export function inferChartSpec(
  rows: Record<string, unknown>[],
): ChartSpec | null {
  if (rows.length < 2) return null;
  const columns = Object.keys(rows[0] ?? {});
  if (columns.length < 2) return null;

  const numeric = columns.filter(
    (c) => numericShare(rows, c) >= NUMERIC_SHARE,
  );
  const nonNumeric = columns.filter((c) => !numeric.includes(c));

  const categoryColumn = nonNumeric[0] ?? columns[0];
  const valueColumn = numeric.find((c) => c !== categoryColumn);
  if (!valueColumn) return null;

  // A time axis is a line; anything else is a magnitude comparison, and
  // horizontal bars are what survive this corpus's category names
  // ("Department of Crown-Indigenous Relations and Northern Affairs").
  const type: ChartType =
    temporalShare(rows, categoryColumn) >= NUMERIC_SHARE ? "line" : "bar";
  return { type, categoryColumn, valueColumn };
}

/**
 * A block's stored chart preferences. Every field is optional: a block
 * that has never been touched stores nothing and gets the inference,
 * which is what makes the chart appear on its own the first time.
 */
export interface ChartOverrides {
  type?: ChartType;
  categoryColumn?: string;
  valueColumn?: string;
  /** The user dismissed the chart for this block. */
  hidden?: boolean;
}

/**
 * The spec actually drawn: the inference, with any stored override laid
 * over it. Shared by the on-screen block and the export so the report
 * cannot disagree with the screen.
 */
export function resolveChartSpec(
  rows: Record<string, unknown>[],
  overrides?: ChartOverrides,
): ChartSpec | null {
  if (overrides?.hidden) return null;
  // A stored override outlives the run that produced it: re-running a
  // block against a reworded question returns different columns, and a
  // preference naming a column that is no longer there has to yield to
  // the fresh inference rather than draw an empty chart.
  const columns = new Set(Object.keys(rows[0] ?? {}));
  const held = (name: string | undefined): string | undefined =>
    name && columns.has(name) ? name : undefined;
  const category = held(overrides?.categoryColumn);
  const value = held(overrides?.valueColumn);

  const inferred = inferChartSpec(rows);
  if (!inferred) {
    // Inference declining is not a veto. A user who names both columns
    // has said more than the heuristic could work out.
    if (category && value) {
      return {
        type: overrides?.type ?? "bar",
        categoryColumn: category,
        valueColumn: value,
      };
    }
    return null;
  }
  return {
    type: overrides?.type ?? inferred.type,
    categoryColumn: category ?? inferred.categoryColumn,
    valueColumn: value ?? inferred.valueColumn,
  };
}

/**
 * Project rows onto the spec: parse, drop what will not parse, order for
 * the form, and cap. Both the cap and the drops come back as counts so
 * the caller can say what is missing rather than quietly showing less.
 */
export function chartData(
  rows: Record<string, unknown>[],
  spec: ChartSpec,
): ChartData {
  const parsed: { label: string; value: number; key: number | null }[] = [];
  let unusable = 0;
  for (const row of rows) {
    const value = toNumber(row[spec.valueColumn]);
    const label = cellToLabel(row[spec.categoryColumn]);
    if (value === null || !label) {
      unusable += 1;
      continue;
    }
    parsed.push({ label, value, key: temporalKey(label) });
  }

  if (spec.type === "line") {
    // An unordered time axis draws a line that is simply false, so sort
    // by the parsed key where every point has one; otherwise trust the
    // order the query returned.
    if (parsed.length > 0 && parsed.every((p) => p.key !== null)) {
      parsed.sort((a, b) => (a.key as number) - (b.key as number));
    }
    const points = parsed.slice(0, MAX_LINE_POINTS);
    return {
      points: points.map(({ label, value }) => ({ label, value })),
      omitted: parsed.length - points.length,
      unusable,
    };
  }

  parsed.sort((a, b) => b.value - a.value);
  const points = parsed.slice(0, MAX_BARS);
  return {
    points: points.map(({ label, value }) => ({ label, value })),
    omitted: parsed.length - points.length,
    unusable,
  };
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

/**
 * Compact magnitude, hand-rolled rather than `Intl` so the same rows
 * always render the same string — the export has to match the screen.
 */
export function formatCompact(value: number): string {
  const abs = Math.abs(value);
  const sign = value < 0 ? "-" : "";
  const scaled = (divisor: number, suffix: string): string => {
    const n = abs / divisor;
    const digits = n < 10 ? 1 : 0;
    return `${sign}${n.toFixed(digits)}${suffix}`;
  };
  if (abs >= 1e12) return scaled(1e12, "T");
  if (abs >= 1e9) return scaled(1e9, "B");
  if (abs >= 1e6) return scaled(1e6, "M");
  if (abs >= 1e3) return scaled(1e3, "K");
  if (Number.isInteger(value)) return String(value);
  return `${sign}${abs.toFixed(2)}`;
}

function escapeXml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

/**
 * Width of `text` at `size`, in px. An estimate — there is no text
 * metric available while building a string — deliberately run a little
 * wide so a gutter sized from it does not clip.
 */
function textWidth(text: string, size: number): number {
  return text.length * size * 0.58;
}

function truncateToWidth(text: string, size: number, max: number): string {
  if (textWidth(text, size) <= max) return text;
  const room = Math.max(1, Math.floor(max / (size * 0.58)) - 1);
  return `${text.slice(0, room).trimEnd()}…`;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

export interface RenderOptions {
  width?: number;
  /** Names the value axis. The block's question names the rest. */
  valueLabel?: string;
}

const LABEL_SIZE = 12;
const VALUE_SIZE = 12;

/**
 * Render to a standalone SVG string: no external font, no stylesheet, no
 * script. Every bar and point carries a `<title>`, which is the hover
 * layer a string-rendered chart can honestly provide — the browser reads
 * it natively on screen and it costs nothing in the PDF.
 */
export function renderChartSvg(
  data: ChartData,
  spec: ChartSpec,
  options: RenderOptions = {},
): string {
  if (data.points.length === 0) return "";
  const width = options.width ?? 720;
  return spec.type === "line"
    ? renderLine(data, spec, width, options)
    : renderBars(data, spec, width, options);
}

function renderBars(
  data: ChartData,
  spec: ChartSpec,
  width: number,
  options: RenderOptions,
): string {
  const { points } = data;
  const rowHeight = 28;
  const barThickness = 20; // ≤ 24px; the 8px remainder is the surface gap
  const top = 8;
  const captionHeight = captionLines(data, options).length * 16;
  const height = top + points.length * rowHeight + 8 + captionHeight;

  // Gutters: category names on the left, value labels at the tips. Both
  // are measured before anything is placed, so a label never overruns
  // the plot or gets clipped by it.
  const valueTexts = points.map((p) => formatCompact(p.value));
  const valueGutter =
    Math.max(...valueTexts.map((t) => textWidth(t, VALUE_SIZE))) + 12;
  // Wide, because this corpus names its categories "Department of
  // Crown-Indigenous Relations and Northern Affairs". Anything longer
  // still truncates, and the full name stays in the hover title.
  const labelGutter = Math.min(300, Math.max(96, width * 0.36));
  const plotLeft = labelGutter + 12;
  const plotWidth = Math.max(24, width - plotLeft - valueGutter);

  // Bars grow from a zero baseline. A negative value is rare in this
  // corpus but must not render as a positive one, so the scale spans
  // whatever range is actually present.
  const values = points.map((p) => p.value);
  const max = Math.max(0, ...values);
  const min = Math.min(0, ...values);
  const span = max - min || 1;
  const zeroX = plotLeft + ((0 - min) / span) * plotWidth;

  const parts: string[] = [];
  points.forEach((point, i) => {
    const y = top + i * rowHeight + (rowHeight - barThickness) / 2;
    const valueX = plotLeft + ((point.value - min) / span) * plotWidth;
    const x = Math.min(zeroX, valueX);
    const barWidth = Math.max(1, Math.abs(valueX - zeroX));
    const label = truncateToWidth(point.label, LABEL_SIZE, labelGutter);
    parts.push(
      `<g><title>${escapeXml(point.label)}: ${escapeXml(
        point.value.toLocaleString("en-CA"),
      )}</title>`,
      `<text x="${labelGutter}" y="${y + barThickness / 2 + 4}" ` +
        `text-anchor="end" font-family="${FONT}" font-size="${LABEL_SIZE}" ` +
        `fill="${MUTED}">${escapeXml(label)}</text>`,
      barPath(x, y, barWidth, barThickness, point.value < 0),
      `<text x="${
        point.value < 0 ? x - 6 : x + barWidth + 6
      }" y="${y + barThickness / 2 + 4}" text-anchor="${
        point.value < 0 ? "end" : "start"
      }" font-family="${FONT}" font-size="${VALUE_SIZE}" fill="${INK}">` +
        `${escapeXml(valueTexts[i])}</text>`,
      `</g>`,
    );
  });

  // A zero rule only where the data actually crosses it. With every bar
  // positive the bar ends are the axis, and the direct labels carry the
  // values, so no gridlines are drawn at all.
  if (min < 0) {
    parts.push(
      `<line x1="${zeroX}" y1="${top}" x2="${zeroX}" y2="${
        top + points.length * rowHeight
      }" stroke="${HAIRLINE}" stroke-width="1" />`,
    );
  }

  parts.push(
    ...caption(data, options, 0, top + points.length * rowHeight + 20),
  );
  return svgDocument(width, height, parts.join(""), chartAltText(data, spec));
}

/** Square at the baseline, 4px rounded at the data end. */
function barPath(
  x: number,
  y: number,
  w: number,
  h: number,
  negative: boolean,
): string {
  const r = Math.min(4, w);
  const d = negative
    ? `M ${x + w} ${y} H ${x + r} A ${r} ${r} 0 0 0 ${x} ${y + r} V ${
        y + h - r
      } A ${r} ${r} 0 0 0 ${x + r} ${y + h} H ${x + w} Z`
    : `M ${x} ${y} H ${x + w - r} A ${r} ${r} 0 0 1 ${x + w} ${y + r} V ${
        y + h - r
      } A ${r} ${r} 0 0 1 ${x + w - r} ${y + h} H ${x} Z`;
  return `<path d="${d}" fill="${SERIES}" />`;
}

function renderLine(
  data: ChartData,
  spec: ChartSpec,
  width: number,
  options: RenderOptions,
): string {
  const { points } = data;
  const top = 16;
  const plotHeight = 220;
  const captionHeight = captionLines(data, options).length * 16;
  const height = top + plotHeight + 42 + captionHeight;

  const values = points.map((p) => p.value);
  const rawMax = Math.max(...values);
  const rawMin = Math.min(...values);
  const max = niceCeil(Math.max(rawMax, 0));
  const min = rawMin < 0 ? -niceCeil(Math.abs(rawMin)) : 0;
  const span = max - min || 1;

  const ticks = [max, min + span / 2, min];
  const tickTexts = formatAxisTicks(ticks);
  const left = Math.max(...tickTexts.map((t) => textWidth(t, VALUE_SIZE))) + 12;
  const endText = formatCompact(points[points.length - 1].value);
  const right = textWidth(endText, VALUE_SIZE) + 16;
  const plotWidth = Math.max(24, width - left - right);

  const xAt = (i: number): number =>
    points.length === 1
      ? left + plotWidth / 2
      : left + (i / (points.length - 1)) * plotWidth;
  const yAt = (v: number): number =>
    top + plotHeight - ((v - min) / span) * plotHeight;

  const parts: string[] = [];
  // Three gridlines, hairline and solid. A line chart has no direct
  // label on every point, so the ticks carry the values the end label
  // does not.
  ticks.forEach((tick, i) => {
    const y = yAt(tick);
    parts.push(
      `<line x1="${left}" y1="${y}" x2="${
        left + plotWidth
      }" y2="${y}" stroke="${HAIRLINE}" stroke-width="1" />`,
      `<text x="${left - 8}" y="${
        y + 4
      }" text-anchor="end" font-family="${FONT}" font-size="${VALUE_SIZE}" ` +
        `fill="${MUTED}">${escapeXml(tickTexts[i])}</text>`,
    );
  });

  const d = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${xAt(i)} ${yAt(p.value)}`)
    .join(" ");
  parts.push(
    `<path d="${d}" fill="none" stroke="${SERIES}" stroke-width="2" ` +
      `stroke-linejoin="round" stroke-linecap="round" />`,
  );

  // Hover targets on every point; a visible marker only on the last one,
  // where the value is also labelled.
  points.forEach((p, i) => {
    parts.push(
      `<circle cx="${xAt(i)}" cy="${yAt(p.value)}" r="8" fill="transparent">` +
        `<title>${escapeXml(p.label)}: ${escapeXml(
          p.value.toLocaleString("en-CA"),
        )}</title></circle>`,
    );
  });
  const lastX = xAt(points.length - 1);
  const lastY = yAt(points[points.length - 1].value);
  parts.push(
    // 2px surface ring so the marker stays legible over the line.
    `<circle cx="${lastX}" cy="${lastY}" r="4" fill="${SERIES}" ` +
      `stroke="${SURFACE}" stroke-width="2" />`,
    `<text x="${lastX + 8}" y="${
      lastY + 4
    }" font-family="${FONT}" font-size="${VALUE_SIZE}" fill="${INK}">` +
      `${escapeXml(endText)}</text>`,
  );

  // First and last category only. Every tick would collide, and the
  // tooltip carries the rest.
  const axisY = top + plotHeight + 18;
  parts.push(
    `<text x="${left}" y="${axisY}" font-family="${FONT}" ` +
      `font-size="${LABEL_SIZE}" fill="${MUTED}">${escapeXml(
        truncateToWidth(points[0].label, LABEL_SIZE, plotWidth / 2 - 8),
      )}</text>`,
  );
  if (points.length > 1) {
    parts.push(
      `<text x="${left + plotWidth}" y="${axisY}" text-anchor="end" ` +
        `font-family="${FONT}" font-size="${LABEL_SIZE}" fill="${MUTED}">` +
        `${escapeXml(
          truncateToWidth(
            points[points.length - 1].label,
            LABEL_SIZE,
            plotWidth / 2 - 8,
          ),
        )}</text>`,
    );
  }

  parts.push(...caption(data, options, 0, axisY + 20));
  return svgDocument(width, height, parts.join(""), chartAltText(data, spec));
}

/**
 * Round a maximum up to a clean number so ticks read as numbers.
 *
 * The ladder is finer than the usual 1/2/5 because that one wastes half
 * the plot on the common case: a 5.3M series rounds to 10M and the line
 * sits in the bottom half of its own chart.
 */
const TICK_LADDER = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10];

function niceCeil(value: number): number {
  if (value <= 0) return 0;
  const magnitude = 10 ** Math.floor(Math.log10(value));
  const normalized = value / magnitude;
  const step = TICK_LADDER.find((s) => normalized <= s) ?? 10;
  return step * magnitude;
}

/**
 * Format a tick set to one shared scale, so an axis never reads
 * `10M` above `5.0M`. Zero is always bare.
 */
function formatAxisTicks(values: number[]): string[] {
  const peak = Math.max(...values.map(Math.abs));
  const [divisor, suffix]: [number, string] =
    peak >= 1e12
      ? [1e12, "T"]
      : peak >= 1e9
        ? [1e9, "B"]
        : peak >= 1e6
          ? [1e6, "M"]
          : peak >= 1e3
            ? [1e3, "K"]
            : [1, ""];
  const scaled = values.map((v) => v / divisor);
  const digits = scaled.every((n) => Number.isInteger(n)) ? 0 : 1;
  return values.map((v) =>
    v === 0 ? "0" : `${(v / divisor).toFixed(digits)}${suffix}`,
  );
}

function captionLines(data: ChartData, options: RenderOptions): string[] {
  const lines: string[] = [];
  if (options.valueLabel) lines.push(options.valueLabel);
  const notes: string[] = [];
  // Never a silent cap: what was left out is part of the chart.
  if (data.omitted > 0) notes.push(`${data.omitted} more not shown`);
  if (data.unusable > 0) notes.push(`${data.unusable} non-numeric skipped`);
  if (notes.length > 0) lines.push(notes.join(" · "));
  return lines;
}

function caption(
  data: ChartData,
  options: RenderOptions,
  x: number,
  y: number,
): string[] {
  return captionLines(data, options).map(
    (line, i) =>
      `<text x="${x}" y="${y + i * 16}" font-family="${FONT}" ` +
      `font-size="11" fill="${MUTED}">${escapeXml(line)}</text>`,
  );
}

function chartAltText(data: ChartData, spec: ChartSpec): string {
  const form = spec.type === "line" ? "Line chart" : "Bar chart";
  return `${form} of ${spec.valueColumn} by ${spec.categoryColumn}, ${data.points.length} of ${data.points.length + data.omitted} categories.`;
}

function svgDocument(
  width: number,
  height: number,
  body: string,
  alt: string,
): string {
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${Math.round(
      height,
    )}" viewBox="0 0 ${width} ${Math.round(height)}" role="img" ` +
    `aria-label="${escapeXml(alt)}">` +
    `<rect width="${width}" height="${Math.round(
      height,
    )}" fill="${SURFACE}" />${body}</svg>`
  );
}

/**
 * Base64 data URI, so the chart can be embedded with ordinary Markdown
 * image syntax. That is what carries it through the export: the print
 * pipeline renders Markdown with no raw-HTML plugin, and an `![](…)`
 * image needs none.
 */
export function svgToDataUri(svg: string): string {
  const bytes = new TextEncoder().encode(svg);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return `data:image/svg+xml;base64,${btoa(binary)}`;
}
