"use client";

import { createRoot } from "react-dom/client";
import { flushSync } from "react-dom";
import ReactMarkdown, { defaultUrlTransform } from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * Charts travel as `data:image/svg+xml;base64,…` images, and
 * `react-markdown`'s default transform allows only http/https/mailto and
 * friends — it would drop the `src` and print a blank where every chart
 * should be, without erroring. So exactly that one prefix is allowed
 * back in; everything else still goes through the default.
 */
const CHART_URI = "data:image/svg+xml;base64,";

function printUrlTransform(url: string): string | null | undefined {
  return url.startsWith(CHART_URI) ? url : defaultUrlTransform(url);
}

/**
 * PDF export is the Markdown export, rendered and handed to the
 * browser's print pipeline.
 *
 * Rendering it with the same ReactMarkdown/remark-gfm pair the notebook
 * uses on screen means there is exactly one document definition to keep
 * correct — a second, PDF-only formatter would drift from the Markdown
 * within a release. The print pipeline also gives real text (selectable,
 * searchable) and real pagination, which a canvas-to-image PDF does not.
 *
 * It renders into an off-screen iframe rather than the page so the
 * notebook's own layout and stylesheet play no part in what comes out.
 */

const PRINT_CSS = `
  *, *::before, *::after { box-sizing: border-box; }
  @page { margin: 18mm 16mm; }
  html, body { margin: 0; padding: 0; }
  body {
    font-family: ui-sans-serif, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.55;
    color: #14181f;
    background: #fff;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  main { padding: 0; }
  h1 { font-size: 21pt; line-height: 1.2; margin: 0 0 4pt; }
  h2 { font-size: 15pt; margin: 18pt 0 6pt; }
  h3 { font-size: 12.5pt; margin: 16pt 0 5pt; }
  h1, h2, h3, h4 { break-after: avoid-page; page-break-after: avoid; }
  p { margin: 0 0 8pt; }
  em { color: #55606f; }
  ul, ol { margin: 0 0 8pt; padding-left: 16pt; }
  a { color: #14181f; text-decoration: none; }
  hr { border: 0; border-top: 1px solid #e3e7ec; margin: 14pt 0; }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 8.5pt;
    margin: 0 0 10pt;
  }
  th, td {
    border: 1px solid #d5dae1;
    padding: 3pt 5pt;
    text-align: left;
    vertical-align: top;
    word-break: break-word;
  }
  th { background: #f4f6f8; font-weight: 600; }
  tr { break-inside: avoid; page-break-inside: avoid; }
  pre {
    background: #f6f7f9;
    border: 1px solid #e3e7ec;
    border-radius: 5px;
    padding: 7pt 9pt;
    margin: 0 0 10pt;
    white-space: pre-wrap;
    word-break: break-word;
    break-inside: avoid;
    page-break-inside: avoid;
  }
  code {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 8.5pt;
  }
  blockquote {
    margin: 0 0 10pt;
    padding-left: 10pt;
    border-left: 2px solid #d5dae1;
    color: #55606f;
  }
  img {
    display: block;
    max-width: 100%;
    height: auto;
    margin: 0 0 10pt;
    break-inside: avoid;
    page-break-inside: avoid;
  }
`;

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * Render `markdown` into a hidden iframe and open the print dialog on
 * it. The document title seeds the browser's default filename.
 *
 * Resolves once the dialog has been handed off — whether the user then
 * saves a PDF, prints, or cancels is the browser's business.
 */
export async function printMarkdownAsPdf(
  docTitle: string,
  markdown: string,
): Promise<void> {
  const iframe = document.createElement("iframe");
  iframe.setAttribute("aria-hidden", "true");
  iframe.setAttribute("tabindex", "-1");
  // Off-screen at a real page size, not `display:none` — a collapsed or
  // undisplayed iframe has nothing laid out for the dialog to capture.
  iframe.style.cssText =
    "position:fixed;left:-10000px;top:0;width:794px;height:1123px;border:0;";

  const loaded = new Promise<void>((resolve) => {
    iframe.addEventListener("load", () => resolve(), { once: true });
  });
  iframe.srcdoc =
    `<!doctype html><html><head><meta charset="utf-8">` +
    `<title>${escapeHtml(docTitle)}</title>` +
    `<style>${PRINT_CSS}</style></head>` +
    `<body><main id="mq-print-root"></main></body></html>`;

  document.body.appendChild(iframe);
  await loaded;

  const doc = iframe.contentDocument;
  const win = iframe.contentWindow;
  const mount = doc?.getElementById("mq-print-root");
  if (!doc || !win || !mount) {
    iframe.remove();
    throw new Error("print_document_unavailable");
  }

  const root = createRoot(mount);
  // Synchronous so the markup is committed before print() is called;
  // a concurrent render would let the dialog open on an empty page.
  flushSync(() => {
    root.render(
      <ReactMarkdown remarkPlugins={[remarkGfm]} urlTransform={printUrlTransform}>
        {markdown}
      </ReactMarkdown>,
    );
  });
  await new Promise<void>((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
  });

  let disposed = false;
  let fallback = 0;
  const dispose = () => {
    if (disposed) return;
    disposed = true;
    window.clearTimeout(fallback);
    root.unmount();
    iframe.remove();
  };
  // `afterprint` is the signal in Chrome and Firefox; Safari is less
  // dependable, so a timer guarantees the iframe is eventually reaped.
  win.addEventListener("afterprint", dispose, { once: true });
  fallback = window.setTimeout(dispose, 120_000);

  try {
    win.focus();
    win.print();
  } catch (err) {
    dispose();
    throw err;
  }
}
