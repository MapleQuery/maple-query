/**
 * The model is told to write plain Markdown, and still sometimes wraps
 * arithmetic in LaTeX — `\( (199.6 - 195.0) / 195.0 \times 100 \)` —
 * which renders as backslash soup. This turns the handful of constructs
 * it actually emits into readable text instead of pulling in a math
 * renderer for sums a reader should see as sums.
 */
export function plainMath(text: string): string {
  if (!text.includes("\\")) return text;
  return text
    .replace(/\\\[|\\\]|\\\(|\\\)/g, "")
    .replace(/\\frac\{([^{}]*)\}\{([^{}]*)\}/g, "($1) / ($2)")
    .replace(/\\text\{([^{}]*)\}/g, "$1")
    .replace(/\\left|\\right/g, "")
    .replace(/\\times/g, "×")
    .replace(/\\div/g, "÷")
    .replace(/\\approx/g, "≈")
    .replace(/\\cdot/g, "·")
    .replace(/\\%/g, "%")
    .replace(/\\,/g, " ");
}
