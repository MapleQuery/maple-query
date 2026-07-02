import type { StoredNotebook } from "@/lib/storage";

/**
 * Concatenate the notebook into a single Markdown document.
 *
 * Layout:
 *   - Prose blocks → their markdown verbatim
 *   - Query blocks →
 *       ### <question>
 *       <assistant text>
 *       ```sql
 *       <sql>
 *       ```
 *       | col1 | col2 |
 *       | ---- | ---- |
 *       | v1   | v2   |
 *       *Sources: <package_ids>*
 */
export function exportNotebookAsMarkdown(nb: StoredNotebook): string {
  const parts: string[] = [];
  parts.push(`# ${nb.title || "Untitled notebook"}`);
  parts.push(
    `_Exported ${new Date().toISOString()} · ${nb.blocks.length} block${nb.blocks.length === 1 ? "" : "s"}_`,
  );
  parts.push("");

  for (const b of nb.blocks) {
    if (b.type === "prose") {
      parts.push(b.markdown.trim() || "_(empty prose)_");
      parts.push("");
    } else {
      parts.push(`### ${b.question || "(empty query)"}`);
      if (b.result?.assistantText) {
        parts.push(b.result.assistantText.trim());
        parts.push("");
      }
      if (b.result?.sql) {
        parts.push("```sql");
        parts.push(b.result.sql.trim());
        parts.push("```");
        parts.push("");
      }
      if (b.result?.rows && b.result.rows.length > 0) {
        parts.push(rowsToMarkdownTable(b.result.rows.slice(0, 20)));
        parts.push("");
      }
      if (b.result?.packageIds && b.result.packageIds.length > 0) {
        parts.push(`_Sources: ${b.result.packageIds.join(", ")}_`);
        parts.push("");
      }
    }
  }

  return parts.join("\n");
}

function rowsToMarkdownTable(rows: Record<string, unknown>[]): string {
  if (rows.length === 0) return "";
  const cols = Object.keys(rows[0] ?? {});
  const escape = (v: unknown): string => {
    if (v == null) return "";
    const s = typeof v === "object" ? JSON.stringify(v) : String(v);
    return s.replace(/\|/g, "\\|").replace(/\n/g, " ");
  };
  const header = `| ${cols.join(" | ")} |`;
  const rule = `| ${cols.map(() => "---").join(" | ")} |`;
  const body = rows
    .map((r) => `| ${cols.map((c) => escape(r[c])).join(" | ")} |`)
    .join("\n");
  return `${header}\n${rule}\n${body}`;
}
