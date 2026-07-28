import type { StoredNotebook } from "@/lib/storage";
import type { DatasetTitleMap } from "@/lib/dataset-titles";

/**
 * Concatenate the notebook into a single Markdown document.
 *
 * Layout:
 *   - Prose blocks → their markdown verbatim
 *   - Query blocks →
 *       ### <question>
 *       _Scoped to: <datasets>_   (only when the block is scoped)
 *       <assistant text>
 *       ```sql
 *       <sql>
 *       ```
 *       | col1 | col2 |
 *       | ---- | ---- |
 *       | v1   | v2   |
 *       *Sources: <datasets>*
 *
 * Datasets are named by title where one is known — a reader of the
 * exported report has no way to look a UUID up. `titles` is the shared
 * cache; a block's own recorded titles win over it.
 */
export function exportNotebookAsMarkdown(
  nb: StoredNotebook,
  titles?: DatasetTitleMap,
): string {
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
      const name = (id: string): string =>
        b.result?.packageTitles?.[id] ?? titles?.[id] ?? id;
      parts.push(`### ${b.question || "(empty query)"}`);
      if (b.scopePackageIds && b.scopePackageIds.length > 0) {
        // A reader of the exported Markdown should be able to see what
        // a question was narrowed to; otherwise a scoped block reads as
        // an unaccountably specific answer.
        parts.push(`_Scoped to: ${b.scopePackageIds.map(name).join(", ")}_`);
        parts.push("");
      }
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
        parts.push(`_Sources: ${b.result.packageIds.map(name).join(", ")}_`);
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
