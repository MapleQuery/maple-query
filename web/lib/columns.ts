/**
 * Recognising the columns whose names the loader had to invent.
 *
 * When a source file puts a title or a blank line above its header row,
 * the CSV reader takes that first line as the header, finds it empty,
 * and numbers the columns positionally instead: `__col_1`, `__col_2`.
 * The values are fine — only the naming is lost. This is the one
 * definition of "unnamed" on the client, and it matches the pattern the
 * loader writes and the recovery detector keys off.
 *
 * The agent recovers real names at query time where it can, but that
 * happens inside a turn and is never written back to the enriched column
 * table — so the browsing surfaces still read the generated names, and
 * have to say so rather than present them as the column's real name.
 */

const GENERATED_COLUMN_RE = /^__col_\d+$/;

export function isGeneratedColumnName(name: string): boolean {
  return GENERATED_COLUMN_RE.test(name);
}

export function countGeneratedColumns(
  columns: { column_name: string }[],
): number {
  return columns.filter((c) => isGeneratedColumnName(c.column_name)).length;
}
