/**
 * Assemble one SQL execution's result set from the two frames that carry
 * it.
 *
 * `run_sql` emits `sql_executed` with the first few rows, so something
 * can render the moment the query returns, and then `rows` with the
 * whole set. Seeding from the first and appending the second duplicates
 * the preview: every result repeated its top rows, and the repeat sat
 * above the fold where it was most visible. A hundred-row `GROUP BY`
 * rendered as "first 20 of 103", with the three largest groups counted
 * twice.
 *
 * So the preview is treated as exactly that — rows held only until the
 * real set arrives. The first `rows` frame for a call replaces it and
 * takes ownership; later frames for the same call append, since `is_last`
 * may be false. `sql_executed` releases ownership again, which is what
 * keeps a second execution in the same turn from appending onto the
 * first: that frame carries no call id of its own to key on.
 */

export interface ResultRows {
  rows: Record<string, unknown>[];
  /**
   * The `sql_call_id` these rows belong to, or `null` while they are
   * still the preview from `sql_executed`.
   */
  ownerCallId: string | null;
}

/**
 * Owner for a `rows` frame that arrived without a `sql_call_id`. The
 * field is optional in the event schema, and such a frame still has to
 * be distinguishable from the preview. Call ids are hex, so the
 * parentheses keep this from ever colliding with a real one.
 */
const UNKEYED = "(unkeyed)";

export const EMPTY_RESULT_ROWS: ResultRows = { rows: [], ownerCallId: null };

/** Hold `sql_executed`'s sample until the real result set arrives. */
export function seedPreviewRows(
  sample: Record<string, unknown>[] | undefined,
): ResultRows {
  return { rows: sample ? [...sample] : [], ownerCallId: null };
}

/**
 * Fold a `rows` frame in: replace the preview on the first frame of a
 * call, append on the rest.
 */
export function mergeRowsFrame(
  current: ResultRows,
  frame: { sql_call_id?: string; rows: Record<string, unknown>[] },
): ResultRows {
  const owner = frame.sql_call_id || UNKEYED;
  if (current.ownerCallId !== owner) {
    return { rows: [...frame.rows], ownerCallId: owner };
  }
  return { rows: [...current.rows, ...frame.rows], ownerCallId: owner };
}
