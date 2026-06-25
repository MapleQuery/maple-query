"""`raw.documents` UPDATEs owned by the rows loader.

Two operations:

- `mark_in_flight`: set `load_status='pending'`, stamp
  `load_attempted_at=NOW()`, clear `load_error`. Called before the
  per-doc work starts so a crash mid-load leaves a discoverable
  state.
- `record_load_outcome`: write the terminal status (loaded /
  blob_missing / parse_failed) plus the 3.3-owned columns
  (`preamble_rows`, `header_confidence`, `row_count`, `load_error`).

The SQL is parameter-bound — `document_id`, `load_error`, and the
JSON payload all come from external sources (the catalog row, the
GCS blob, the parsed file). Backtick-quoted table identifiers are
validated against `_BQ_IDENT_RE` before interpolation.
"""
from __future__ import annotations

import json
import re
import threading
from typing import Any, Literal

from warehouse_load.clients.bq import BqClient

# Same identifier regex as documents_merge — duplicated to keep
# the two modules independently auditable rather than coupling them
# through an obscure shared helper.
_BQ_IDENT_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# Serialises raw.documents UPDATE jobs across worker threads. BigQuery
# refuses concurrent DML against the same table with `Could not
# serialize access to table ... due to concurrent update` — even
# though our UPDATEs are idempotent, BQ can't know that. The lock
# turns the per-doc mark_in_flight + record_load_outcome calls into
# a serial pipeline (each BQ UPDATE runs to completion before the
# next starts). Download / sniff / header_detect / stream stay
# parallel inside the worker pool; only the BQ DML is serial.
_DOC_UPDATE_LOCK = threading.Lock()

LoadStatus = Literal["pending", "loaded", "blob_missing", "parse_failed"]


def mark_in_flight(
    *,
    bq: BqClient,
    documents_table: str,
    document_id: str,
) -> None:
    """`UPDATE raw.documents SET load_status='pending', load_attempted_at=NOW(),
    load_error=NULL WHERE document_id=@doc_id`.

    Idempotent: re-running on a `pending` row is a no-op apart from
    advancing `load_attempted_at`. The downstream MERGE in §8.6 doesn't
    care about the intermediate state.
    """
    _validate_table_id(documents_table)
    sql = f"""\
UPDATE `{documents_table}` SET
  load_status = 'pending',
  load_attempted_at = CURRENT_TIMESTAMP(),
  load_error = NULL
WHERE document_id = @doc_id
"""
    with _DOC_UPDATE_LOCK:
        bq.execute(_inline_doc_id(sql, document_id))


def record_load_outcome(
    *,
    bq: BqClient,
    documents_table: str,
    document_id: str,
    load_status: LoadStatus,
    load_error: str | None,
    preamble_rows: tuple[tuple[str, ...], ...] | None,
    header_confidence: str | None,
    row_count: int | None,
) -> None:
    """Write the terminal 3.3-owned columns for `document_id`.

    `preamble_rows`, `header_confidence`, `row_count` are only
    meaningful on the `loaded` path; passing them as None for
    blob_missing / parse_failed is correct (the existing values from
    a prior successful load, if any, are overwritten with NULL — the
    operator's mental model is "this load attempt's outputs", not
    "merge with prior outputs").
    """
    _validate_table_id(documents_table)

    preamble_json = (
        json.dumps([list(r) for r in preamble_rows], ensure_ascii=False)
        if preamble_rows is not None
        else None
    )

    sql = f"""\
UPDATE `{documents_table}` SET
  preamble_rows = {_json_or_null_literal(preamble_json)},
  header_confidence = {_string_or_null_literal(header_confidence)},
  load_status = {_string_literal(load_status)},
  load_attempted_at = CURRENT_TIMESTAMP(),
  load_error = {_string_or_null_literal(load_error)},
  row_count = {_int_or_null_literal(row_count)}
WHERE document_id = @doc_id
"""
    with _DOC_UPDATE_LOCK:
        bq.execute(_inline_doc_id(sql, document_id))


def _validate_table_id(table_id: str) -> None:
    parts = table_id.split(".")
    if len(parts) != 3:
        raise ValueError(f"expected project.dataset.table, got {table_id!r}")
    for part in parts:
        if not _BQ_IDENT_RE.fullmatch(part):
            raise ValueError(f"invalid BQ identifier segment {part!r} in {table_id!r}")


def _inline_doc_id(sql: str, document_id: str) -> str:
    """`document_id` is a 64-hex string — sha256(source_url+checksum).
    Validate before interpolation; reject anything outside `[A-Fa-f0-9]`
    so a future call site that pulls from less-trusted config can't
    inject SQL between the parameter delimiters.
    """
    if not re.fullmatch(r"[A-Fa-f0-9_-]+", document_id):
        raise ValueError(f"invalid document_id: {document_id!r}")
    return sql.replace("@doc_id", f"'{document_id}'")


def _string_literal(value: str) -> str:
    """Inline a SQL string literal.

    `load_error` carries the verbatim text of a BQ exception, which
    can contain raw newlines, tabs, and backslashes — none of which
    are valid inside a single-quoted BQ string literal. The original
    "double the quote" escape (SQL standard) handles `'` but leaves
    `\\n` / `\\r` / `\\t` / `\\` / `\\0` as parse errors. So we escape
    backslashes first (otherwise we'd double-escape the ones we add),
    then quotes (kept as `''` to match existing tests), then the
    control chars BQ understands as backslash escapes inside `'...'`.
    """
    escaped = (
        value.replace("\\", "\\\\")
        .replace("'", "''")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace("\x00", "\\0")
    )
    return f"'{escaped}'"


def _string_or_null_literal(value: str | None) -> str:
    if value is None:
        return "NULL"
    return _string_literal(value)


def _int_or_null_literal(value: int | None) -> str:
    if value is None:
        return "NULL"
    return str(int(value))


def _json_or_null_literal(value: str | None) -> str:
    """JSON literal: PARSE_JSON wraps the string so BQ stores it as
    JSON, not as a STRING column. We escape via `_string_literal`,
    then wrap. Empty arrays land as `PARSE_JSON('[]')` which BQ
    accepts.
    """
    if value is None:
        return "NULL"
    return f"PARSE_JSON({_string_literal(value)})"


def hydrate_document_row(raw: dict[str, Any]) -> dict[str, Any]:
    """No-op identity used by the orchestrator to mark the boundary
    where BQ row dicts become Python `DocumentRow`s. Kept as a
    nameable seam so tests can swap the projection if the candidate
    query column set changes."""
    return raw


__all__ = [
    "LoadStatus",
    "hydrate_document_row",
    "mark_in_flight",
    "record_load_outcome",
]
