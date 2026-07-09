"""Deterministic SQL normalization shared by every LLM-SQL entry point.

Both the agent's `run_sql` tool and the offline eval runner funnel
model-written SQL through `normalize_sql` before the guard sees it:
bare / placeholder `raw.rows` table references are rewritten to the
fully-qualified form, and literal JSONPath arguments of the JSON
functions get their non-identifier segments double-quoted. The guard
still runs after normalization and stays the authority on which tables
are allowed — normalization never widens what it accepts.

Span-based rewriting, not AST regeneration, on purpose: sqlglot's
generator reformats untouched SQL (comment style, keyword quoting), so
round-tripping the tree would make the evidence rail show SQL the
model never wrote even when nothing needed fixing. The masking pass
below gives the regex scanners an accurate 1:1 view instead.
"""
from __future__ import annotations

import re
from typing import Any

from semantic_enrich.config.settings import Settings
from semantic_enrich.providers.logging import get_logger

_LOG = get_logger("semantic_enrich.sql_normalize")


def normalize_sql(
    sql: str, *, settings: Settings
) -> tuple[str, dict[str, Any]]:
    """Apply every normalization pass. Returns `(sql, normalizations)`.

    `normalizations` is the model-facing report (`tables_rewritten`,
    `json_paths_quoted`); empty when the SQL needed nothing.
    """
    sql, tables_rewritten = normalize_table_references(sql, settings=settings)
    sql, json_paths_quoted = autoquote_json_paths(sql)
    normalizations: dict[str, Any] = {}
    if json_paths_quoted:
        normalizations["json_paths_quoted"] = json_paths_quoted
        _LOG.info(
            "agent_autoquote_applied",
            json_paths=json_paths_quoted,
            count=len(json_paths_quoted),
        )
    if tables_rewritten:
        normalizations["tables_rewritten"] = tables_rewritten
        _LOG.info(
            "agent_table_ref_normalized",
            tables=tables_rewritten,
            count=len(tables_rewritten),
        )
    return sql, normalizations


# Characters inside backtick identifiers that would confuse the span
# scanners (quote parity in the masker itself, paren/comma depth in
# `_second_arg_span`). Everything else — letters, dots, `<project>`
# angle brackets — must stay visible so the table-ref pattern can still
# match backticked references.
_BACKTICK_BLANKED = frozenset("'\"(),")


def _mask_string_literals(sql: str) -> str:
    """Blank string literals, comments, and hazardous identifier chars,
    preserving length.

    Positions line up 1:1 with the original, so regex spans found on
    the masked text can be applied back to the original. Handles:

    - `'…'` / `"…"` string literals (contents blanked; delimiters kept)
    - `'''…'''` / `\"\"\"…\"\"\"` triple-quoted strings (fully blanked)
    - `--` / `#` line comments and `/* … */` block comments (fully
      blanked — an apostrophe in a comment must never open a phantom
      string literal)
    - `` `…` `` identifiers: kept visible so table-ref matching works,
      but quote/paren/comma characters inside are blanked
    """
    out = list(sql)
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        if ch in ("'", '"'):
            if sql[i : i + 3] == ch * 3:  # triple-quoted string
                close = sql.find(ch * 3, i + 3)
                end = n if close == -1 else close + 3
                for k in range(i, end):
                    out[k] = " "
                i = end
                continue
            quote = ch
            i += 1
            while i < n and sql[i] != quote:
                if sql[i] == "\\" and i + 1 < n:
                    out[i] = " "
                    i += 1
                out[i] = " "
                i += 1
            i += 1  # closing quote (or end of input)
        elif ch == "`":
            i += 1
            while i < n and sql[i] != "`":
                if sql[i] in _BACKTICK_BLANKED:
                    out[i] = " "
                i += 1
            i += 1
        elif (ch == "-" and sql[i : i + 2] == "--") or ch == "#":
            while i < n and sql[i] != "\n":
                out[i] = " "
                i += 1
        elif ch == "/" and sql[i : i + 2] == "/*":
            close = sql.find("*/", i + 2)
            end = n if close == -1 else close + 2
            for k in range(i, end):
                out[k] = " "
            i = end
        else:
            i += 1
    return "".join(out)


def _table_ref_pattern(settings: Settings) -> re.Pattern[str]:
    raw = re.escape(settings.bq_dataset_raw)
    rows = re.escape(settings.bq_rows_table)
    placeholder = r"(?i:<project(?:_id)?>|PROJECT_ID)"
    ref = (
        # `<project>.raw.rows` / PROJECT_ID.raw.rows — any backtick mix.
        rf"`?{placeholder}`?\.`?{raw}`?\.`?{rows}`?"
        # `raw.rows` (backticked, no project).
        rf"|`{raw}\.{rows}`"
        # bare raw.rows (no project; FROM/JOIN anchor below rules out a
        # project-qualified reference matching here).
        rf"|{raw}\.{rows}\b"
    )
    return re.compile(rf"(?i:\b(FROM|JOIN))\s+({ref})")


def normalize_table_references(
    sql: str, *, settings: Settings
) -> tuple[str, list[str]]:
    """Rewrite bare / placeholder `raw.rows` references after FROM and
    JOIN to the fully-qualified form. Returns `(sql, rewritten_refs)`.

    Scans a literal-masked copy so string literals and comments are
    never touched."""
    project_id = settings.gcp_project_id
    if not project_id:
        return sql, []
    canonical = (
        f"`{project_id}.{settings.bq_dataset_raw}.{settings.bq_rows_table}`"
    )
    masked = _mask_string_literals(sql)
    pieces: list[str] = []
    rewritten: list[str] = []
    last = 0
    for m in _table_ref_pattern(settings).finditer(masked):
        start, end = m.span(2)
        pieces.append(sql[last:start])
        pieces.append(canonical)
        rewritten.append(sql[start:end])
        last = end
    pieces.append(sql[last:])
    return "".join(pieces), rewritten


# Longer names first so e.g. JSON_VALUE_ARRAY never half-matches as
# JSON_VALUE (the trailing `\s*\(` would then fail on the `_`).
_JSON_FUNC_RE = re.compile(
    r"(?i)\b(?:JSON_EXTRACT_STRING_ARRAY|JSON_EXTRACT_ARRAY"
    r"|JSON_EXTRACT_SCALAR|JSON_EXTRACT"
    r"|JSON_VALUE_ARRAY|JSON_VALUE"
    r"|JSON_QUERY_ARRAY|JSON_QUERY)\s*\("
)
_BARE_SEGMENT_OK_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def autoquote_json_paths(sql: str) -> tuple[str, list[str]]:
    """Double-quote non-identifier segments in literal JSONPath args of
    JSON function calls. Returns `(sql, original_paths_that_changed)`.

    Bare segments like `2020-21_Expenditures` silently return NULL for
    every row; the correct rewrite is unambiguous, so the tool applies
    it rather than bouncing the call back to the model. Anything the
    scanner can't parse confidently is left untouched — the guard
    remains the rejection path for exotic shapes."""
    masked = _mask_string_literals(sql)
    replacements: list[tuple[int, int, str, str]] = []
    for m in _JSON_FUNC_RE.finditer(masked):
        span = _second_arg_span(masked, m.end() - 1)
        if span is None:
            continue
        start, end = span
        arg = sql[start:end]
        stripped = arg.strip()
        if (
            len(stripped) < 2
            or stripped[0] not in ("'", '"')
            or stripped[-1] != stripped[0]
        ):
            continue  # not a string literal (parameter, concatenation)
        delim = stripped[0]
        path = stripped[1:-1]
        if "\\" in path or "'" in path:
            continue  # escapes / would break the single-quoted rewrite
        if delim == '"' and '"' in path:
            continue  # embedded quotes — not confidently parseable
        quoted = _quote_json_path(path)
        if quoted is None:
            continue
        lead = start + arg.index(stripped)
        # The rewrite is always emitted as a single-quoted literal so
        # the JSONPath's own `"` segment quotes never need escaping.
        replacements.append(
            (lead, lead + len(stripped), f"'{quoted}'", path)
        )
    if not replacements:
        return sql, []
    # A nested JSON call's path arg can sit before its enclosing call's
    # second arg — apply replacements in position order.
    replacements.sort(key=lambda r: r[0])
    pieces = []
    changed_paths = []
    last = 0
    for start, end, text, path in replacements:
        pieces.append(sql[last:start])
        pieces.append(text)
        changed_paths.append(path)
        last = end
    pieces.append(sql[last:])
    return "".join(pieces), changed_paths


def _second_arg_span(sql: str, open_paren: int) -> tuple[int, int] | None:
    """Span of a call's second top-level argument, or None when the call
    has fewer or more than exactly two arguments (the JSON functions we
    rewrite take exactly two) or runs off the end. Tracks paren depth
    and skips string literals; callers pass the MASKED text so comments
    and string contents can't desync the scan."""
    i = open_paren + 1
    n = len(sql)
    depth = 1
    arg_index = 0
    arg_start = i
    while i < n:
        ch = sql[i]
        if ch == "'" or ch == '"':
            quote = ch
            i += 1
            while i < n and sql[i] != quote:
                if sql[i] == "\\":
                    i += 1
                i += 1
            if i >= n:
                return None
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return (arg_start, i) if arg_index == 1 else None
        elif ch == "," and depth == 1:
            if arg_index == 1:
                return None  # a third argument — leave it to the guard
            arg_index += 1
            arg_start = i + 1
        i += 1
    return None


def _quote_json_path(path: str) -> str | None:
    """Rewrite a JSONPath so every non-identifier segment is quoted.

    Returns None when nothing changed or the path can't be parsed
    confidently (array subscripts, embedded quotes, empty segments)."""
    if not path.startswith("$") or "[" in path:
        return None
    segments: list[str] = []
    changed = False
    i = 1
    n = len(path)
    while i < n:
        if path[i] != ".":
            return None
        i += 1
        if i < n and path[i] == '"':
            close = path.find('"', i + 1)
            if close == -1:
                return None
            segments.append(path[i : close + 1])
            i = close + 1
        else:
            j = i
            while j < n and path[j] != ".":
                if path[j] == '"':
                    return None
                j += 1
            segment = path[i:j]
            if not segment:
                return None
            if _BARE_SEGMENT_OK_RE.fullmatch(segment):
                segments.append(segment)
            else:
                segments.append(f'"{segment}"')
                changed = True
            i = j
    if not changed:
        return None
    return "$" + "".join(f".{s}" for s in segments)
