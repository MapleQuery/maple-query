"""Reading and rewriting the document/column references in model SQL.

Two jobs, both textual, both sitting between what the model wrote and
what the guard validates.

**Translation.** A model shown a recovered column name will write it.
`JSON_VALUE(r.row, '$."Total Amount ($000)"')` has to become
`JSON_VALUE(r.row, '$.__col_3')` before anything runs, because that is
what the stored row actually holds.

This happens *before* the guard, not after. The guard's job is to
validate the SQL that will really execute — but more concretely, a
JSONPath to a key that does not exist does not fail in BigQuery, it
returns NULL. A missed translation would therefore sail through the
guard's dry run and produce a silent all-NULL aggregate: a number that
is wrong with nothing anywhere saying so. Ordering is the safety
property here, not a stylistic preference.

**Preamble exclusion.** The recovered header row is still a row, as are
the blank lines above it. They mostly behave: their values are text, so
`SAFE_CAST(... AS FLOAT64)` yields NULL and they drop out of SUM and AVG
on their own. They do not drop out of `COUNT(*)`, where they inflate the
answer by the preamble depth — invisible on a table of thousands, and
material on one with twenty rows.

Rewriting is span-based rather than AST regeneration, the same posture
`sql_normalize` takes and for the same reason: sqlglot's generator
reformats untouched SQL, and the evidence rail would show the user SQL
nobody wrote. sqlglot is used here only to *inspect* shape, never to
regenerate.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

import sqlglot
import sqlglot.expressions as exp

# `$.<key>` — the top-level JSONPath key, either bare or double-quoted.
# Bare keys allow leading digits and hyphens here because we want to
# CATCH those references (the prompt tells the model to quote them,
# but this regex is the safety net for when the model doesn't). If a
# bare `$.2020-21_Foo` sneaks through we still want to know the model
# intended the key `2020-21_Foo` so we can pairing-check it.
JSONPATH_TOP_KEY_RE = re.compile(
    r"""\$\.(?:"([^"]+)"|([A-Za-z0-9_][A-Za-z0-9_\-]*))"""
)

# Extract literal `document_id IN ('a', 'b', ...)` predicates. Only the
# literal shape counts — a subquery IN or a JOIN is already rejected
# by the sql_guard, so we don't need to defend against them here.
DOC_IDS_IN_RE = re.compile(
    r"""document_id\s+IN\s*\(([^)]+)\)""", re.IGNORECASE
)
_ID_LITERAL_RE = re.compile(r"""['"]([^'"]+)['"]""")

# Document ids are warehouse checksums. Anything else does not get
# interpolated into a predicate.
_SAFE_DOC_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def extract_json_path_columns(sql: str) -> set[str]:
    keys: set[str] = set()
    for m in JSONPATH_TOP_KEY_RE.finditer(sql):
        key = m.group(1) or m.group(2)
        if key:
            keys.add(key)
    return keys


def extract_inlined_document_ids(sql: str) -> set[str]:
    ids: set[str] = set()
    for m in DOC_IDS_IN_RE.finditer(sql):
        inner = m.group(1)
        for id_m in _ID_LITERAL_RE.finditer(inner):
            ids.add(id_m.group(1))
    return ids


@dataclass(frozen=True)
class AliasConflict:
    """One recovered name that means different columns in different
    documents the query inlined."""

    name: str
    keys_by_document: dict[str, str]


@dataclass(frozen=True)
class AliasResult:
    sql: str
    translated: dict[str, str] = field(default_factory=dict)
    """recovered name -> the positional key it was rewritten to."""

    conflicts: tuple[AliasConflict, ...] = ()
    """Non-empty means nothing was translated and the caller must refuse."""

    preamble_excluded: dict[str, int] = field(default_factory=dict)
    """doc_id -> header row index now excluded from the query."""

    preamble_skipped: bool = False
    """True when a known preamble could not be excluded safely, so the
    caller can say so rather than quietly returning inflated counts."""


def apply_recovered_names(
    sql: str,
    *,
    recovered_names: Mapping[str, Mapping[str, str]],
    header_rows: Mapping[str, int],
    doc_columns: Mapping[str, list[str]] | None = None,
) -> AliasResult:
    """Rewrite recovered column names to positional keys and drop the
    preamble rows, for the documents this SQL actually inlines.

    With header recovery off there is nothing on state, so both passes
    are no-ops by construction rather than by a second feature flag.
    """
    if not recovered_names and not header_rows:
        return AliasResult(sql=sql)

    scoped_docs = extract_inlined_document_ids(sql)
    if not scoped_docs:
        return AliasResult(sql=sql)

    referenced = extract_json_path_columns(sql)
    resolutions = _resolve(
        referenced=referenced,
        scoped_docs=scoped_docs,
        recovered_names=recovered_names,
        doc_columns=doc_columns or {},
    )
    conflicts = tuple(
        AliasConflict(name=name, keys_by_document=by_doc)
        for name, by_doc in sorted(resolutions.items())
        if len(set(by_doc.values())) > 1
    )
    if conflicts:
        # Declining beats picking a winner, for the same reason the
        # detector declines: a rejected query announces itself and a
        # wrong column does not.
        return AliasResult(sql=sql, conflicts=conflicts)

    translated = {
        name: next(iter(by_doc.values()))
        for name, by_doc in resolutions.items()
        # A name that already resolves to itself is a real column, not
        # something to rewrite.
        if next(iter(by_doc.values())) != name
    }
    rewritten = _rewrite_json_paths(sql, translated) if translated else sql

    rewritten, excluded, skipped = _exclude_preamble(
        rewritten,
        header_rows={
            d: k for d, k in header_rows.items() if d in scoped_docs and k > 0
        },
    )
    return AliasResult(
        sql=rewritten,
        translated=translated,
        preamble_excluded=excluded,
        preamble_skipped=skipped,
    )


def _resolve(
    *,
    referenced: set[str],
    scoped_docs: set[str],
    recovered_names: Mapping[str, Mapping[str, str]],
    doc_columns: Mapping[str, list[str]],
) -> dict[str, dict[str, str]]:
    """name -> {doc_id: the key that name means in that document}.

    Only names the SQL actually references, and only documents it
    inlines. A document where the name is already a real column resolves
    to the name itself, which is how a real `Total` in one document and
    a recovered `Total` in another get caught as a conflict rather than
    silently rewriting both.
    """
    out: dict[str, dict[str, str]] = {}
    for doc_id in scoped_docs:
        by_name = {
            name: key
            for key, name in recovered_names.get(doc_id, {}).items()
        }
        real = set(doc_columns.get(doc_id, []))
        for name in referenced:
            if name in by_name:
                out.setdefault(name, {})[doc_id] = by_name[name]
            elif name in real:
                out.setdefault(name, {})[doc_id] = name
    return out


def _rewrite_json_paths(sql: str, translated: Mapping[str, str]) -> str:
    """Replace `$.<recovered name>` with `$.<positional key>` in place.

    Span-based: only the matched JSONPath segment changes, so the rest
    of the model's SQL survives byte-for-byte.
    """

    def _sub(match: re.Match[str]) -> str:
        key = match.group(1) or match.group(2)
        replacement = translated.get(key or "")
        if replacement is None:
            return match.group(0)
        return f"$.{replacement}"

    return JSONPATH_TOP_KEY_RE.sub(_sub, sql)


def _exclude_preamble(
    sql: str, *, header_rows: Mapping[str, int]
) -> tuple[str, dict[str, int], bool]:
    """AND a per-document `row_index` floor onto the existing filter.

    The predicate is written per document — `NOT (document_id = 'x' AND
    row_index <= 2)` — rather than as a bare `row_index > 2`, so it is a
    tautology for any document without a known header row and stays
    correct when one query inlines several documents with different
    preamble depths. That also makes it safe to append after every
    `document_id IN (...)` predicate, including each arm of a per-doc
    UNION ALL split.
    """
    usable = {
        doc_id: index
        for doc_id, index in sorted(header_rows.items())
        if _SAFE_DOC_ID_RE.fullmatch(doc_id)
    }
    if not usable:
        return sql, {}, False
    if not _where_is_pure_conjunction(sql):
        # An OR in the WHERE means appending after the IN-predicate span
        # could land inside a disjunct and change what the query means.
        # Report it instead of guessing.
        return sql, {}, True

    predicate = "".join(
        f" AND NOT (document_id = '{doc_id}' AND row_index <= {index})"
        for doc_id, index in usable.items()
    )
    out: list[str] = []
    cursor = 0
    for match in DOC_IDS_IN_RE.finditer(sql):
        out.append(sql[cursor : match.end()])
        out.append(predicate)
        cursor = match.end()
    if cursor == 0:
        return sql, {}, True
    out.append(sql[cursor:])
    return "".join(out), usable, False


def _where_is_pure_conjunction(sql: str) -> bool:
    """Whether every WHERE clause in the statement is an AND-chain.

    Conservative: an OR anywhere in any WHERE disqualifies the whole
    statement, even one in an unrelated subquery. The cost of being
    wrong here is a silently altered predicate, so the check errs toward
    doing nothing.
    """
    try:
        tree = sqlglot.parse_one(sql, dialect="bigquery")
    except Exception:
        return False
    if tree is None:
        return False
    return all(
        next(where.find_all(exp.Or), None) is None
        for where in tree.find_all(exp.Where)
    )
