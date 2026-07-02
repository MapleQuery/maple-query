"""SELECT-only, single-statement, whitelisted, cost-capped SQL guard.

The safety story of the harness: every failure here is terminal for the
question, no retry, no auto-fix beyond the deterministic LIMIT wrapper.
Auto-fixing hallucinated column names would mask the prompt failures
the operator needs to see.

Structural checks walk the sqlglot AST. Textual belt-and-braces catches
the two things AST walks miss cheaply:
  - forbidden keywords the AST swallows inside CTE aliases or comments
  - a trailing `;` followed by anything non-whitespace (multi-statement
    guard the AST wouldn't refuse if the second statement is empty).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import sqlglot
import sqlglot.expressions as exp
from sqlglot.errors import ParseError

from semantic_enrich.clients.bq import BqClient
from semantic_enrich.config.settings import Settings

_DIALECT = "bigquery"

# Length gate on the model's raw SQL. `strict: true` Structured Outputs
# doesn't enforce minLength; a stub-length or absurdly long response is
# handled here. Upper bound is generous — real per-package aggregates
# with a worked JSON_VALUE chain sit at ~2 KB; 20 KB catches a runaway.
_MIN_SQL_LEN = 20
_MAX_SQL_LEN = 20_000


@dataclass(frozen=True)
class GuardResult:
    """Outcome of the guard pipeline.

    `sql_final` is the model's SQL, possibly wrapped in a LIMIT-100
    subquery; the executor runs this verbatim. `dry_run_bytes` /
    `dry_run_error` come from `BqClient.dry_run_bytes` when the guard
    reached that step (all structural checks passed).
    """

    accepted: bool
    reason: str | None
    sql_final: str
    dry_run_bytes: int | None
    dry_run_error: str | None
    limit_wrapped: bool


def guard(*, sql: str, bq: BqClient, settings: Settings) -> GuardResult:
    """Run the full guard pipeline; first failure wins."""
    sql = sql.strip().rstrip(";").strip()
    if len(sql) < _MIN_SQL_LEN:
        return _rejected("sql_invalid: sql too short", sql)
    if len(sql) > _MAX_SQL_LEN:
        return _rejected("sql_invalid: sql too long", sql)

    forbidden = _forbidden_keyword(sql, settings)
    if forbidden is not None:
        return _rejected(f"sql_forbidden_keyword: {forbidden}", sql)

    if _looks_multi_statement(sql):
        return _rejected("sql_multi_statement", sql)

    try:
        trees = sqlglot.parse(sql, dialect=_DIALECT)
    except ParseError as exc:
        return _rejected(f"sql_parse_error: {exc}", sql)
    if len(trees) != 1:
        return _rejected("sql_multi_statement", sql)
    tree = trees[0]
    if tree is None:
        return _rejected("sql_parse_error: empty parse", sql)

    if not _is_select_root(tree):
        return _rejected("sql_not_select", sql)

    dataset_violation = _dataset_violation(tree, settings)
    if dataset_violation is not None:
        return _rejected(dataset_violation, sql)

    project_violation = _project_violation(tree, settings)
    if project_violation is not None:
        return _rejected(project_violation, sql)

    sql_final, limit_wrapped = _ensure_limit(tree, sql, settings)

    try:
        dry_run_bytes = bq.dry_run_bytes(
            sql_final, timeout_ms=settings.eval_dry_run_timeout_ms
        )
    except Exception as exc:
        # Surface the raw BQ error verbatim so the report shows exactly
        # what BQ said, rather than a swallowed / re-typed message.
        return GuardResult(
            accepted=False,
            reason=f"sql_dry_run_failed: {exc}",
            sql_final=sql_final,
            dry_run_bytes=None,
            dry_run_error=str(exc),
            limit_wrapped=limit_wrapped,
        )

    if dry_run_bytes > settings.eval_max_bytes_billed:
        return GuardResult(
            accepted=False,
            reason=(
                f"sql_cost_too_high: {dry_run_bytes} > "
                f"{settings.eval_max_bytes_billed}"
            ),
            sql_final=sql_final,
            dry_run_bytes=dry_run_bytes,
            dry_run_error=None,
            limit_wrapped=limit_wrapped,
        )

    return GuardResult(
        accepted=True,
        reason=None,
        sql_final=sql_final,
        dry_run_bytes=dry_run_bytes,
        dry_run_error=None,
        limit_wrapped=limit_wrapped,
    )


def _rejected(reason: str, sql: str) -> GuardResult:
    return GuardResult(
        accepted=False,
        reason=reason,
        sql_final=sql,
        dry_run_bytes=None,
        dry_run_error=None,
        limit_wrapped=False,
    )


def _forbidden_keyword(sql: str, settings: Settings) -> str | None:
    """Word-boundary regex against the upper-cased SQL. Word boundaries
    keep `INSERT_DATE` (a column name) from false-positiving on
    `INSERT`."""
    upper = sql.upper()
    for kw in settings.eval_forbidden_keywords:
        if re.search(rf"\b{re.escape(kw)}\b", upper):
            return kw
    return None


def _looks_multi_statement(sql: str) -> bool:
    """Textual multi-statement guard.

    Trailing `;`s were stripped before we got here; any remaining `;`
    followed by non-whitespace is a second statement the AST walk might
    accept if the second statement itself parses to a Select."""
    return any(
        ch == ";" and sql[i + 1 :].strip() for i, ch in enumerate(sql)
    )


def _is_select_root(tree: exp.Expression) -> bool:
    """AST root must be a `Select`, or a `With` whose body is a
    `Select` (CTE query)."""
    if isinstance(tree, exp.Select):
        return True
    if isinstance(tree, exp.With):
        return isinstance(tree.this, exp.Select)
    # Some queries parse as `Union`, `Subquery`, etc. Reject all of them
    # for now; the harness's canonical query shape is a single SELECT
    # (possibly with CTEs), which matches every fixture question.
    return False


def _dataset_violation(
    tree: exp.Expression, settings: Settings
) -> str | None:
    """Walk every `Table` node and reject if its dataset isn't in the
    allow-list. CTE aliases and other non-Table sources
    (UNNEST, TABLE @param) skip the check by construction because they
    don't have a `db` component."""
    cte_names = _cte_names(tree)
    allowed = set(settings.eval_allowed_datasets)
    for table in tree.find_all(exp.Table):
        # A CTE reference has no db (dataset) segment; skip it.
        dataset = table.args.get("db")
        if dataset is None:
            # Unqualified reference — if it's a CTE alias, allow; else
            # reject with a specific reason so the operator sees the
            # missing dataset qualifier.
            if table.name in cte_names:
                continue
            return f"sql_dataset_not_allowed: <unqualified>({table.name})"
        dataset_name = dataset.name if hasattr(dataset, "name") else str(dataset)
        if dataset_name not in allowed:
            return f"sql_dataset_not_allowed: {dataset_name}"
    return None


def _project_violation(
    tree: exp.Expression, settings: Settings
) -> str | None:
    """Reject an explicit project id that doesn't match the harness's
    configured project. Unqualified project (BQ falls back to the
    session default) is fine."""
    for table in tree.find_all(exp.Table):
        catalog = table.args.get("catalog")
        if catalog is None:
            continue
        project_name = catalog.name if hasattr(catalog, "name") else str(catalog)
        if (
            settings.gcp_project_id
            and project_name != settings.gcp_project_id
        ):
            return f"sql_wrong_project: {project_name}"
    return None


def _cte_names(tree: exp.Expression) -> set[str]:
    names: set[str] = set()
    for cte in tree.find_all(exp.CTE):
        alias = cte.args.get("alias")
        if alias is not None and hasattr(alias, "name"):
            names.add(alias.name)
    return names


def _ensure_limit(
    tree: exp.Expression, original_sql: str, settings: Settings
) -> tuple[str, bool]:
    """Return `(sql_final, limit_wrapped)`.

    - `LIMIT <= row_limit` present → pass through the original SQL.
    - Missing LIMIT or `LIMIT > row_limit` → wrap in a subquery with
      `LIMIT row_limit`.

    The wrapper never rewrites the inner SQL (auto-fixing hallucinated
    identifiers would mask prompt failures the report needs to surface);
    it just clamps rows."""
    row_limit = settings.eval_row_limit

    inner_select = tree.this if isinstance(tree, exp.With) else tree
    limit_node = None
    if isinstance(inner_select, exp.Select):
        limit_node = inner_select.args.get("limit")
    if limit_node is not None:
        limit_expr = limit_node.expression if hasattr(
            limit_node, "expression"
        ) else None
        if isinstance(limit_expr, exp.Literal) and limit_expr.is_int:
            existing = int(limit_expr.name)
            if existing <= row_limit:
                return original_sql, False

    wrapped = f"SELECT * FROM ( {original_sql} ) AS _wrap LIMIT {row_limit}"
    return wrapped, True
