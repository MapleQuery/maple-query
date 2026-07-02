"""§16.1 unit tests for the SQL guard.

Every guard rule round-tripped:
- forbidden keywords (word boundaries, so INSERT_DATE column is fine)
- multi-statement (textual + AST)
- non-SELECT root
- dataset whitelist (unqualified refs rejected; CTE aliases skip check)
- project whitelist
- missing LIMIT → auto-wrap
- LIMIT > row_limit → auto-wrap
- dry-run bytes cap
"""
from __future__ import annotations

import pytest

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.sql_guard import guard


class _FakeBq:
    """Minimal BqClient stand-in — only implements `dry_run_bytes`.
    Tests set `bytes_processed` (or `raise_exc`) per instance."""

    def __init__(
        self, *, bytes_processed: int = 100_000_000, raise_exc: Exception | None = None
    ) -> None:
        self.bytes_processed = bytes_processed
        self.raise_exc = raise_exc
        self.calls = 0

    def dry_run_bytes(self, sql: str, *, params: object = (), timeout_ms: int) -> int:
        self.calls += 1
        if self.raise_exc:
            raise self.raise_exc
        return self.bytes_processed


def _settings() -> Settings:
    return Settings(gcp_project_id="proj")


def test_select_accepted() -> None:
    sql = "SELECT 1 AS n FROM `proj.raw.rows` LIMIT 10"
    result = guard(sql=sql, bq=_FakeBq(), settings=_settings())
    assert result.accepted
    assert result.limit_wrapped is False
    assert result.sql_final == sql
    assert result.dry_run_bytes == 100_000_000


def test_missing_limit_gets_wrapped() -> None:
    sql = "SELECT * FROM `proj.raw.documents`"
    result = guard(sql=sql, bq=_FakeBq(), settings=_settings())
    assert result.accepted
    assert result.limit_wrapped is True
    assert result.sql_final.endswith("LIMIT 100")


def test_limit_over_row_limit_gets_wrapped() -> None:
    sql = "SELECT * FROM `proj.raw.documents` LIMIT 5000"
    result = guard(sql=sql, bq=_FakeBq(), settings=_settings())
    assert result.accepted
    assert result.limit_wrapped is True
    assert "LIMIT 100" in result.sql_final


def test_forbidden_keyword_rejected() -> None:
    sql = "INSERT INTO `proj.raw.rows` (a) VALUES (1)"
    result = guard(sql=sql, bq=_FakeBq(), settings=_settings())
    assert not result.accepted
    assert result.reason is not None
    assert result.reason.startswith("sql_forbidden_keyword: INSERT")


def test_insert_date_column_not_false_positive() -> None:
    """Word boundaries: INSERT_DATE as a column name is fine."""
    sql = "SELECT INSERT_DATE FROM `proj.raw.documents` LIMIT 10"
    result = guard(sql=sql, bq=_FakeBq(), settings=_settings())
    assert result.accepted, result.reason


def test_dataset_not_allowed() -> None:
    sql = "SELECT 1 FROM `proj.other.foo` LIMIT 10"
    result = guard(sql=sql, bq=_FakeBq(), settings=_settings())
    assert not result.accepted
    assert result.reason is not None
    assert result.reason.startswith("sql_dataset_not_allowed: other")


def test_wrong_project_rejected() -> None:
    sql = "SELECT 1 FROM `wrong.raw.rows` LIMIT 10"
    result = guard(sql=sql, bq=_FakeBq(), settings=_settings())
    assert not result.accepted
    assert result.reason is not None
    assert result.reason.startswith("sql_wrong_project: wrong")


def test_multi_statement_textual_rejected() -> None:
    sql = "SELECT 1 FROM `proj.raw.rows` LIMIT 1; SELECT 2 FROM `proj.raw.rows`"
    result = guard(sql=sql, bq=_FakeBq(), settings=_settings())
    assert not result.accepted
    assert result.reason == "sql_multi_statement"


def test_cost_too_high_rejected() -> None:
    sql = "SELECT * FROM `proj.raw.rows` LIMIT 10"
    bq = _FakeBq(bytes_processed=100 * 1024 * 1024 * 1024)  # 100 GB
    result = guard(sql=sql, bq=bq, settings=_settings())
    assert not result.accepted
    assert result.reason is not None
    assert result.reason.startswith("sql_cost_too_high")


def test_dry_run_failure_captured() -> None:
    sql = "SELECT * FROM `proj.raw.rows` LIMIT 10"
    bq = _FakeBq(raise_exc=RuntimeError("bad SQL"))
    result = guard(sql=sql, bq=bq, settings=_settings())
    assert not result.accepted
    assert result.reason is not None
    assert result.reason.startswith("sql_dry_run_failed")
    assert result.dry_run_error == "bad SQL"


def test_stub_sql_rejected() -> None:
    result = guard(sql="SELECT 1", bq=_FakeBq(), settings=_settings())
    assert not result.accepted
    assert result.reason == "sql_invalid: sql too short"


def test_absurdly_long_sql_rejected() -> None:
    padding = "x" * 25_000
    sql = f"SELECT 1 AS a {padding} FROM `proj.raw.rows` LIMIT 10"
    result = guard(sql=sql, bq=_FakeBq(), settings=_settings())
    assert not result.accepted
    assert result.reason == "sql_invalid: sql too long"


def test_cte_alias_skips_dataset_check() -> None:
    sql = (
        "WITH t AS (SELECT 1 AS n FROM `proj.raw.rows`) "
        "SELECT * FROM t LIMIT 10"
    )
    result = guard(sql=sql, bq=_FakeBq(), settings=_settings())
    assert result.accepted, result.reason


def test_parse_error_rejected() -> None:
    result = guard(
        sql="SELECT FROM WHERE WHERE WHERE",
        bq=_FakeBq(),
        settings=_settings(),
    )
    assert not result.accepted
    assert result.reason is not None


def test_non_select_root_rejected() -> None:
    # `DROP` is caught by the keyword regex first — cover the AST path
    # with something the regex misses: `EXPLAIN`.
    sql = "EXPLAIN SELECT 1 FROM `proj.raw.rows` LIMIT 1"
    result = guard(sql=sql, bq=_FakeBq(), settings=_settings())
    assert not result.accepted


def test_unqualified_table_rejected() -> None:
    sql = "SELECT 1 FROM rows LIMIT 10"
    result = guard(sql=sql, bq=_FakeBq(), settings=_settings())
    assert not result.accepted
    assert result.reason is not None
    assert result.reason.startswith("sql_dataset_not_allowed: <unqualified>")


def test_union_not_select_rejected() -> None:
    """AST root that isn't SELECT / WITH-SELECT — e.g. a set-operation
    root — rejects with `sql_not_select`."""
    sql = (
        "(SELECT 1 FROM `proj.raw.rows`) UNION ALL "
        "(SELECT 2 FROM `proj.raw.rows`) LIMIT 10"
    )
    result = guard(sql=sql, bq=_FakeBq(), settings=_settings())
    assert not result.accepted
    assert result.reason == "sql_not_select"


@pytest.mark.parametrize(
    "kw",
    ["UPDATE", "DELETE", "MERGE", "CREATE", "DROP", "ALTER", "GRANT",
     "REVOKE", "TRUNCATE", "CALL"],
)
def test_all_forbidden_keywords_rejected(kw: str) -> None:
    sql = f"{kw} FROM `proj.raw.rows` LIMIT 10"
    result = guard(sql=sql, bq=_FakeBq(), settings=_settings())
    assert not result.accepted
    assert result.reason is not None
    assert result.reason.startswith(f"sql_forbidden_keyword: {kw}")
