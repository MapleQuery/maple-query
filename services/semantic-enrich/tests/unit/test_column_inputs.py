"""Unit tests for `core.column_inputs`.

Allowlist regex semantics + SQL builder shape checks. Integration
behaviour against a FakeBqClient lives in
`tests/integration/test_column_inputs.py`.
"""
from __future__ import annotations

import re

import pytest

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.column_inputs import (
    _build_candidate_sql,
    _fetch_sample_values,
)


def _allowlist() -> re.Pattern[str]:
    return re.compile(Settings().column_name_allowlist_re)


@pytest.mark.parametrize(
    "name",
    [
        "year",
        "amt_cad",
        "Province",
        "fiscal-year",
        "naics.code",
        "path/to/key",
        "Province Code",
        "C0",
    ],
)
def test_allowlist_admits_normal_columns(name: str) -> None:
    assert _allowlist().match(name) is not None


@pytest.mark.parametrize(
    "name",
    [
        "",
        '"quoted"',
        "back`tick",
        "back\\slash",
        "$path.attack",
        "-leading-dash",
        ".leading-dot",
        "/leading-slash",
        " leading-space",
        "x" * 201,
    ],
)
def test_allowlist_rejects_attack_patterns(name: str) -> None:
    assert _allowlist().match(name) is None


def test_candidate_sql_includes_limit_clause_when_with_limit() -> None:
    sql = _build_candidate_sql(
        project_id="p",
        dataset_raw="raw",
        documents_table="documents",
        with_limit=True,
    )
    assert "LIMIT @limit_packages" in sql
    assert "load_status = 'loaded'" in sql
    assert "AND package_id IS NOT NULL" in sql


def test_candidate_sql_omits_limit_clause_when_no_limit() -> None:
    sql = _build_candidate_sql(
        project_id="p",
        dataset_raw="raw",
        documents_table="documents",
        with_limit=False,
    )
    assert "LIMIT" not in sql


def test_fetch_sample_values_returns_empty_dict_for_empty_input() -> None:
    """Defensive — caller should already filter, but the function
    must not issue a BQ query for zero columns."""
    bq_calls: list[dict[str, object]] = []

    class FakeBq:
        def query_rows(self, sql: str, *, params=()):  # type: ignore[no-untyped-def]
            bq_calls.append({"sql": sql})
            return iter([])

        # The Protocol surface — the test only exercises query_rows.
        def execute(self, sql: str) -> None: ...
        def execute_with_params(self, sql: str, *, params=()): ...  # type: ignore[no-untyped-def]
        def append_jsonl_file(self, **kwargs): ...  # type: ignore[no-untyped-def]
        def create_staging_table(self, **kwargs): ...  # type: ignore[no-untyped-def]
        def delete_table(self, *a, **kw): ...  # type: ignore[no-untyped-def]

    out = _fetch_sample_values(
        bq=FakeBq(),  # type: ignore[arg-type]
        project_id="p",
        dataset_raw="raw",
        rows_table="rows",
        document_id="doc",
        column_names=[],
        per_column_cap=10,
    )
    assert out == {}
    assert bq_calls == []
