"""SQL builders + decoders for the package-grouper queries."""
from __future__ import annotations

from semantic_enrich.core.package_grouper import (
    build_candidate_params,
    build_candidate_sql,
    build_doc_columns_sql,
    build_sample_rows_sql,
    column_union,
    decode_candidate_row,
    truncate_columns,
)


def test_build_candidate_sql_shape_with_limit() -> None:
    sql = build_candidate_sql(
        project_id="proj",
        dataset_raw="raw",
        documents_table="documents",
        with_limit=True,
    )
    assert "`proj.raw.documents`" in sql
    assert "load_status = 'loaded'" in sql
    assert "package_id IS NOT NULL" in sql
    assert "GROUP BY package_id" in sql
    assert "ORDER BY package_id" in sql
    assert "@already_extracted" in sql
    assert "LIMIT @limit_packages" in sql
    # `@p IS NULL` is the "no filter" predicate. The Python BQ SDK
    # serialises empty-list bindings as NULL ARRAY on the wire, so
    # IS-NULL is what we have to test against (an ARRAY_LENGTH=0
    # predicate evaluates to NULL → FALSE in WHERE).
    assert "@limit_orgs IS NULL" in sql
    assert "@limit_package_ids IS NULL" in sql
    assert "@already_extracted IS NULL" in sql


def test_build_candidate_sql_omits_limit_when_unset() -> None:
    sql = build_candidate_sql(
        project_id="proj",
        dataset_raw="raw",
        documents_table="documents",
        with_limit=False,
    )
    assert "LIMIT" not in sql
    assert "@limit_packages" not in sql


def test_build_candidate_params_includes_limit_when_set() -> None:
    params = build_candidate_params(
        limit_orgs=["org-a", "org-b"],
        limit_package_ids=None,
        already_extracted=["pkg-x"],
        limit_packages=10,
    )
    names = {p.name for p in params}
    assert names == {
        "limit_orgs",
        "limit_package_ids",
        "already_extracted",
        "limit_packages",
    }
    by_name = {p.name: p for p in params}
    assert by_name["limit_packages"].value == 10
    assert by_name["already_extracted"].values == ["pkg-x"]


def test_build_candidate_params_omits_limit_when_unset() -> None:
    params = build_candidate_params(
        limit_orgs=None,
        limit_package_ids=None,
        already_extracted=[],
        limit_packages=None,
    )
    names = {p.name for p in params}
    # `limit_packages` not in params; SQL omits the LIMIT clause too.
    assert names == {"limit_orgs", "limit_package_ids", "already_extracted"}
    by_name = {p.name: p for p in params}
    assert by_name["limit_orgs"].values == []
    assert by_name["limit_package_ids"].values == []


def test_decode_candidate_row_pulls_resources() -> None:
    row = {
        "package_id": "pkg-001",
        "resources": [
            {
                "document_id": "doc-1",
                "title": "Foo",
                "subjects": ["s1", "s2"],
                "organization_code": "org",
                "file_format": "csv",
                "resource_last_modified": None,
                "row_count": 100,
            }
        ],
    }
    pid, resources = decode_candidate_row(row)
    assert pid == "pkg-001"
    assert len(resources) == 1
    assert resources[0].document_id == "doc-1"
    assert resources[0].subjects == ("s1", "s2")


def test_build_doc_columns_sql_uses_json_keys() -> None:
    sql = build_doc_columns_sql(
        project_id="proj", dataset_raw="raw", rows_table="rows"
    )
    # Bare JSON_KEYS(row): raw.rows.row is a native JSON object since
    # the loader fix + reload. The legacy PARSE_JSON(STRING(row))
    # unwrap throws on object rows and must not reappear here.
    assert "JSON_KEYS(row)" in sql
    assert "PARSE_JSON" not in sql
    assert "QUALIFY" in sql
    assert "@document_ids" in sql
    # One row per document — the picker needs per-doc headers, not a
    # flattened union.
    assert "document_id," in sql


def test_column_union_is_sorted_distinct() -> None:
    assert column_union(
        {"doc-a": ["year", "amt"], "doc-b": ["amt", "region"]}
    ) == ["amt", "region", "year"]


def test_column_union_empty() -> None:
    assert column_union({}) == []


def test_build_sample_rows_sql_uses_row_index() -> None:
    sql = build_sample_rows_sql(
        project_id="proj", dataset_raw="raw", rows_table="rows"
    )
    assert "row_index" in sql
    assert "@document_id" in sql
    assert "@indices" in sql


def test_truncate_columns_under_cap() -> None:
    kept, total = truncate_columns(names=["a", "b", "c"], cap=10)
    assert kept == ("a", "b", "c")
    assert total is None


def test_truncate_columns_over_cap() -> None:
    names = [f"c{i}" for i in range(50)]
    kept, total = truncate_columns(names=names, cap=40)
    assert len(kept) == 40
    assert total == 50
