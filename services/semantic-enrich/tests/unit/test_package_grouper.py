"""SQL builders + decoders for the package-grouper queries."""
from __future__ import annotations

from semantic_enrich.core.package_grouper import (
    build_candidate_params,
    build_candidate_sql,
    build_column_union_sql,
    build_sample_rows_sql,
    decode_candidate_row,
    truncate_columns,
)


def test_build_candidate_sql_shape() -> None:
    sql = build_candidate_sql(
        project_id="proj", dataset_raw="raw", documents_table="documents"
    )
    assert "`proj.raw.documents`" in sql
    assert "load_status = 'loaded'" in sql
    assert "package_id IS NOT NULL" in sql
    assert "GROUP BY package_id" in sql
    assert "ORDER BY package_id" in sql
    assert "@limit_packages" in sql
    assert "@already_extracted" in sql


def test_build_candidate_params_passes_limits() -> None:
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
    # limit_packages is the bound integer.
    assert by_name["limit_packages"].value == 10
    # already_extracted carries the list verbatim.
    assert by_name["already_extracted"].values == ["pkg-x"]


def test_build_candidate_params_null_filters() -> None:
    params = build_candidate_params(
        limit_orgs=None,
        limit_package_ids=None,
        already_extracted=[],
        limit_packages=None,
    )
    by_name = {p.name: p for p in params}
    # NULL limit_orgs / limit_package_ids: ArrayQueryParameter with
    # None values, which the @p IS NULL guard short-circuits.
    assert by_name["limit_orgs"].values is None
    assert by_name["limit_package_ids"].values is None
    assert by_name["limit_packages"].value is None


def test_decode_candidate_row_pulls_resources() -> None:
    row = {
        "package_id": "pkg-001",
        "resources": [
            {
                "document_id": "doc-1",
                "title": "Foo",
                "description": "Bar",
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


def test_build_column_union_sql_uses_json_keys() -> None:
    sql = build_column_union_sql(
        project_id="proj", dataset_raw="raw", rows_table="rows"
    )
    assert "JSON_KEYS(row)" in sql
    assert "QUALIFY" in sql
    assert "@document_ids" in sql


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
