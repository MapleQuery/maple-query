"""§16.2 integration tests for the retrieval SQL shape."""
from __future__ import annotations

import math

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.retrieval import (
    embed_question,
    retrieve_columns,
    retrieve_documents,
    retrieve_packages,
)
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient


def _settings() -> Settings:
    return Settings(gcp_project_id="proj")


def test_embed_question_asserts_dim() -> None:
    client = FakeOpenAIClient()
    vec = embed_question(
        openai_client=client, question="q?", settings=_settings()
    )
    assert len(vec) == 1536


def test_package_search_sql_shape() -> None:
    bq = FakeBqClient()
    bq.register_query("VECTOR_SEARCH", [{"package_id": "pkg-1"}])
    vec = [1.0 / math.sqrt(1536)] * 1536
    packages, _ = retrieve_packages(bq=bq, question_vec=vec, settings=_settings())
    assert len(packages) == 1
    assert packages[0].package_id == "pkg-1"

    call = bq.calls[-1]
    assert "VECTOR_SEARCH(" in call["sql"]
    assert "COSINE" in call["sql"]
    assert "top_k => @k_packages" in call["sql"]
    # bound params
    names = {p.name for p in call["params"]}
    assert names == {"question_vec", "k_packages"}


def test_column_search_scoped() -> None:
    bq = FakeBqClient()
    bq.register_query("scoped_packages", [{"package_id": "pkg-1", "column_name": "TOT_EXP"}])
    vec = [1.0 / math.sqrt(1536)] * 1536
    cols, _ = retrieve_columns(
        bq=bq,
        question_vec=vec,
        scoped_packages=["pkg-1"],
        settings=_settings(),
    )
    assert len(cols) == 1

    call = bq.calls[-1]
    assert "WHERE package_id IN UNNEST(@scoped_packages)" in call["sql"]
    assert "top_k => @k_columns" in call["sql"]
    names = {p.name for p in call["params"]}
    assert names == {"question_vec", "scoped_packages", "k_columns"}


def test_retrieve_documents_scoped_and_filtered() -> None:
    """Documents retrieval feeds the literal-IN filter in the SQL-gen
    prompt: only `load_status='loaded'` docs get inlined, scoped to
    the candidate packages, capped per-package. A second query pulls
    the per-doc column set from `raw.rows` (JSON-keys of the first row
    per doc, PARSE_JSON-unwrapped) so the model can pair the right
    column with the right doc."""
    bq = FakeBqClient()
    bq.register_query(
        "load_status = 'loaded'",
        [
            {
                "document_id": "doc-1",
                "package_id": "pkg-1",
                "title": "Housing 2023",
                "row_count": 1234,
                "resource_last_modified": None,
            },
            {
                "document_id": "doc-2",
                "package_id": "pkg-1",
                "title": None,
                "row_count": None,
                "resource_last_modified": None,
            },
        ],
    )
    bq.register_query(
        "JSON_KEYS(PARSE_JSON(STRING(row)))",
        [
            {"document_id": "doc-1", "columns": ["Amount", "Organization"]},
            {"document_id": "doc-2", "columns": []},
        ],
    )
    docs, _ = retrieve_documents(
        bq=bq,
        package_ids=["pkg-1", "pkg-2"],
        settings=_settings(),
    )
    assert [d.document_id for d in docs] == ["doc-1", "doc-2"]
    assert docs[0].title == "Housing 2023"
    assert docs[0].columns == ("Amount", "Organization")
    assert docs[1].title is None
    assert docs[1].columns == ()

    docs_call = bq.calls[-2]
    assert "load_status = 'loaded'" in docs_call["sql"]
    assert "package_id IN UNNEST(@package_ids)" in docs_call["sql"]
    assert {p.name for p in docs_call["params"]} == {
        "package_ids", "max_per_package"
    }

    keys_call = bq.calls[-1]
    # Per-doc keys are cluster-pruned by a literal IN-list on document_id
    # and PARSE_JSON-unwrapped because raw.rows.row is a JSON-string scalar.
    assert "IN UNNEST(@document_ids)" in keys_call["sql"]
    assert "JSON_KEYS(PARSE_JSON(STRING(row)))" in keys_call["sql"]
    assert {p.name for p in keys_call["params"]} == {"document_ids"}


def test_retrieve_documents_empty_docs_skips_keys_query() -> None:
    """If the docs query returns nothing, don't send an empty-array
    IN-list to the keys query — return an empty list."""
    bq = FakeBqClient()
    bq.register_query("load_status = 'loaded'", [])
    docs, _ = retrieve_documents(
        bq=bq,
        package_ids=["pkg-missing"],
        settings=_settings(),
    )
    assert docs == []
    # Only the docs query fired — no keys query.
    assert len(bq.calls) == 1
