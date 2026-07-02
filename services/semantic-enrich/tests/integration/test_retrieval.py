"""§16.2 integration tests for the retrieval SQL shape."""
from __future__ import annotations

import math

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.retrieval import (
    embed_question,
    retrieve_columns,
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
