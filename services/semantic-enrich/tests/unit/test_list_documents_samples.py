"""The consolidated list_documents read path: one bounded raw.rows
job for column key sets + sample values, keys-query fallback, and the
garbage-header quality flag."""
from __future__ import annotations

import json
import math
from typing import Any

from semantic_enrich.clients.bq import BoundedQueryResult
from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events, agent_tools
from semantic_enrich.core.retrieval import fetch_document_columns_and_samples
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient


def _settings(**overrides: Any) -> Settings:
    return Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
    ).model_copy(update=overrides)


def _sample_result(rows: list[dict[str, Any]]) -> BoundedQueryResult:
    return BoundedQueryResult(
        rows=rows,
        total_bytes_billed=1024,
        slot_ms=1,
        elapsed_ms=3,
        timed_out=False,
        error=None,
    )


def _ctx(
    *, bq: FakeBqClient, settings: Settings | None = None
) -> tuple[agent_tools.ToolContext, list[agent_events.AgentEvent]]:
    state = agent_tools.LoopState(
        conversation_id="c1", turn_id="t1", question="q"
    )
    events: list[agent_events.AgentEvent] = []
    ctx = agent_tools.ToolContext(
        bq=bq,
        openai_client=FakeOpenAIClient(
            vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536
        ),
        settings=settings or _settings(),
        state=state,
        emit=events.append,
    )
    return ctx, events


def test_samples_extracted_and_truncated() -> None:
    bq = FakeBqClient()
    long_value = "N" * 100
    bq.register_bounded_query(
        "TO_JSON_STRING(row)",
        _sample_result(
            [
                {
                    "document_id": "doc-1",
                    "row_index": 1,
                    "row_json": json.dumps(
                        {"Org": "National Capital Commission", "Amt": 5}
                    ),
                },
                {
                    "document_id": "doc-1",
                    "row_index": 0,
                    "row_json": json.dumps({"Org": long_value, "Amt": None}),
                },
            ]
        ),
    )
    columns, samples = fetch_document_columns_and_samples(
        bq=bq, doc_ids=["doc-1"], settings=_settings()
    )
    # Rows are consumed in row_index order; values truncate to the cap
    # with the shared ellipsis marker, so a clipped value is
    # distinguishable from a genuinely short one.
    assert samples["doc-1"]["Org"] == [
        "N" * 39 + "\u2026",
        "National Capital Commission",
    ]
    # Non-string values are stringified; NULLs are skipped.
    assert samples["doc-1"]["Amt"] == ["5"]
    # Columns derive from the lowest-index row's key set — NULL-valued
    # keys included, exactly as JSON_KEYS(row) reported them.
    assert columns == {"doc-1": ["Org", "Amt"]}


def test_sampling_query_prunes_on_document_id_cluster() -> None:
    bq = FakeBqClient()
    fetch_document_columns_and_samples(
        bq=bq, doc_ids=["doc-1"], settings=_settings()
    )
    assert len(bq.bounded_calls) == 1
    sql = bq.bounded_calls[0]
    assert "document_id IN UNNEST(@doc_ids)" in sql
    assert "row_index < @n" in sql
    assert "`proj.raw.rows`" in sql


def test_column_cap_respected() -> None:
    bq = FakeBqClient()
    wide_row = {f"col_{i}": f"v{i}" for i in range(50)}
    bq.register_bounded_query(
        "TO_JSON_STRING(row)",
        _sample_result(
            [
                {
                    "document_id": "doc-1",
                    "row_index": 0,
                    "row_json": json.dumps(wide_row),
                }
            ]
        ),
    )
    settings = _settings(agent_sample_values_max_columns=30)
    _columns, samples = fetch_document_columns_and_samples(
        bq=bq, doc_ids=["doc-1"], settings=settings
    )
    assert len(samples["doc-1"]) == 30


def test_sampling_timeout_falls_back_to_keys_query() -> None:
    # Samples are advisory (degrade to empty); columns are load-bearing
    # and must survive via the old JSON_KEYS(row) query.
    bq = FakeBqClient()
    bq.register_bounded_query(
        "TO_JSON_STRING(row)",
        BoundedQueryResult(
            rows=[], total_bytes_billed=0, slot_ms=0,
            elapsed_ms=5001, timed_out=True, error=None,
        ),
    )
    bq.register_query(
        "JSON_KEYS(row)",
        [{"document_id": "doc-1", "columns": ["Org", "Amt"]}],
    )
    columns, samples = fetch_document_columns_and_samples(
        bq=bq, doc_ids=["doc-1"], settings=_settings()
    )
    assert samples == {}
    assert columns == {"doc-1": ["Org", "Amt"]}
    # Degraded path: exactly two jobs (bounded + keys fallback).
    assert len(bq.bounded_calls) == 1
    keys_calls = [
        c for c in bq.calls if "JSON_KEYS(row)" in str(c.get("sql", ""))
    ]
    assert len(keys_calls) == 1
    assert fetch_document_columns_and_samples(
        bq=bq, doc_ids=[], settings=_settings()
    ) == ({}, {})


def test_malformed_row_json_skipped() -> None:
    bq = FakeBqClient()
    bq.register_bounded_query(
        "TO_JSON_STRING(row)",
        _sample_result(
            [
                {"document_id": "doc-1", "row_index": 0, "row_json": "{oops"},
                {"document_id": "doc-1", "row_index": 1, "row_json": "[1,2]"},
                {
                    "document_id": "doc-1",
                    "row_index": 2,
                    "row_json": json.dumps({"Org": "x"}),
                },
            ]
        ),
    )
    columns, samples = fetch_document_columns_and_samples(
        bq=bq, doc_ids=["doc-1"], settings=_settings()
    )
    assert samples == {"doc-1": {"Org": ["x"]}}
    # The first *parseable dict* row supplies the key set.
    assert columns == {"doc-1": ["Org"]}


def _register_docs(bq: FakeBqClient, docs: list[dict[str, Any]]) -> None:
    bq.register_query("load_status = 'loaded'", docs)


def test_generated_header_docs_flagged_and_demoted() -> None:
    bq = FakeBqClient()
    _register_docs(
        bq,
        [
            {
                "document_id": "doc-garbage",
                "package_id": "pkg-1",
                "title": "Garbage headers",
                "row_count": 10,
                "resource_last_modified": None,
            },
            {
                "document_id": "doc-clean",
                "package_id": "pkg-1",
                "title": "Clean",
                "row_count": 10,
                "resource_last_modified": None,
            },
        ],
    )
    bq.register_query(
        "JSON_KEYS(row)",
        [
            {
                "document_id": "doc-garbage",
                "columns": [
                    "Canada_Mortgage_and_Housing_Corporation",
                    "__col_1",
                    "__col_2",
                    "__col_3",
                ],
            },
            {"document_id": "doc-clean", "columns": ["Org", "Amount"]},
        ],
    )
    ctx, events = _ctx(bq=bq)
    ctx.state.known_package_ids.add("pkg-1")
    result = agent_tools.run_list_documents(
        ctx=ctx, args={"package_ids": ["pkg-1"]}
    )
    docs = result["documents"]
    # Demoted, not dropped: the garbage doc sorts after every clean doc.
    assert [d["document_id"] for d in docs] == ["doc-clean", "doc-garbage"]
    assert docs[1]["quality"] == "low_generated_headers"
    # Clean docs omit the key entirely.
    assert "quality" not in docs[0]
    listed = [e for e in events if e.event_type == "documents_listed"]
    assert len(listed) == 1
    assert isinstance(listed[0], agent_events.DocumentsListed)
    assert listed[0].documents[1]["quality"] == "low_generated_headers"


def test_exactly_at_ratio_threshold_is_clean() -> None:
    """The flag fires strictly above the ratio, not at it."""
    bq = FakeBqClient()
    _register_docs(
        bq,
        [
            {
                "document_id": "doc-half",
                "package_id": "pkg-1",
                "title": "Half generated",
                "row_count": 10,
                "resource_last_modified": None,
            }
        ],
    )
    bq.register_query(
        "JSON_KEYS(row)",
        [{"document_id": "doc-half", "columns": ["Org", "__col_1"]}],
    )
    ctx, _events = _ctx(bq=bq)
    ctx.state.known_package_ids.add("pkg-1")
    result = agent_tools.run_list_documents(
        ctx=ctx, args={"package_ids": ["pkg-1"]}
    )
    assert "quality" not in result["documents"][0]


def test_column_samples_ride_alongside_columns_list() -> None:
    bq = FakeBqClient()
    _register_docs(
        bq,
        [
            {
                "document_id": "doc-1",
                "package_id": "pkg-1",
                "title": "T",
                "row_count": 10,
                "resource_last_modified": None,
            }
        ],
    )
    bq.register_query(
        "JSON_KEYS(row)",
        [{"document_id": "doc-1", "columns": ["Org", "Amt"]}],
    )
    bq.register_bounded_query(
        "TO_JSON_STRING(row)",
        _sample_result(
            [
                {
                    "document_id": "doc-1",
                    "row_index": 0,
                    "row_json": json.dumps({"Org": "Total", "Amt": "1"}),
                }
            ]
        ),
    )
    ctx, _events = _ctx(bq=bq)
    ctx.state.known_package_ids.add("pkg-1")
    result = agent_tools.run_list_documents(
        ctx=ctx, args={"package_ids": ["pkg-1"]}
    )
    doc = result["documents"][0]
    # `columns` keeps its plain list shape (FE compatibility); samples
    # are a sibling field.
    assert doc["columns"] == ["Org", "Amt"]
    assert doc["column_samples"] == {"Org": ["Total"], "Amt": ["1"]}


def test_generated_header_ratio_empty_columns() -> None:
    assert agent_tools._generated_header_ratio([]) == 0.0
    assert agent_tools._generated_header_ratio(["__col_1", "__col_2"]) == 1.0
    assert agent_tools._generated_header_ratio(["__col_1x"]) == 0.0


def test_null_columns_do_not_consume_sample_cap() -> None:
    """Sparse wide row: NULL-valued keys must not eat the column cap
    and evict the value-bearing columns behind them."""
    bq = FakeBqClient()
    sparse = {"a": None, "b": None, "c": None, "d": "val-d", "e": "val-e"}
    bq.register_bounded_query(
        "TO_JSON_STRING(row)",
        _sample_result(
            [
                {
                    "document_id": "doc-1",
                    "row_index": 0,
                    "row_json": json.dumps(sparse),
                }
            ]
        ),
    )
    settings = _settings(agent_sample_values_max_columns=3)
    _columns, samples = fetch_document_columns_and_samples(
        bq=bq, doc_ids=["doc-1"], settings=settings
    )
    assert samples["doc-1"] == {"d": ["val-d"], "e": ["val-e"]}


# ── consolidated read path (tool level) ──


def test_one_rows_job_per_call_on_happy_path() -> None:
    bq = FakeBqClient()
    _register_docs(
        bq,
        [
            {
                "document_id": "doc-1",
                "package_id": "pkg-1",
                "title": "Doc",
                "row_count": 10,
                "resource_last_modified": None,
            }
        ],
    )
    bq.register_bounded_query(
        "TO_JSON_STRING(row)",
        _sample_result(
            [
                {
                    "document_id": "doc-1",
                    "row_index": 0,
                    "row_json": json.dumps({"Org": "x", "Amt": 5}),
                }
            ]
        ),
    )
    ctx, _ = _ctx(bq=bq)
    ctx.state.known_package_ids.add("pkg-1")
    result = agent_tools.run_list_documents(
        ctx=ctx, args={"package_ids": ["pkg-1"]}
    )
    assert result["documents"][0]["columns"] == ["Org", "Amt"]
    assert result["documents"][0]["column_samples"] == {
        "Org": ["x"],
        "Amt": ["5"],
    }
    # Exactly one raw.rows job: the bounded samples query. The old
    # JSON_KEYS(row) query must not run on the happy path.
    assert len(bq.bounded_calls) == 1
    keys_calls = [
        c for c in bq.calls if "JSON_KEYS(row)" in str(c.get("sql", ""))
    ]
    assert keys_calls == []
    assert ctx.state.doc_columns["doc-1"] == ["Org", "Amt"]


def test_columns_parity_between_merged_and_keys_paths() -> None:
    """The client-side key derivation yields exactly what the old
    JSON_KEYS(row) query returned, order included — the tripwire for a
    future loader ever writing nested row bodies."""
    body = {"A": 1, "B": None, "C": "x"}
    merged_bq = FakeBqClient()
    merged_bq.register_bounded_query(
        "TO_JSON_STRING(row)",
        _sample_result(
            [
                {
                    "document_id": "doc-1",
                    "row_index": 0,
                    "row_json": json.dumps(body),
                }
            ]
        ),
    )
    merged_columns, _ = fetch_document_columns_and_samples(
        bq=merged_bq, doc_ids=["doc-1"], settings=_settings()
    )

    fallback_bq = FakeBqClient()
    fallback_bq.register_bounded_query(
        "TO_JSON_STRING(row)",
        BoundedQueryResult(
            rows=[], total_bytes_billed=0, slot_ms=0,
            elapsed_ms=5001, timed_out=True, error=None,
        ),
    )
    fallback_bq.register_query(
        "JSON_KEYS(row)",
        [{"document_id": "doc-1", "columns": list(body.keys())}],
    )
    fallback_columns, _ = fetch_document_columns_and_samples(
        bq=fallback_bq, doc_ids=["doc-1"], settings=_settings()
    )
    assert merged_columns == fallback_columns == {"doc-1": ["A", "B", "C"]}


def test_sampling_timeout_does_not_disable_pairing_check() -> None:
    """The 6.2 contract regression guard: a bounded-path timeout must
    not leave doc_columns empty, or run_sql's doc/column pairing check
    silently skips."""
    bq = FakeBqClient()
    _register_docs(
        bq,
        [
            {
                "document_id": "doc-1",
                "package_id": "pkg-1",
                "title": "Doc",
                "row_count": 10,
                "resource_last_modified": None,
            }
        ],
    )
    bq.register_bounded_query(
        "TO_JSON_STRING(row)",
        BoundedQueryResult(
            rows=[], total_bytes_billed=0, slot_ms=0,
            elapsed_ms=5001, timed_out=True, error=None,
        ),
    )
    bq.register_query(
        "JSON_KEYS(row)",
        [{"document_id": "doc-1", "columns": ["Org", "Amount"]}],
    )
    ctx, _ = _ctx(bq=bq)
    ctx.state.known_package_ids.add("pkg-1")
    result = agent_tools.run_list_documents(
        ctx=ctx, args={"package_ids": ["pkg-1"]}
    )
    assert result["documents"][0]["columns"] == ["Org", "Amount"]
    assert "column_samples" not in result["documents"][0]
    assert ctx.state.doc_columns["doc-1"] == ["Org", "Amount"]
    # A bad column reference is still caught, not silently passed.
    sql_result = agent_tools.run_run_sql(
        ctx=ctx,
        args={
            "sql": (
                "SELECT JSON_VALUE(r.row, '$.Not_A_Column') AS x "
                "FROM `proj.raw.rows` AS r "
                "WHERE r.document_id IN ('doc-1') LIMIT 10"
            ),
            "rationale": "pairing regression guard",
        },
    )
    assert sql_result["status"] == "column_not_in_doc"
