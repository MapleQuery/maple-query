"""Per-column sample values + garbage-header quality flag on
list_documents."""
from __future__ import annotations

import json
import math
from typing import Any

from semantic_enrich.clients.bq import BoundedQueryResult
from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events, agent_tools
from semantic_enrich.core.retrieval import fetch_document_samples
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
    samples = fetch_document_samples(
        bq=bq, doc_ids=["doc-1"], settings=_settings()
    )
    # Rows are consumed in row_index order; values truncate to the cap.
    assert samples["doc-1"]["Org"] == [
        "N" * 40,
        "National Capital Commission",
    ]
    # Non-string values are stringified; NULLs are skipped.
    assert samples["doc-1"]["Amt"] == ["5"]


def test_sampling_query_prunes_on_document_id_cluster() -> None:
    bq = FakeBqClient()
    fetch_document_samples(bq=bq, doc_ids=["doc-1"], settings=_settings())
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
    samples = fetch_document_samples(
        bq=bq, doc_ids=["doc-1"], settings=settings
    )
    assert len(samples["doc-1"]) == 30


def test_sampling_failure_degrades_to_empty() -> None:
    bq = FakeBqClient()
    bq.register_bounded_query(
        "TO_JSON_STRING(row)",
        BoundedQueryResult(
            rows=[], total_bytes_billed=0, slot_ms=0,
            elapsed_ms=5001, timed_out=True, error=None,
        ),
    )
    assert (
        fetch_document_samples(
            bq=bq, doc_ids=["doc-1"], settings=_settings()
        )
        == {}
    )
    assert fetch_document_samples(bq=bq, doc_ids=[], settings=_settings()) == {}


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
    samples = fetch_document_samples(
        bq=bq, doc_ids=["doc-1"], settings=_settings()
    )
    assert samples == {"doc-1": {"Org": ["x"]}}


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
