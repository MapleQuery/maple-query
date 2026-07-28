"""`required_columns` filter on list_documents.

Replaces the "compute the SET INTERSECTION in your head" prompt prose:
returned docs are guaranteed safe to inline together for the listed
columns.
"""
from __future__ import annotations

import math
from typing import Any

import pytest

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events, agent_tools
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient


def _settings() -> Settings:
    return Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
    )


def _bq_with_docs() -> FakeBqClient:
    bq = FakeBqClient()
    bq.register_query(
        "load_status = 'loaded'",
        [
            {
                "document_id": "doc-A",
                "package_id": "pkg-1",
                "title": "A",
                "row_count": 10,
                "resource_last_modified": None,
            },
            {
                "document_id": "doc-B",
                "package_id": "pkg-1",
                "title": "B",
                "row_count": 20,
                "resource_last_modified": None,
            },
        ],
    )
    bq.register_query(
        "JSON_KEYS(row)",
        [
            {
                "document_id": "doc-A",
                "columns": ["FISCAL_YEAR", "Amount", "Org"],
            },
            {"document_id": "doc-B", "columns": ["Amount", "Description"]},
        ],
    )
    return bq


def _ctx(
    bq: FakeBqClient,
) -> tuple[agent_tools.ToolContext, list[agent_events.AgentEvent]]:
    state = agent_tools.LoopState(
        conversation_id="c1", turn_id="t1", question="q"
    )
    state.known_package_ids.add("pkg-1")
    events: list[agent_events.AgentEvent] = []
    ctx = agent_tools.ToolContext(
        bq=bq,
        openai_client=FakeOpenAIClient(
            vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536
        ),
        settings=_settings(),
        state=state,
        emit=events.append,
    )
    return ctx, events


def test_filter_keeps_only_satisfying_docs() -> None:
    ctx, events = _ctx(_bq_with_docs())
    result = agent_tools.run_list_documents(
        ctx=ctx,
        args={
            "package_ids": ["pkg-1"],
            "required_columns": ["FISCAL_YEAR", "Amount"],
        },
    )
    assert [d["document_id"] for d in result["documents"]] == ["doc-A"]
    assert result["filtered_out"] == [
        {"document_id": "doc-B", "missing_columns": ["FISCAL_YEAR"]}
    ]
    assert "required_columns_unsatisfiable" not in result
    listed = [e for e in events if e.event_type == "documents_listed"]
    assert isinstance(listed[0], agent_events.DocumentsListed)
    assert listed[0].filtered_out == result["filtered_out"]


def test_all_docs_satisfying_omits_filtered_out() -> None:
    ctx, _events = _ctx(_bq_with_docs())
    result = agent_tools.run_list_documents(
        ctx=ctx,
        args={"package_ids": ["pkg-1"], "required_columns": ["Amount"]},
    )
    assert len(result["documents"]) == 2
    assert "filtered_out" not in result


def test_unsatisfiable_returns_full_list_with_flag() -> None:
    """An empty result would push the model toward surrender — return
    the unfiltered listing plus the flag instead."""
    ctx, events = _ctx(_bq_with_docs())
    result = agent_tools.run_list_documents(
        ctx=ctx,
        args={
            "package_ids": ["pkg-1"],
            "required_columns": ["NOT_A_COLUMN"],
        },
    )
    assert result["required_columns_unsatisfiable"] is True
    assert result["unmatched_columns"] == ["NOT_A_COLUMN"]
    assert [d["document_id"] for d in result["documents"]] == [
        "doc-A",
        "doc-B",
    ]
    assert "filtered_out" not in result
    listed = [e for e in events if e.event_type == "documents_listed"]
    assert isinstance(listed[0], agent_events.DocumentsListed)
    assert listed[0].required_columns_unsatisfiable is True
    assert listed[0].filtered_out is None


# ── The `2025-26 Estimates` failure ──
#
# Real headers from two of that package's five documents. A turn scoped
# to the package asked for a sum of expenditures by department, the model
# passed those two words as `required_columns`, the exact-equality filter
# matched nothing, and the answer told the user the dataset does not hold
# the data — while `organization-summary` was in the listing the whole
# time.
_ORG_SUMMARY = [
    "Organization",
    "Vote",
    "Description",
    "2023-24 Expenditures",
    "2025-26 Main Estimates",
]
_STATUTORY = [
    "Department, Agency or Crown corporation",
    "2023–24 Expenditures",
    "2025–26 Main Estimates",
]


def _bq_estimates_package() -> FakeBqClient:
    bq = FakeBqClient()
    bq.register_query(
        "load_status = 'loaded'",
        [
            {
                "document_id": "doc-org-summary",
                "package_id": "pkg-1",
                "title": "organization-summary",
                "row_count": 586,
                "resource_last_modified": None,
            },
            {
                "document_id": "doc-statutory",
                "package_id": "pkg-1",
                "title": "statutory-forecasts",
                "row_count": 482,
                "resource_last_modified": None,
            },
        ],
    )
    bq.register_query(
        "JSON_KEYS(row)",
        [
            {"document_id": "doc-org-summary", "columns": _ORG_SUMMARY},
            {"document_id": "doc-statutory", "columns": _STATUTORY},
        ],
    )
    return bq


def test_question_words_no_longer_read_as_missing_data() -> None:
    ctx, _events = _ctx(_bq_estimates_package())
    result = agent_tools.run_list_documents(
        ctx=ctx,
        args={
            "package_ids": ["pkg-1"],
            "required_columns": ["Expenditures", "Department"],
        },
    )
    # Both concepts are present, so nothing is unsatisfiable.
    assert "required_columns_unsatisfiable" not in result
    assert result["required_columns_inexact"] is True
    # And the listing is NOT narrowed: only `statutory-forecasts` carries
    # both, and it is the document whose first column holds department
    # names on section rows and item names on data rows. Narrowing to it
    # on a loose match would hand back a confidently wrong GROUP BY.
    assert [d["document_id"] for d in result["documents"]] == [
        "doc-org-summary",
        "doc-statutory",
    ]
    assert result["column_matches"]["doc-org-summary"] == {
        "Expenditures": ["2023-24 Expenditures"]
    }
    assert result["column_matches"]["doc-statutory"] == {
        "Expenditures": ["2023–24 Expenditures"],
        "Department": ["Department, Agency or Crown corporation"],
    }
    # The steer must be about column names, never about the package —
    # under a user-set scope the package is not the model's to change,
    # and "reconsider your package choice" is what ended the real turn.
    guidance = result["guidance"].lower()
    assert "column_matches" in guidance
    assert "reconsider your package" not in guidance
    assert "reformulate your dataset search" not in guidance


def test_inexact_filter_does_not_burn_the_reformulation_signal() -> None:
    """A vocabulary mistake is not a weak-retrieval signal. Treating it
    as one spends the turn's free reformulation and steers to clarify."""
    ctx, _events = _ctx(_bq_estimates_package())
    agent_tools.run_list_documents(
        ctx=ctx,
        args={
            "package_ids": ["pkg-1"],
            "required_columns": ["Expenditures", "Department"],
        },
    )
    assert ctx.state.weak_signal_seen is False


def test_unmatched_column_reports_where_the_others_live() -> None:
    ctx, _events = _ctx(_bq_estimates_package())
    result = agent_tools.run_list_documents(
        ctx=ctx,
        args={
            "package_ids": ["pkg-1"],
            "required_columns": ["Expenditures", "Airplane"],
        },
    )
    assert result["required_columns_unsatisfiable"] is True
    assert result["unmatched_columns"] == ["Airplane"]
    assert result["column_availability"]["Airplane"] == []
    assert result["column_availability"]["Expenditures"] == [
        {
            "document_id": "doc-org-summary",
            "matching_columns": ["2023-24 Expenditures"],
        },
        {
            "document_id": "doc-statutory",
            "matching_columns": ["2023–24 Expenditures"],
        },
    ]
    assert ctx.state.weak_signal_seen is False


def test_exact_match_still_hard_filters() -> None:
    """The original guarantee survives: when literal names do resolve,
    the listing narrows exactly as before."""
    ctx, _events = _ctx(_bq_estimates_package())
    result = agent_tools.run_list_documents(
        ctx=ctx,
        args={
            "package_ids": ["pkg-1"],
            "required_columns": ["Organization", "2023-24 Expenditures"],
        },
    )
    assert [d["document_id"] for d in result["documents"]] == [
        "doc-org-summary"
    ]
    assert "required_columns_inexact" not in result
    assert result["filtered_out"] == [
        {
            "document_id": "doc-statutory",
            "missing_columns": ["Organization", "2023-24 Expenditures"],
        }
    ]


def test_empty_required_columns_list_is_a_noop() -> None:
    ctx, _events = _ctx(_bq_with_docs())
    result = agent_tools.run_list_documents(
        ctx=ctx,
        args={"package_ids": ["pkg-1"], "required_columns": []},
    )
    assert len(result["documents"]) == 2
    assert "filtered_out" not in result
    assert "required_columns_unsatisfiable" not in result


def test_invalid_required_columns_type_rejected() -> None:
    ctx, _events = _ctx(_bq_with_docs())
    with pytest.raises(agent_tools.InvalidToolArgsError):
        agent_tools.run_list_documents(
            ctx=ctx,
            args={"package_ids": ["pkg-1"], "required_columns": [1, 2]},
        )


def test_package_ids_bounds_still_enforced() -> None:
    ctx, _events = _ctx(_bq_with_docs())
    with pytest.raises(agent_tools.InvalidToolArgsError):
        agent_tools.run_list_documents(ctx=ctx, args={"package_ids": []})
    with pytest.raises(agent_tools.InvalidToolArgsError):
        agent_tools.run_list_documents(
            ctx=ctx, args={"package_ids": [f"pkg-{i}" for i in range(11)]}
        )


def test_filtered_out_docs_still_tracked_for_pairing_check() -> None:
    """A doc the filter removed stays in state so run_sql can still
    pairing-check it if the model inlines it anyway."""
    ctx, _events = _ctx(_bq_with_docs())
    agent_tools.run_list_documents(
        ctx=ctx,
        args={
            "package_ids": ["pkg-1"],
            "required_columns": ["FISCAL_YEAR"],
        },
    )
    assert "doc-B" in ctx.state.known_document_ids
    assert ctx.state.doc_columns["doc-B"] == ["Amount", "Description"]


def test_schema_declares_required_columns_param() -> None:
    schemas = {
        s["function"]["name"]: s["function"] for s in agent_tools.tool_schemas()
    }
    props: dict[str, Any] = schemas["list_documents"]["parameters"][
        "properties"
    ]
    assert "required_columns" in props
    # Optional — not in required.
    assert "required_columns" not in (
        schemas["list_documents"]["parameters"]["required"]
    )
