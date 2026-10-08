"""Live StatCan tools: search ranking, coordinate validation, period
windows, and the tool contract (events, trace state, named errors)."""
from __future__ import annotations

from typing import Any

import pytest

from semantic_enrich.clients.statcan import StatCanError
from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events, agent_tools, statcan_tools

CUBES = [
    {
        "productId": 18100004,
        "cubeTitleEn": "Consumer Price Index, monthly, not seasonally adjusted",
        "cubeStartDate": "1914-01-01",
        "cubeEndDate": "2026-08-01",
        "archived": "2",
        "frequencyCode": 6,
    },
    {
        "productId": 25100015,
        "cubeTitleEn": "Electric power generation, monthly generation by type of electricity",
        "cubeStartDate": "2008-01-01",
        "cubeEndDate": "2026-07-01",
        "archived": "2",
        "frequencyCode": 6,
    },
    {
        "productId": 17100008,
        "cubeTitleEn": "Estimates of the components of demographic growth, annual",
        "cubeStartDate": "1971-01-01",
        "cubeEndDate": "2025-01-01",
        "archived": "2",
        "frequencyCode": 12,
    },
    {
        "productId": 17100001,
        "cubeTitleEn": "Estimates of the components of demographic growth, annual (archived)",
        "cubeStartDate": "1971-01-01",
        "cubeEndDate": "2010-01-01",
        "archived": "1",
        "frequencyCode": 12,
    },
]

META = {
    "productId": 18100004,
    "cubeTitleEn": "Consumer Price Index, monthly, not seasonally adjusted",
    "cubeStartDate": "1914-01-01",
    "cubeEndDate": "2026-08-01",
    "frequencyCode": 6,
    "archiveStatusCode": "2",
    "dimension": [
        {
            "dimensionPositionId": 1,
            "dimensionNameEn": "Geography",
            "member": [
                {"memberId": 2, "memberNameEn": "Canada", "parentMemberId": None},
                {"memberId": 11, "memberNameEn": "Ontario", "parentMemberId": 2},
            ],
        },
        {
            "dimensionPositionId": 2,
            "dimensionNameEn": "Products and product groups",
            "member": [
                {"memberId": 2, "memberNameEn": "All-items", "parentMemberId": None, "memberUomCode": 17},
                {"memberId": 3, "memberNameEn": "Food", "parentMemberId": 2, "memberUomCode": 17},
                {
                    "memberId": 4,
                    "memberNameEn": "Food purchased from stores",
                    "parentMemberId": 3,
                    "memberUomCode": 17,
                },
            ],
        },
    ],
}

CODES = {
    "uom": [{"memberUomCode": 17, "memberUomEn": "2002=100"}],
    "scalar": [{"scalarFactorCode": 0, "scalarFactorDescEn": "units"}],
    "symbol": [{"symbolCode": "1", "symbolDescEn": "preliminary"}],
}


def _points(*pairs: tuple[str, float]) -> list[dict[str, Any]]:
    return [{"refPer": p, "value": v, "scalarFactorCode": 0, "symbolCode": 0} for p, v in pairs]


class FakeStatCan:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.requests: list[tuple[int, list[str], int]] = []

    def list_cubes(self) -> list[dict[str, Any]]:
        if self.fail:
            raise StatCanError("WDS 503")
        return CUBES

    def cube_metadata(self, product_id: int) -> dict[str, Any]:
        if self.fail:
            raise StatCanError("WDS 503")
        return META

    def series_latest_n(
        self, product_id: int, coordinates: list[str], latest_n: int
    ) -> list[dict[str, Any]]:
        self.requests.append((product_id, coordinates, latest_n))
        return [
            {
                "vectorId": 41690973,
                "memberUomCode": 17,
                "vectorDataPoint": _points(
                    ("2014-12-01", 124.5),
                    ("2015-01-01", 124.3),
                    ("2025-01-01", 161.3),
                    ("2026-08-01", 169.8),
                ),
            }
            for _ in coordinates
        ]

    def code_sets(self) -> dict[str, Any]:
        return CODES


def _ctx(client: FakeStatCan) -> tuple[agent_tools.ToolContext, list[Any]]:
    events: list[Any] = []
    state = agent_tools.LoopState(conversation_id="c", turn_id="t", question="q")
    ctx = agent_tools.ToolContext(
        bq=None,  # type: ignore[arg-type]
        openai_client=None,  # type: ignore[arg-type]
        settings=Settings(),
        state=state,
        emit=events.append,
        statcan=client,  # type: ignore[arg-type]
    )
    return ctx, events


# ── pure helpers ──


def test_search_ranks_cpi_for_inflation_and_not_electric_power() -> None:
    top = statcan_tools.search_tables(CUBES, "purchasing power since 2015", k=2)
    assert top[0]["table_id"] == "18-10-0004-01"
    grocery = statcan_tools.search_tables(CUBES, "grocery inflation", k=1)
    assert grocery[0]["product_id"] == 18100004


def test_search_prefers_current_tables_over_archived() -> None:
    top = statcan_tools.search_tables(CUBES, "immigration population growth", k=2)
    assert top[0]["product_id"] == 17100008
    assert top[0]["current"] is True


def test_table_ids_round_trip() -> None:
    assert statcan_tools.table_id(18100004) == "18-10-0004-01"
    for form in (18100004, "18100004", "18-10-0004-01", "1810000401"):
        assert statcan_tools.parse_product_id(form) == 18100004
    with pytest.raises(ValueError):
        statcan_tools.parse_product_id("12345")


def test_coordinates_pad_to_ten_and_reject_unknown_members() -> None:
    coords, labels = statcan_tools.build_coordinates(META, [[2, 4]])
    assert coords == ["2.4.0.0.0.0.0.0.0.0"]
    assert labels == ["Canada; Food purchased from stores"]
    with pytest.raises(ValueError, match="exactly 2 member ids"):
        statcan_tools.build_coordinates(META, [[2]])
    with pytest.raises(ValueError, match="describe_statcan_table"):
        statcan_tools.build_coordinates(META, [[2, 999]])


def test_latest_n_covers_the_start_period() -> None:
    n = statcan_tools.latest_n_for(6, "2015-01", None)
    assert n >= 12 * 11
    assert statcan_tools.latest_n_for(12, None, None) == statcan_tools.DEFAULT_LATEST_N
    with pytest.raises(ValueError):
        statcan_tools.latest_n_for(6, "last spring", None)


def test_describe_truncates_large_dimensions_toward_the_filter() -> None:
    members = [
        {"memberId": i, "memberNameEn": f"Item {i}", "parentMemberId": 1 if i > 1 else None}
        for i in range(1, 80)
    ]
    members.append({"memberId": 99, "memberNameEn": "United States", "parentMemberId": 1})
    meta = {**META, "dimension": [{"dimensionPositionId": 1, "dimensionNameEn": "Country", "member": members}]}
    out = statcan_tools.describe_table(meta, CODES, member_filter="united states", max_members=10)
    dim = out["dimensions"][0]
    assert dim["truncated"] is True
    assert dim["member_count"] == 80
    assert any(m["id"] == 99 for m in dim["members"])
    assert any(m["id"] == 1 for m in dim["members"])


# ── tool contract ──


def test_get_data_filters_window_applies_units_and_records_state() -> None:
    client = FakeStatCan()
    ctx, events = _ctx(client)
    out = agent_tools.run_get_statcan_data(
        ctx=ctx,
        args={"product_id": "18-10-0004-01", "series": [[2, 2]], "start_period": "2015", "end_period": "2025"},
    )
    assert out["status"] == "ok"
    series = out["series"][0]
    assert series["unit"] == "2002=100"
    assert [p[0] for p in series["points"]] == ["2015-01", "2025-01"]
    assert out["cite_as"].startswith("[Consumer Price Index")
    assert "pid=1810000401" in out["cite_as"]
    assert ctx.state.statcan_tables == {"18-10-0004-01": META["cubeTitleEn"]}
    data_events = [e for e in events if isinstance(e, agent_events.SourceData)]
    assert len(data_events) == 1
    assert data_events[0].row_count == 2
    assert data_events[0].rows[0]["series"] == "Canada; All-items"


def test_bad_member_ids_are_a_tool_error_the_model_can_fix() -> None:
    ctx, _ = _ctx(FakeStatCan())
    with pytest.raises(agent_tools.InvalidToolArgsError, match="member id 7"):
        agent_tools.run_get_statcan_data(ctx=ctx, args={"product_id": "18100004", "series": [[2, 7]]})


def test_unreachable_statcan_is_a_named_error_not_a_crash() -> None:
    ctx, _ = _ctx(FakeStatCan(fail=True))
    out = agent_tools.run_search_statcan_tables(ctx=ctx, args={"query": "cpi"})
    assert out["status"] == "source_error"
    assert out["reason"] == "statcan_unavailable"


def test_search_emits_a_source_search_event() -> None:
    ctx, events = _ctx(FakeStatCan())
    out = agent_tools.run_search_statcan_tables(ctx=ctx, args={"query": "consumer price index"})
    assert out["candidates"][0]["table_id"] == "18-10-0004-01"
    assert isinstance(events[0], agent_events.SourceSearch)


def test_tools_are_registered_and_round_trip_as_sse() -> None:
    names = {t["function"]["name"] for t in agent_tools.tool_schemas()}
    assert {"search_statcan_tables", "describe_statcan_table", "get_statcan_data"} <= names
    event = agent_events.SourceData(
        source="statcan", table_id="x", title="t", url="u", request={}, row_count=0, rows=[]
    )
    assert agent_events.from_sse_frame(event.to_sse_frame()) == event
