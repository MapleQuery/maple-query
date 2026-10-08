"""Live open.canada.ca DataStore tools: column validation, amendment
de-duplication, derived time groupings, the row cap, and the tool
contract."""

from __future__ import annotations

from typing import Any

import pytest

from semantic_enrich.clients.opencanada import OpenCanadaError
from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events, agent_tools, opencanada_tools

RID = "1d15a62f-5656-49ad-8c88-f40ce689d831"
FIELDS = [
    {"id": c, "type": "text"}
    for c in ("ref_number", "amendment_number", "agreement_value", "agreement_start_date", "owner_org")
]
ROWS = [
    # One agreement amended once: only the amendment counts.
    {
        "ref_number": "A",
        "amendment_number": "0",
        "agreement_value": "100",
        "agreement_start_date": "2023-05-01",
        "owner_org": "gac",
    },
    {
        "ref_number": "A",
        "amendment_number": "1",
        "agreement_value": "150",
        "agreement_start_date": "2023-05-01",
        "owner_org": "gac",
    },
    {
        "ref_number": "B",
        "amendment_number": "0",
        "agreement_value": "1,000",
        "agreement_start_date": "2024-02-10",
        "owner_org": "gac",
    },
    {
        "ref_number": "C",
        "amendment_number": "0",
        "agreement_value": "40",
        "agreement_start_date": "2024-06-01",
        "owner_org": "dnd",
    },
]


class FakeCkan:
    def __init__(self, *, total: int | None = None, fail: bool = False) -> None:
        self.total = total
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def package_search(self, query: str, rows: int) -> list[dict[str, Any]]:
        return [
            {
                "id": "p1",
                "title": "Reports",
                "organization": {"title": "X"},
                "resources": [{"id": "r", "format": "PDF"}],
            },
            {
                "id": "p2",
                "title": "Grants",
                "organization": {"title": "TBS"},
                "resources": [
                    {"id": "n", "name": "Grants Nothing to Report", "datastore_active": True},
                    {"id": RID, "name": "Grants", "datastore_active": True},
                ],
            },
        ]

    def datastore_search(self, params: dict[str, Any]) -> dict[str, Any]:
        if self.fail:
            raise OpenCanadaError("Field 'nope' not found")
        self.calls.append(params)
        offset = int(params.get("offset", 0))
        return {"total": self.total or len(ROWS), "records": ROWS[offset : offset + int(params["limit"])]}

    def resource_fields(self, resource_id: str) -> tuple[list[dict[str, Any]], int]:
        return FIELDS, len(ROWS)

    def resource_show(self, resource_id: str) -> dict[str, Any]:
        return {"name": "Grants", "package_id": "p2"}


def test_dedupe_keeps_the_latest_amendment_and_groups_by_fiscal_year() -> None:
    out = opencanada_tools.run_query(
        FakeCkan(),
        resource_id=RID,
        group_by=["fiscal_year:agreement_start_date"],
        sum_columns=["agreement_value"],
        dedupe={"key": "ref_number", "order": "amendment_number"},
    )
    assert out["amendments_collapsed"] == 1
    assert out["groups"] == [
        {"fiscal_year:agreement_start_date": "2023-24", "count": 2, "sum_agreement_value": 1150.0},
        {"fiscal_year:agreement_start_date": "2024-25", "count": 1, "sum_agreement_value": 40.0},
    ]
    assert out["totals"]["sum_agreement_value"] == 1190.0


def test_where_filters_after_fetch() -> None:
    out = opencanada_tools.run_query(
        FakeCkan(),
        resource_id=RID,
        where=[{"column": "agreement_value", "op": ">=", "value": 150}],
        fields=["ref_number", "agreement_value"],
    )
    assert [r["ref_number"] for r in out["rows"]] == ["A", "B"]


def test_unknown_column_names_the_real_ones() -> None:
    with pytest.raises(opencanada_tools.QueryArgsError, match="recipient_country"):
        opencanada_tools.run_query(FakeCkan(), resource_id=RID, filters={"recipient_country": "UA"})


def test_aggregation_refuses_more_rows_than_it_can_read() -> None:
    out = opencanada_tools.run_query(
        FakeCkan(total=1_000_000), resource_id=RID, group_by=["owner_org"], sum_columns=["agreement_value"]
    )
    assert out["status"] == "too_broad"


def test_search_puts_queryable_packages_first_and_hides_nothing_to_report() -> None:
    pkgs = opencanada_tools.search_packages(FakeCkan(), "grants")
    assert pkgs[0]["package_id"] == "p2"
    assert [r["resource_id"] for r in pkgs[0]["resources"]] == [RID]
    assert pkgs[1]["queryable"] is False


def _ctx(client: FakeCkan) -> tuple[agent_tools.ToolContext, list[Any]]:
    events: list[Any] = []
    state = agent_tools.LoopState(conversation_id="c", turn_id="t", question="q")
    return (
        agent_tools.ToolContext(
            bq=None,  # type: ignore[arg-type]
            openai_client=None,  # type: ignore[arg-type]
            settings=Settings(),
            state=state,
            emit=events.append,
            opencanada=client,  # type: ignore[arg-type]
        ),
        events,
    )


def test_tool_emits_source_data_and_a_citation() -> None:
    ctx, events = _ctx(FakeCkan())
    out = agent_tools.run_query_open_canada(
        ctx=ctx,
        args={"resource_id": RID, "group_by": ["owner_org"], "sum_columns": ["agreement_value"]},
    )
    assert (
        out["cite_as"]
        == f"[Grants (open.canada.ca)](https://open.canada.ca/data/en/dataset/p2/resource/{RID})"
    )
    assert ctx.state.opencanada_tables == {RID: "Grants"}
    data = [e for e in events if isinstance(e, agent_events.SourceData)]
    assert data and data[0].source == "open.canada.ca"


def test_ckan_failure_is_a_named_error() -> None:
    ctx, _ = _ctx(FakeCkan(fail=True))
    out = agent_tools.run_query_open_canada(ctx=ctx, args={"resource_id": RID, "fields": ["owner_org"]})
    assert out["reason"] == "opencanada_error"
    assert "nope" in out["message"]


def test_derived_column_in_filters_becomes_a_where() -> None:
    out = opencanada_tools.run_query(
        FakeCkan(),
        resource_id=RID,
        filters={"fiscal_year:agreement_start_date": "2024-25"},
        group_by=["owner_org"],
        sum_columns=["agreement_value"],
        dedupe={"key": "ref_number", "order": "amendment_number"},
    )
    # B (2024-02-10) is fiscal 2023-24; only C (2024-06-01) is 2024-25.
    assert out["groups"] == [{"owner_org": "dnd", "count": 1, "sum_agreement_value": 40.0}]


def test_a_thin_latest_period_is_flagged_incomplete() -> None:
    groups = [
        {"y": "2021", "sum_v": 100.0},
        {"y": "2022", "sum_v": 400.0},
        {"y": "2023", "sum_v": 480.0},
        {"y": "2024", "sum_v": 230.0},
        {"y": "2025", "sum_v": 8.0},
    ]
    opencanada_tools._flag_thin_tail(groups, "sum_v")
    assert groups[-1].get("likely_incomplete") is True
    assert "likely_incomplete" not in groups[-2]


def test_bare_fiscal_year_names_the_derivation() -> None:
    with pytest.raises(opencanada_tools.QueryArgsError, match="fiscal_year:agreement_start_date"):
        opencanada_tools.run_query(FakeCkan(), resource_id=RID, group_by=["fiscal_year"])


def test_date_window_bounds_from_where_and_derived_filters() -> None:
    assert opencanada_tools._date_window(
        [{"column": "fiscal_year:agreement_start_date", "op": "=", "value": "2024-25"}]
    ) == ("agreement_start_date", "2024-04-01")
    assert opencanada_tools._date_window(
        [{"column": "agreement_start_date", "op": ">=", "value": "2026-01-01"}]
    ) == ("agreement_start_date", "2026-01-01")
    assert opencanada_tools._date_window([{"column": "owner_org", "op": "=", "value": "x"}]) is None


def test_text_money_columns_sort_numerically() -> None:
    client = FakeCkan()
    out = opencanada_tools.run_query(
        client,
        resource_id=RID,
        fields=["ref_number", "agreement_value"],
        sort="agreement_value desc",
        limit=2,
    )
    # "1,000" > "150" > "100" > "40" numerically; as strings "40" would win.
    assert [r["agreement_value"] for r in out["rows"]] == ["1,000", "150"]
    assert all("sort" not in c for c in client.calls)
