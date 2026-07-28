"""Translation reaches the guard, not just the comment above it.

The whole safety argument for this feature is about *ordering*. A
JSONPath to a key that does not exist does not fail in BigQuery — it
returns NULL. So SQL written with a recovered name that never got
translated would pass the guard's dry run, execute, and produce an
all-NULL aggregate: a wrong number with nothing anywhere saying so.

That makes "the guard sees the translated SQL" a property worth asserting
on the guard's actual recorded input rather than trusting a comment
about it, which is what these tests do.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from semantic_enrich.clients.bq import BoundedQueryResult
from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_tools
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient

DOC = "doc-1"
OTHER = "doc-2"
RECOVERED = {"__col_2": "Total Amount ($000)"}


@pytest.fixture()
def guard_inputs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the SQL every `guard` call actually receives."""
    seen: list[str] = []
    real = agent_tools.guard

    def _spy(*, sql: str, bq: Any, settings: Any) -> Any:
        seen.append(sql)
        return real(sql=sql, bq=bq, settings=settings)

    monkeypatch.setattr(agent_tools, "guard", _spy)
    return seen


def _settings(**overrides: Any) -> Settings:
    return Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
    ).model_copy(update=overrides)


def _ctx(
    *,
    recovered: dict[str, dict[str, str]] | None = None,
    header_rows: dict[str, int] | None = None,
    doc_columns: dict[str, list[str]] | None = None,
) -> agent_tools.ToolContext:
    bq = FakeBqClient()
    bq.bounded_default = BoundedQueryResult(
        rows=[{"total": 1.0}],
        total_bytes_billed=1024,
        slot_ms=1,
        elapsed_ms=2,
        timed_out=False,
        error=None,
    )
    state = agent_tools.LoopState(
        conversation_id="c1", turn_id="t1", question="q"
    )
    state.doc_columns.update(
        doc_columns or {DOC: ["Province", "__col_1", "__col_2"]}
    )
    state.known_document_ids.update(state.doc_columns)
    state.doc_recovered_names.update(recovered or {})
    state.doc_header_row.update(header_rows or {})
    return agent_tools.ToolContext(
        bq=bq,
        openai_client=FakeOpenAIClient(vector_factory=lambda _t: [0.0] * 1536),
        settings=_settings(),
        state=state,
        emit=lambda _e: None,
    )


def _run(ctx: agent_tools.ToolContext, sql: str) -> dict[str, Any]:
    return agent_tools.run_run_sql(ctx=ctx, args={"sql": sql, "rationale": "test"})


def _select(path: str, doc_ids: str = f"'{DOC}'") -> str:
    return (
        f"SELECT SUM(SAFE_CAST(JSON_VALUE(row, '{path}') AS FLOAT64)) AS total "
        f"FROM `proj.raw.rows` WHERE document_id IN ({doc_ids}) LIMIT 10"
    )


def test_the_guard_receives_the_translated_sql(
    guard_inputs: list[str],
) -> None:
    ctx = _ctx(recovered={DOC: RECOVERED})
    _run(ctx, _select('$."Total Amount ($000)"'))
    assert guard_inputs, "guard was never reached"
    assert "'$.__col_2'" in guard_inputs[-1]
    assert "Total Amount ($000)" not in guard_inputs[-1]


def test_the_guard_receives_the_preamble_predicate(
    guard_inputs: list[str],
) -> None:
    ctx = _ctx(recovered={DOC: RECOVERED}, header_rows={DOC: 2})
    _run(ctx, _select("$.__col_2"))
    assert f"NOT (document_id = '{DOC}' AND row_index <= 2)" in guard_inputs[-1]


def test_a_recovered_name_does_not_trip_the_pairing_check() -> None:
    """The model writes the name it was shown. Being told that column
    does not exist is the confusing error this exists to remove."""
    ctx = _ctx(recovered={DOC: RECOVERED})
    result = _run(ctx, _select('$."Total Amount ($000)"'))
    assert result.get("status") != "column_not_in_doc"


def test_an_invented_name_still_errors_and_points_somewhere_useful() -> None:
    ctx = _ctx(recovered={DOC: RECOVERED})
    result = _run(ctx, _select('$."Invented Column"'))
    assert result["status"] == "column_not_in_doc"
    assert "Invented Column" in result["message"]


def test_the_error_names_the_document_a_recovered_name_belongs_to() -> None:
    """Querying doc-2 with a name recovered for doc-1 should say where
    that name actually lives, not just that it is absent here."""
    ctx = _ctx(
        recovered={DOC: RECOVERED},
        doc_columns={DOC: ["__col_2"], OTHER: ["__col_9"]},
    )
    result = _run(ctx, _select('$."Total Amount ($000)"', f"'{OTHER}'"))
    assert result["status"] == "column_not_in_doc"
    assert DOC in json.dumps(result["violations"])


def test_a_conflict_refuses_before_the_guard_runs(
    guard_inputs: list[str],
) -> None:
    ctx = _ctx(
        recovered={DOC: {"__col_2": "Total"}, OTHER: {"__col_5": "Total"}},
        doc_columns={DOC: ["__col_2"], OTHER: ["__col_5"]},
    )
    result = _run(ctx, _select('$."Total"', f"'{DOC}', '{OTHER}'"))
    assert result["status"] == "recovered_name_conflict"
    assert guard_inputs == [], "a conflicting query must never be dry-run"
    assert result["conflicts"][0]["name"] == "Total"
    # The message has to be actionable: narrow the scope, or address the
    # positional key directly.
    assert "__col_2" in result["message"]
    assert "__col_5" in result["message"]


def test_the_translation_is_reported_back_to_the_model() -> None:
    ctx = _ctx(recovered={DOC: RECOVERED}, header_rows={DOC: 2})
    result = _run(ctx, _select('$."Total Amount ($000)"'))
    normalizations = result["normalizations"]
    assert normalizations["recovered_names_resolved"] == {
        "Total Amount ($000)": "__col_2"
    }
    assert normalizations["preamble_rows_excluded"] == {DOC: 2}


def test_with_recovery_off_the_sql_path_is_untouched(
    guard_inputs: list[str],
) -> None:
    """Nothing on state means both passes are no-ops by construction,
    which is why there is no second feature flag here."""
    sql = _select("$.__col_2")
    ctx = _ctx()
    result = _run(ctx, sql)
    assert guard_inputs[-1] == sql
    assert "recovered_names_resolved" not in result.get("normalizations", {})
    assert "preamble_rows_excluded" not in result.get("normalizations", {})
