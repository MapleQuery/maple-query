"""An inlined `document_id` that list_documents never returned.

The observed failure: the model copied a 64-character digest out of
list_documents, dropped a 14-character run from the middle of it, and
ran the query anyway. The guard passed it (a literal `document_id IN`
predicate was present), the pairing check skipped it (an invented id has
no columns to pair against), it matched zero rows, and the answer told
the user the data might not be there.
"""
from __future__ import annotations

from semantic_enrich.core import agent_tools

# The real digest, and the mangled one, from that turn.
GOOD = "087aa65c8c5f917cf2a7a178933d71b57d893e20c718e1ffef2b20e05fbc398f"
MANGLED = "087aa65c8c5f917cf2a7a178933d71b57d893e20c05fbc398f"


def _state(*known: str) -> agent_tools.LoopState:
    state = agent_tools.LoopState(
        conversation_id="c", turn_id="t", question="q"
    )
    for doc_id in known:
        state.known_document_ids.add(doc_id)
        state.doc_columns[doc_id] = ["Organization", "2025-26 Main Estimates"]
    return state


def _sql(doc_id: str) -> str:
    return (
        "SELECT JSON_VALUE(r.row, '$.Organization') AS department, "
        "SUM(SAFE_CAST(JSON_VALUE(r.row, '$.\"2025-26 Main Estimates\"') "
        "AS FLOAT64)) AS total FROM `proj.raw.rows` AS r "
        f"WHERE r.document_id IN ('{doc_id}') GROUP BY department LIMIT 100"
    )


def test_the_mangled_id_is_caught_and_the_real_one_offered() -> None:
    unknown, msg = agent_tools.check_document_ids_known(
        sql=_sql(MANGLED), state=_state(GOOD)
    )
    assert len(unknown) == 1
    assert unknown[0]["document_id"] == MANGLED
    assert unknown[0]["did_you_mean"] == [GOOD]
    assert msg is not None
    assert "unknown_document_id" in msg
    assert GOOD in msg
    # The model must not read an empty result as absent data.
    assert "NOT evidence that the data is missing" in msg


def test_a_listed_id_passes() -> None:
    unknown, msg = agent_tools.check_document_ids_known(
        sql=_sql(GOOD), state=_state(GOOD)
    )
    assert unknown == []
    assert msg is None


def test_an_unrelated_id_gets_no_suggestion() -> None:
    other = "f" * 64
    unknown, msg = agent_tools.check_document_ids_known(
        sql=_sql(other), state=_state(GOOD)
    )
    assert unknown[0]["did_you_mean"] == []
    assert msg is not None
    assert "You most likely meant" not in msg


def test_skipped_when_list_documents_did_not_run() -> None:
    """`POST /sql/run` drives this tool with a scratch state, and an
    inline retry may reuse an id from an earlier turn. Neither can be
    vouched for here, and neither should be refused."""
    bare = agent_tools.LoopState(
        conversation_id="c", turn_id="t", question="q"
    )
    unknown, msg = agent_tools.check_document_ids_known(
        sql=_sql(MANGLED), state=bare
    )
    assert unknown == []
    assert msg is None


def test_one_bad_id_among_good_ones_is_still_caught() -> None:
    second = "a" * 64
    sql = (
        "SELECT 1 FROM `proj.raw.rows` AS r WHERE r.document_id IN "
        f"('{GOOD}', '{MANGLED}', '{second}') LIMIT 10"
    )
    unknown, _msg = agent_tools.check_document_ids_known(
        sql=sql, state=_state(GOOD, second)
    )
    assert [u["document_id"] for u in unknown] == [MANGLED]


def test_run_sql_refuses_before_executing() -> None:
    """The whole point: no BigQuery call, and a status the model can act
    on instead of an empty result set."""
    calls: list[str] = []

    class _Bq:
        def __getattr__(self, name: str):  # pragma: no cover - guard
            def _fail(*_a: object, **_k: object) -> object:
                calls.append(name)
                raise AssertionError(f"BigQuery was called: {name}")

            return _fail

    events: list[object] = []
    ctx = agent_tools.ToolContext(
        bq=_Bq(),  # type: ignore[arg-type]
        openai_client=None,  # type: ignore[arg-type]
        settings=agent_tools.Settings(
            gcp_project_id="proj",
            openai_api_key="sk-test",  # type: ignore[arg-type]
        ),
        state=_state(GOOD),
        emit=events.append,
    )
    result = agent_tools.run_run_sql(
        ctx=ctx, args={"sql": _sql(MANGLED), "rationale": "breakdown"}
    )
    assert result["status"] == "unknown_document_id"
    assert GOOD in result["message"]
    assert calls == []
    # Refused before execution, so it costs no SQL budget.
    assert ctx.state.sql_execution_count == 0
