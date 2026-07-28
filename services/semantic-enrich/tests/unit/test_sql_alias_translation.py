"""Rewriting recovered column names to the keys the stored rows hold.

Pure: no warehouse, no model. The rewrite is span-based, so the
assertions check that everything *except* the JSONPath segment survives
byte-for-byte — a generator round-trip would reformat SQL the model
wrote and the evidence rail would show the user something nobody typed.
"""
from __future__ import annotations

from semantic_enrich.core.sql_header_alias import apply_recovered_names

DOC = "doc-1"
OTHER = "doc-2"
RECOVERED = {DOC: {"__col_1": "Number of Unique Applicants",
                   "__col_2": "Total Amount ($000)"}}


def _sql(select: str, doc_ids: str = f"'{DOC}'") -> str:
    return (
        f"SELECT {select} FROM `p.raw.rows` "
        f"WHERE document_id IN ({doc_ids}) LIMIT 10"
    )


def test_recovered_name_becomes_its_positional_key() -> None:
    result = apply_recovered_names(
        _sql("""JSON_VALUE(row, '$."Total Amount ($000)"') AS amt"""),
        recovered_names=RECOVERED,
        header_rows={},
    )
    assert "'$.__col_2'" in result.sql
    assert "Total Amount ($000)" not in result.sql
    assert result.translated == {"Total Amount ($000)": "__col_2"}


def test_positional_key_passes_through_untouched() -> None:
    sql = _sql("SUM(SAFE_CAST(JSON_VALUE(row, '$.__col_2') AS FLOAT64))")
    result = apply_recovered_names(
        sql, recovered_names=RECOVERED, header_rows={}
    )
    assert result.sql == sql
    assert result.translated == {}


def test_a_name_matching_no_recovery_is_left_for_the_guard() -> None:
    """Rewriting only what it knows means an invented name still reaches
    the pairing check as an invented name."""
    sql = _sql("""JSON_VALUE(row, '$."Invented Column"')""")
    result = apply_recovered_names(
        sql, recovered_names=RECOVERED, header_rows={}
    )
    assert result.sql == sql
    assert result.translated == {}


def test_several_references_in_one_statement_all_translate() -> None:
    result = apply_recovered_names(
        _sql(
            """SUM(SAFE_CAST(JSON_VALUE(row, '$."Total Amount ($000)"') """
            """AS FLOAT64)), """
            """SUM(SAFE_CAST(JSON_VALUE(row, """
            """'$."Number of Unique Applicants"') AS FLOAT64)), """
            """JSON_VALUE(row, '$."Total Amount ($000)"')"""
        ),
        recovered_names=RECOVERED,
        header_rows={},
    )
    assert result.sql.count("'$.__col_2'") == 2
    assert result.sql.count("'$.__col_1'") == 1
    assert len(result.translated) == 2


def test_only_the_jsonpath_segment_changes() -> None:
    """Everything around the rewrite is preserved verbatim, including
    formatting the model chose."""
    sql = (
        "SELECT   /* keep me */\n"
        """  SUM(SAFE_CAST(JSON_VALUE(r.row, '$."Total Amount ($000)"') """
        "AS FLOAT64)) AS total\n"
        f"FROM `p.raw.rows` r WHERE document_id IN ('{DOC}')  LIMIT   5"
    )
    result = apply_recovered_names(
        sql, recovered_names=RECOVERED, header_rows={}
    )
    assert result.sql == sql.replace(
        """'$."Total Amount ($000)"'""", "'$.__col_2'"
    )


def test_a_document_not_inlined_does_not_lend_its_names() -> None:
    """Recovery is per document. A name recovered for doc-1 must not
    rewrite a query that only inlines doc-2."""
    sql = _sql("""JSON_VALUE(row, '$."Total Amount ($000)"')""", f"'{OTHER}'")
    result = apply_recovered_names(
        sql, recovered_names=RECOVERED, header_rows={}
    )
    assert result.sql == sql
    assert result.translated == {}


def test_nothing_on_state_is_a_no_op() -> None:
    sql = _sql("""JSON_VALUE(row, '$."Total Amount ($000)"')""")
    result = apply_recovered_names(sql, recovered_names={}, header_rows={})
    assert result.sql == sql
    assert result.translated == {}
    assert result.conflicts == ()


def test_sql_with_no_inlined_documents_is_left_alone() -> None:
    sql = "SELECT 1"
    result = apply_recovered_names(
        sql, recovered_names=RECOVERED, header_rows={}
    )
    assert result.sql == sql


# ── conflict ──


def test_two_documents_disagreeing_on_a_name_decline() -> None:
    result = apply_recovered_names(
        _sql(
            """JSON_VALUE(row, '$."Total"')""", f"'{DOC}', '{OTHER}'"
        ),
        recovered_names={
            DOC: {"__col_2": "Total"},
            OTHER: {"__col_5": "Total"},
        },
        header_rows={},
    )
    assert result.translated == {}
    assert [c.name for c in result.conflicts] == ["Total"]
    assert result.conflicts[0].keys_by_document == {
        DOC: "__col_2",
        OTHER: "__col_5",
    }


def test_a_real_column_conflicting_with_a_recovered_one_declines() -> None:
    """One document publishes `Total`; another recovers `Total` onto
    `__col_5`. Rewriting both would silently read different columns."""
    result = apply_recovered_names(
        _sql("""JSON_VALUE(row, '$."Total"')""", f"'{DOC}', '{OTHER}'"),
        recovered_names={OTHER: {"__col_5": "Total"}},
        header_rows={},
        doc_columns={DOC: ["Total", "Region"], OTHER: ["__col_5"]},
    )
    assert result.translated == {}
    assert [c.name for c in result.conflicts] == ["Total"]


def test_documents_that_agree_still_translate() -> None:
    result = apply_recovered_names(
        _sql("""JSON_VALUE(row, '$."Total"')""", f"'{DOC}', '{OTHER}'"),
        recovered_names={
            DOC: {"__col_2": "Total"},
            OTHER: {"__col_2": "Total"},
        },
        header_rows={},
    )
    assert result.conflicts == ()
    assert result.translated == {"Total": "__col_2"}


def test_a_conflict_on_an_unreferenced_name_is_not_a_conflict() -> None:
    """Only names the SQL actually uses can be ambiguous."""
    result = apply_recovered_names(
        _sql("""JSON_VALUE(row, '$.__col_9')""", f"'{DOC}', '{OTHER}'"),
        recovered_names={
            DOC: {"__col_2": "Total"},
            OTHER: {"__col_5": "Total"},
        },
        header_rows={},
    )
    assert result.conflicts == ()
