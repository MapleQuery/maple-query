from __future__ import annotations

import pytest

from curate.core.merge import GuardrailError, check_shrink, merge_sql


def test_merge_is_a_snapshot_with_a_change_check() -> None:
    sql = merge_sql(
        target="p.curated.people",
        staging="p.curated._stage",
        key=("person_id",),
        columns=["person_id", "name", "updated_at", "run_id"],
    )
    assert "ON T.`person_id` = S.`person_id`" in sql
    assert "WHEN NOT MATCHED BY SOURCE THEN DELETE" in sql
    # A re-run with identical data is a no-op: bookkeeping columns alone
    # never trigger an update.
    assert "TO_JSON_STRING(T.`name`)" in sql
    assert "TO_JSON_STRING(T.`run_id`)" not in sql


def test_unsafe_identifiers_are_refused() -> None:
    with pytest.raises(ValueError):
        merge_sql(target="p.curated.people; DROP", staging="s", key=("k",), columns=["k"])


def test_shrink_guardrail() -> None:
    check_shrink(table="t", current=0, snapshot=0, allow=False)
    check_shrink(table="t", current=100, snapshot=60, allow=False)
    with pytest.raises(GuardrailError):
        check_shrink(table="t", current=100, snapshot=40, allow=False)
    check_shrink(table="t", current=100, snapshot=40, allow=True)
