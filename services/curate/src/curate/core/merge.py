"""Stage-then-MERGE for curated tables, with snapshot semantics.

Every curate run computes a table's COMPLETE contents (all people, all
terms, every linked contribution), so the MERGE makes the target equal
to the staging snapshot: update changed rows, insert new ones, delete
rows the snapshot no longer has. Re-running with the same inputs is a
no-op.

The delete arm is guarded like warehouse-load's mass-blob-missing
guardrail: a snapshot less than half the size of the table it replaces
aborts the run (an upstream outage that returned a partial list must
not wipe the table).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_IDENT = re.compile(r"^[A-Za-z0-9_.-]+$")
SHRINK_FLOOR = 0.5


class GuardrailError(RuntimeError):
    """The snapshot would shrink the table past the guardrail."""


@dataclass(frozen=True)
class TableSpec:
    name: str  # table id within the curated dataset
    schema_file: str  # infra/terraform/schemas/<file>
    key: tuple[str, ...]


PEOPLE = TableSpec("people", "curated_people.json", ("person_id",))
PERSON_NAMES = TableSpec("person_names", "curated_person_names.json", ("person_id", "name_norm", "origin"))
PERSON_TERMS = TableSpec("person_terms", "curated_person_terms.json", ("term_id",))
PERSON_EXPENSES = TableSpec("person_expenses", "curated_person_expenses.json", ("expense_key",))
PERSON_CONTRIBUTIONS = TableSpec(
    "person_contributions", "curated_person_contributions.json", ("contribution_key",)
)


def _q(ident: str) -> str:
    if not _IDENT.match(ident):
        raise ValueError(f"unsafe identifier: {ident!r}")
    return f"`{ident}`"


def merge_sql(*, target: str, staging: str, key: tuple[str, ...], columns: list[str]) -> str:
    for c in columns:
        _q(c)
    on = " AND ".join(f"T.{_q(k)} = S.{_q(k)}" for k in key)
    non_key = [c for c in columns if c not in key]
    changed = (
        " OR ".join(
            f"NOT (TO_JSON_STRING(T.{_q(c)}) = TO_JSON_STRING(S.{_q(c)}))"
            for c in non_key
            if c not in ("run_id", "updated_at")
        )
        or "FALSE"
    )
    sets = ", ".join(f"{_q(c)} = S.{_q(c)}" for c in non_key)
    cols = ", ".join(_q(c) for c in columns)
    vals = ", ".join(f"S.{_q(c)}" for c in columns)
    return (
        f"MERGE {_q(target)} T USING {_q(staging)} S ON {on}\n"
        f"WHEN MATCHED AND ({changed}) THEN UPDATE SET {sets}\n"
        f"WHEN NOT MATCHED BY TARGET THEN INSERT ({cols}) VALUES ({vals})\n"
        f"WHEN NOT MATCHED BY SOURCE THEN DELETE"
    )


def check_shrink(*, table: str, current: int, snapshot: int, allow: bool) -> None:
    if allow or current == 0:
        return
    if snapshot < SHRINK_FLOOR * current:
        raise GuardrailError(
            f"{table}: snapshot has {snapshot} rows against {current} stored "
            f"(< {SHRINK_FLOOR:.0%}). Refusing to delete the difference; pass "
            "--allow-shrink if the source really lost that many rows."
        )
