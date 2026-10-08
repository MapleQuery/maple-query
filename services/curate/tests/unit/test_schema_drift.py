"""The dataclasses and infra/terraform/schemas/curated_*.json describe the
same tables; a field added on one side only fails here."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from curate.core.merge import PEOPLE, PERSON_CONTRIBUTIONS, PERSON_NAMES, PERSON_TERMS, TableSpec
from curate.types import Person, PersonContribution, PersonName, PersonTerm

SCHEMAS = Path(__file__).resolve().parents[4] / "infra" / "terraform" / "schemas"
BOOKKEEPING = {"run_id", "updated_at"}


@pytest.mark.parametrize(
    ("spec", "cls"),
    [
        (PEOPLE, Person),
        (PERSON_NAMES, PersonName),
        (PERSON_TERMS, PersonTerm),
        (PERSON_CONTRIBUTIONS, PersonContribution),
    ],
)
def test_dataclass_matches_schema(spec: TableSpec, cls: type) -> None:
    schema = json.loads((SCHEMAS / spec.schema_file).read_text())
    schema_cols = [f["name"] for f in schema if f["name"] not in BOOKKEEPING]
    assert schema_cols == [f.name for f in dataclasses.fields(cls)]
    assert set(spec.key) <= set(schema_cols)
