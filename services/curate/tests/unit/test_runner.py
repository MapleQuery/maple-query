from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import timedelta
from pathlib import Path
from typing import IO, Any

import pytest

from curate.config.settings import Settings
from curate.core.merge import GuardrailError
from curate.core.runner import Deps, Plan, run_contributions, run_people
from curate.providers.logging import get_logger
from tests.unit.test_contributions import HEADER
from tests.unit.test_people import DETAILS, MEMBERSHIPS, POLITICIANS


class FakeOp:
    def __init__(self) -> None:
        self.detail_calls: list[str] = []

    def list_all(self, path: str) -> list[dict[str, Any]]:
        return POLITICIANS if path.startswith("/politicians/?") else MEMBERSHIPS

    def detail(self, path: str) -> dict[str, Any]:
        self.detail_calls.append(path)
        return DETAILS.get(path, {})


class FakeBq:
    def __init__(
        self, counts: dict[str, int] | None = None, rows: dict[str, list[dict[str, Any]]] | None = None
    ):
        self.counts = counts or {}
        self.rows = rows or {}
        self.loaded: dict[str, list[dict[str, Any]]] = {}
        self.sql: list[str] = []

    def load_staging(
        self, *, table_id: str, rows: list[dict[str, Any]], schema: Any, expires_in: timedelta
    ) -> int:
        self.loaded[table_id] = rows
        return len(rows)

    def execute(self, sql: str) -> None:
        self.sql.append(sql)

    def query_rows(self, sql: str) -> Any:
        for name, rows in self.rows.items():
            if f".{name}`" in sql:
                return iter(rows)
        return iter([])

    def count_rows(self, table_id: str) -> int:
        return self.counts.get(table_id.rsplit(".", 1)[-1], 0)


def _settings() -> Settings:
    return Settings(gcp_project_id="proj", run_id="run1234567890abcd")


def test_people_dry_run_writes_jsonl_and_touches_no_bigquery(tmp_path: Path) -> None:
    deps = Deps(settings=_settings(), log=get_logger("t"), op=FakeOp())
    out = run_people(deps, Plan(dry_run=True, out_dir=tmp_path))
    # Poilievre: display name + alternate names, which all normalise to
    # one form (one row per origin); 438 and never-elected: display name.
    assert out == {"people": 3, "person_names": 4, "person_terms": 3}
    people = [json.loads(x) for x in (tmp_path / "people.jsonl").read_text().splitlines()]
    assert {p["person_id"] for p in people} == {"op:pierre-poilievre", "op:438", "op:never-elected"}
    assert all(p["run_id"] == "run1234567890abcd" and p["updated_at"] for p in people)


def test_people_real_run_stages_and_merges_each_table() -> None:
    bq = FakeBq()
    op = FakeOp()
    deps = Deps(settings=_settings(), log=get_logger("t"), op=op, bq=bq)
    run_people(deps, Plan(dry_run=False))
    assert {t.rsplit(".", 1)[-1].split("_run")[0] for t in bq.loaded} == {
        "_stage_people",
        "_stage_person_names",
        "_stage_person_terms",
    }
    assert len(bq.sql) == 3 and all(s.startswith("MERGE `proj.curated.") for s in bq.sql)
    # Nothing stored yet, so every politician's detail was fetched.
    assert len(op.detail_calls) == 3


def test_people_fetches_detail_only_for_new_people_and_sitting_mps() -> None:
    known = {
        "people": [
            {
                "person_id": "op:438",
                "openparliament_url": "https://openparliament.ca/politicians/438/",
                "given_name": "Fabian",
                "family_name": "Manning",
            },
            {
                "person_id": "op:pierre-poilievre",
                "openparliament_url": "https://openparliament.ca/politicians/pierre-poilievre/",
                "given_name": "Pierre",
                "family_name": "Poilievre",
            },
        ]
    }
    op = FakeOp()
    run_people(
        Deps(settings=_settings(), log=get_logger("t"), op=op, bq=FakeBq(rows=known)), Plan(dry_run=False)
    )
    # 438 is stored and not sitting -> skipped; Poilievre is sitting -> refreshed.
    assert sorted(op.detail_calls) == ["/politicians/never-elected/", "/politicians/pierre-poilievre/"]


def test_shrink_guardrail_stops_a_partial_upstream_list() -> None:
    bq = FakeBq(counts={"people": 1000})
    with pytest.raises(GuardrailError):
        run_people(Deps(settings=_settings(), log=get_logger("t"), op=FakeOp(), bq=bq), Plan(dry_run=False))
    assert bq.sql == []


def _zip(rows: list[dict[str, str]]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(HEADER))
    w.writeheader()
    for r in rows:
        w.writerow(r)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("PoliticalFinance/od_cntrbtn_de_e.csv", "﻿" + buf.getvalue())
    return out.getvalue()


class FakeGcs:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def newest_object(self, *, prefix: str, contains: str) -> str | None:
        return f"{prefix}resource_last_modified=2026-10-03/fmt=zip__id=abc__od_cntrbtn_de_e.zip"

    def open_binary(self, name: str) -> IO[bytes]:
        return io.BytesIO(self.body)


def test_contributions_from_people_dir_and_gcs_archive(tmp_path: Path) -> None:
    deps = Deps(settings=_settings(), log=get_logger("t"), op=FakeOp())
    run_people(deps, Plan(dry_run=True, out_dir=tmp_path))
    body = _zip(
        [
            {
                **HEADER,
                "Recipient": "Poilievre, Pierre",
                "Recipient last name": "Poilievre",
                "Recipient first name": "Pierre",
                "Fiscal/Election date": "2021-09-20",
            },
            {**HEADER, "Political Entity": "Registered parties"},
        ]
    )
    deps = Deps(settings=_settings(), log=get_logger("t"), gcs=FakeGcs(body))
    out = run_contributions(deps, Plan(dry_run=True, out_dir=tmp_path / "c"), people_dir=tmp_path)
    assert out["linked_rows"] == 1 and out["rows_read"] == 2
    [g] = [json.loads(x) for x in (tmp_path / "c" / "person_contributions.jsonl").read_text().splitlines()]
    assert g["person_id"] == "op:pierre-poilievre"
    assert g["source_object"].startswith("gs://maplequery-raw/raw/")
    assert g["contributor_name"] is None
