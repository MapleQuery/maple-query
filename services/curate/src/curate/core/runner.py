"""Orchestration: fetch, build, then write each table (stage + MERGE),
or, with --dry-run, write JSONL files to a directory instead of
touching BigQuery."""

from __future__ import annotations

import csv
import dataclasses
import io
import json
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import IO, Any

from curate.clients.bq import BqClient
from curate.clients.gcs import GcsClient
from curate.clients.openparliament import OpenParliamentClient
from curate.clients.ourcommons import OurCommonsClient
from curate.config.settings import Settings
from curate.core import contributions as contrib
from curate.core import expenses as exp
from curate.core import people as people_mod
from curate.core.merge import (
    PEOPLE,
    PERSON_CONTRIBUTIONS,
    PERSON_EXPENSES,
    PERSON_NAMES,
    PERSON_TERMS,
    TableSpec,
    check_shrink,
    merge_sql,
)
from curate.core.schema_loader import load_schema
from curate.types import PersonName, PersonTerm

SITE = people_mod.SITE


@dataclass
class Deps:
    settings: Settings
    log: Any
    op: OpenParliamentClient | None = None
    bq: BqClient | None = None
    gcs: GcsClient | None = None
    ourcommons: OurCommonsClient | None = None


@dataclass
class Plan:
    dry_run: bool
    out_dir: Path | None = None
    allow_shrink: bool = False
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))


def _jsonable(obj: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in dataclasses.asdict(obj).items():
        out[k] = v.isoformat() if isinstance(v, date) else v
    return out


def write_table(deps: Deps, plan: Plan, spec: TableSpec, records: list[Any], *, stamped: bool) -> int:
    rows = [_jsonable(r) for r in records]
    for r in rows:
        r["run_id"] = deps.settings.run_id
        if stamped:
            r["updated_at"] = plan.started_at.isoformat()
    if plan.dry_run:
        if plan.out_dir is None:
            raise ValueError("--dry-run needs --out DIR")
        plan.out_dir.mkdir(parents=True, exist_ok=True)
        path = plan.out_dir / f"{spec.name}.jsonl"
        with path.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        deps.log.info("would_have_written", table=spec.name, rows=len(rows), path=str(path))
        return len(rows)

    assert deps.bq is not None
    s = deps.settings
    schema = load_schema(s.schemas_dir / spec.schema_file)
    target = f"{s.gcp_project_id}.{s.bq_dataset_curated}.{spec.name}"
    staging = f"{s.gcp_project_id}.{s.bq_dataset_curated}._stage_{spec.name}_{s.run_id[:12]}"
    current = deps.bq.count_rows(target)
    check_shrink(table=spec.name, current=current, snapshot=len(rows), allow=plan.allow_shrink)
    deps.bq.load_staging(
        table_id=staging, rows=rows, schema=schema, expires_in=timedelta(hours=s.staging_ttl_hours)
    )
    deps.bq.execute(merge_sql(target=target, staging=staging, key=spec.key, columns=[f.name for f in schema]))
    deps.log.info("table_merged", table=spec.name, rows=len(rows), previous=current)
    return len(rows)


# ── people ──


def _stored_details(deps: Deps) -> dict[str, dict[str, Any]]:
    """What a previous run stored, shaped like detail objects, so this
    run re-fetches only people who are new or still sitting."""
    if deps.bq is None:
        return {}
    s = deps.settings
    ds = f"`{s.gcp_project_id}.{s.bq_dataset_curated}"
    details: dict[str, dict[str, Any]] = {}
    for r in deps.bq.query_rows(
        f"SELECT person_id, openparliament_url, given_name, family_name FROM {ds}.people` "
        "WHERE given_name IS NOT NULL"
    ):
        url = str(r["openparliament_url"]).removeprefix(SITE)
        details[url] = {
            "given_name": r["given_name"],
            "family_name": r["family_name"],
            "other_info": {},
            "memberships": [],
        }
    by_pid = {people_mod.person_id_for(u): u for u in details}
    for r in deps.bq.query_rows(
        f"SELECT person_id, name_raw FROM {ds}.person_names` WHERE origin = 'alternate_name'"
    ):
        known_url = by_pid.get(str(r["person_id"]))
        if known_url:
            details[known_url]["other_info"].setdefault("alternate_name", []).append(r["name_raw"])
    for r in deps.bq.query_rows(
        f"SELECT term_id, person_id, start_date, end_date, party, riding_name, province, label "
        f"FROM {ds}.person_terms`"
    ):
        known_url = by_pid.get(str(r["person_id"]))
        if known_url:
            details[known_url]["memberships"].append(
                {
                    "url": r["term_id"],
                    "start_date": str(r["start_date"]),
                    "end_date": str(r["end_date"]) if r.get("end_date") else None,
                    "party": {"short_name": {"en": r.get("party")}},
                    "riding": {"name": {"en": r.get("riding_name")}, "province": r.get("province")},
                    "label": {"en": r.get("label")},
                }
            )
    return details


def run_people(deps: Deps, plan: Plan, *, max_detail: int | None = None) -> dict[str, int]:
    assert deps.op is not None
    politicians = deps.op.list_all("/politicians/?include=all")
    sitting = deps.op.list_all("/politicians/")
    if not politicians or not sitting:
        raise RuntimeError("openparliament.ca returned an empty politician list")
    current = {str(p.get("url")) for p in sitting}
    details = {} if plan.dry_run else _stored_details(deps)
    need = [
        str(p["url"]) for p in politicians if str(p.get("url")) not in details or str(p.get("url")) in current
    ]
    if max_detail is not None:
        need = need[:max_detail]
    for url in need:
        details[url] = deps.op.detail(url)
    deps.log.info(
        "people_fetched", politicians=len(politicians), sitting=len(sitting), details_fetched=len(need)
    )
    tables = people_mod.build(politicians, details)
    return {
        "people": write_table(deps, plan, PEOPLE, tables.people, stamped=True),
        "person_names": write_table(deps, plan, PERSON_NAMES, tables.names, stamped=False),
        "person_terms": write_table(deps, plan, PERSON_TERMS, tables.terms, stamped=False),
    }


# ── contributions ──


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _index(deps: Deps, people_dir: Path | None) -> contrib.NameIndex:
    if people_dir is not None:
        names_rows = list(_read_jsonl(people_dir / "person_names.jsonl"))
        terms_rows = list(_read_jsonl(people_dir / "person_terms.jsonl"))
    else:
        assert deps.bq is not None
        s = deps.settings
        ds = f"`{s.gcp_project_id}.{s.bq_dataset_curated}"
        names_rows = list(
            deps.bq.query_rows(f"SELECT person_id, name_norm, name_raw, origin FROM {ds}.person_names`")
        )
        terms_rows = list(
            deps.bq.query_rows(
                f"SELECT term_id, person_id, start_date, end_date, party FROM {ds}.person_terms`"
            )
        )
    names = [
        PersonName(r["person_id"], r["name_norm"], r.get("name_raw", ""), r.get("origin", ""))
        for r in names_rows
    ]

    def d(v: Any) -> date | None:
        if v is None or isinstance(v, date):
            return v
        return date.fromisoformat(str(v))

    terms = []
    for r in terms_rows:
        start = d(r["start_date"])
        if start is None:
            continue
        terms.append(
            PersonTerm(
                str(r["term_id"]),
                str(r["person_id"]),
                "MP",
                start,
                d(r.get("end_date")),
                r.get("party"),
                None,
                None,
                None,
            )
        )
    if not names:
        raise RuntimeError("no people to link to: run `curate people` first")
    return contrib.NameIndex.build(names, terms)


def _csv_rows(archive: IO[bytes]) -> Iterator[dict[str, str]]:
    with zipfile.ZipFile(archive) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise contrib.SourceShapeError(f"expected one CSV in the archive, found {members}")
        with zf.open(members[0]) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            yield from csv.DictReader(text)


def run_contributions(
    deps: Deps, plan: Plan, *, source_file: Path | None = None, people_dir: Path | None = None
) -> dict[str, Any]:
    index = _index(deps, people_dir)
    s = deps.settings
    if source_file is not None:
        source_object = f"file://{source_file.resolve()}"
        archive: IO[bytes] = source_file.open("rb")
    else:
        assert deps.gcs is not None
        name = deps.gcs.newest_object(prefix=s.contributions_prefix, contains=s.contributions_object_contains)
        if name is None:
            raise RuntimeError(
                f"no object containing {s.contributions_object_contains!r} under {s.contributions_prefix}: "
                "run ingest with --limit-orgs elections --accept-archives first"
            )
        source_object = f"gs://{s.gcs_bucket}/{name}"
        archive = deps.gcs.open_binary(name)
    stats = contrib.LinkStats()
    with archive:
        records = list(contrib.link(_csv_rows(archive), index, source_object=source_object, stats=stats))
    deps.log.info(
        "contributions_linked",
        source=source_object,
        rows_read=stats.rows_read,
        person_rows=stats.person_rows,
        linked_rows=stats.linked_rows,
        ambiguous_rows=stats.ambiguous_rows,
        unmatched_rows=stats.unmatched_rows,
        skipped=dict(stats.skipped),
        groups=len(records),
    )
    written = write_table(deps, plan, PERSON_CONTRIBUTIONS, records, stamped=False)
    return {
        "person_contributions": written,
        "rows_read": stats.rows_read,
        "linked_rows": stats.linked_rows,
        "ambiguous_rows": stats.ambiguous_rows,
        "unmatched_rows": stats.unmatched_rows,
        "skipped": dict(stats.skipped),
    }


# ── expenses ──


def run_expenses(
    deps: Deps, plan: Plan, *, people_dir: Path | None = None, max_quarters: int | None = None
) -> dict[str, Any]:
    assert deps.ourcommons is not None
    index = _index(deps, people_dir)
    refs = deps.ourcommons.quarters()
    if max_quarters is not None:
        refs = refs[-max_quarters:]
    stats = exp.ExpenseStats()
    records: list[Any] = []
    for ref in refs:
        records.extend(exp.link(deps.ourcommons.report(ref), index, stats))
    deps.log.info(
        "expenses_linked",
        quarters=len(refs),
        rows_read=stats.rows_read,
        linked=stats.linked,
        ambiguous=stats.ambiguous,
        top_unmatched=stats.unmatched_names.most_common(25),
        unmatched=stats.unmatched,
        skipped=dict(stats.skipped),
    )
    written = write_table(deps, plan, PERSON_EXPENSES, records, stamped=False)
    return {
        "person_expenses": written,
        "quarters": len(refs),
        "rows_read": stats.rows_read,
        "linked": stats.linked,
        "ambiguous": stats.ambiguous,
        "unmatched": stats.unmatched,
        "skipped": dict(stats.skipped),
    }
