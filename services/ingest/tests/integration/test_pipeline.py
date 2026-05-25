"""Integration test for the pipeline using in-memory fakes.

Covers: happy path, GCS-based dedup on re-run, dry-run, format filter,
EN/FR pairing, limit-orgs filter, and the JSONL run-log writer.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from ingest.clients.ckan import Dataset
from ingest.clients.gcs import IdempotentSkip, PathCollision, Uploaded
from ingest.clients.http import Downloaded
from ingest.config.settings import Settings
from ingest.config.sources import SourcesConfig
from ingest.core.pipeline import RunRequest, run
from ingest.core.runlog import RunLogWriter
from ingest.providers.logging import configure_logging

configure_logging(dev=False)


# --- Fakes ----------------------------------------------------------------


@dataclass
class FakeHttp:
    bodies: dict[str, bytes]

    def download(self, url: str, *, etag=None, last_modified=None):
        body = self.bodies.get(url)
        if body is None:
            raise RuntimeError(f"FakeHttp has no body for {url}")
        return Downloaded(
            body=body, status=200, headers={"Content-Type": "text/csv"}, elapsed_ms=1
        )


@dataclass
class FakeCkan:
    datasets: list[Dataset]
    calls: list[dict] = field(default_factory=list)

    def search(
        self, *, subject, formats=None, organization=None, since=None, page_size=200
    ) -> Iterator[Dataset]:
        self.calls.append(
            {"subject": subject, "formats": formats, "organization": organization, "since": since}
        )
        for d in self.datasets:
            if organization and d.organization_code != organization:
                continue
            yield d


@dataclass
class FakeGcs:
    uploads: dict[str, bytes] = field(default_factory=dict)

    def upload(self, *, object_name, body, content_type=None):
        gcs_uri = f"gs://test-bucket/{object_name}"
        if object_name in self.uploads:
            if self.uploads[object_name] == body:
                return IdempotentSkip(gcs_uri=gcs_uri, existing_md5_b64="x")
            return PathCollision(gcs_uri=gcs_uri, existing_md5_b64="a", attempted_md5_b64="b")
        self.uploads[object_name] = body
        return Uploaded(gcs_uri=gcs_uri, generation=1)


# --- Fixtures -------------------------------------------------------------


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        gcp_project_id="test-project",
        gcs_bucket="test-bucket",
        runlog_dir=tmp_path / "runlog",
    )


@pytest.fixture
def sources() -> SourcesConfig:
    return SourcesConfig.model_validate(
        [
            {
                "country": "ca",
                "source": "ckan-opencanada",
                "api_base": "https://x.example/api/3/action",
                "organizations": [{"code": "fin"}],
            },
        ]
    )


@pytest.fixture
def runlog_path(settings: Settings) -> Path:
    return settings.runlog_dir / f"{settings.run_id}.jsonl"


def _make_dataset(*, id_: str, org: str, subjects: list[str], resources: list[dict]) -> Dataset:
    return Dataset.model_validate(
        {
            "id": id_,
            "name": id_,
            "title": f"Dataset {id_}",
            "organization": {"name": org},
            "metadata_modified": "2026-05-01T00:00:00",
            "subject": subjects,
            "resources": resources,
        }
    )


def _read_runlog(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# --- Tests ----------------------------------------------------------------


def test_happy_path_uploads_and_writes_runlog(
    settings: Settings, sources: SourcesConfig, runlog_path: Path
) -> None:
    csv_url = "https://x.example/data/report-en.csv"
    csv_body = b"a,b,c\n1,2,3\n"

    ds = _make_dataset(
        id_="d1",
        org="fin",
        subjects=["government_and_politics"],
        resources=[
            {"id": "r1", "url": csv_url, "name": "report", "format": "CSV", "language": ["en"]},
        ],
    )
    ckan = FakeCkan(datasets=[ds])
    http = FakeHttp(bodies={csv_url: csv_body})
    gcs = FakeGcs()

    with RunLogWriter(path=runlog_path) as runlog:
        summary = run(
            settings=settings,
            sources=sources,
            request=RunRequest(subject="government_and_politics", formats=("csv",)),
            ckans={"ckan-opencanada": ckan},
            http=http,
            gcs=gcs,
            runlog=runlog,
        )

    assert summary.success == 1
    assert summary.quarantined == 0
    assert summary.failed == 0
    assert summary.skipped_by_gcs_dedup == 0
    assert len(gcs.uploads) == 1

    entries = _read_runlog(runlog_path)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["ingestion_status"] == "success"
    assert entry["subjects"] == ["government_and_politics"]
    assert entry["file_format"] == "csv"
    assert entry["language"] == "en"
    assert entry["source_url"] == csv_url
    assert entry["gcs_uri"].startswith("gs://test-bucket/raw/")


def test_rerun_is_idempotent_via_gcs(
    settings: Settings, sources: SourcesConfig, runlog_path: Path
) -> None:
    csv_url = "https://x.example/data/report-en.csv"
    csv_body = b"a,b,c\n1,2,3\n"
    ds = _make_dataset(
        id_="d1",
        org="fin",
        subjects=["x"],
        resources=[{"id": "r1", "url": csv_url, "format": "CSV", "language": ["en"]}],
    )
    ckan = FakeCkan(datasets=[ds])
    http = FakeHttp(bodies={csv_url: csv_body})
    gcs = FakeGcs()

    with RunLogWriter(path=runlog_path) as runlog:
        run(
            settings=settings, sources=sources,
            request=RunRequest(subject="x", formats=("csv",)),
            ckans={"ckan-opencanada": ckan}, http=http, gcs=gcs, runlog=runlog,
        )

    uploads_after_first = dict(gcs.uploads)

    # Second pass — same bytes, same path; GCS returns IdempotentSkip.
    with RunLogWriter(path=runlog_path) as runlog:
        summary2 = run(
            settings=settings, sources=sources,
            request=RunRequest(subject="x", formats=("csv",)),
            ckans={"ckan-opencanada": ckan}, http=http, gcs=gcs, runlog=runlog,
        )

    assert summary2.skipped_by_gcs_dedup == 1
    assert summary2.success == 0
    assert gcs.uploads == uploads_after_first


def test_dry_run_writes_nothing(
    settings: Settings, sources: SourcesConfig, runlog_path: Path
) -> None:
    csv_url = "https://x.example/data/report-en.csv"
    ds = _make_dataset(
        id_="d1",
        org="fin",
        subjects=["x"],
        resources=[{"id": "r1", "url": csv_url, "format": "CSV", "language": ["en"]}],
    )
    ckan = FakeCkan(datasets=[ds])
    http = FakeHttp(bodies={csv_url: b"a,b\n1,2\n"})
    gcs = FakeGcs()

    with RunLogWriter(path=runlog_path) as runlog:
        summary = run(
            settings=settings, sources=sources,
            request=RunRequest(subject="x", formats=("csv",), dry_run=True),
            ckans={"ckan-opencanada": ckan}, http=http, gcs=gcs, runlog=runlog,
        )

    assert summary.success == 1
    assert gcs.uploads == {}
    assert _read_runlog(runlog_path) == []


def test_format_filter_drops_non_matching_siblings(
    settings: Settings, sources: SourcesConfig, runlog_path: Path
) -> None:
    csv_url = "https://x.example/a-en.csv"
    xml_url = "https://x.example/a-en.xml"
    ds = _make_dataset(
        id_="d1",
        org="fin",
        subjects=["x"],
        resources=[
            {"id": "r-csv", "url": csv_url, "format": "CSV", "language": ["en"]},
            {"id": "r-xml", "url": xml_url, "format": "XML", "language": ["en"]},
        ],
    )
    ckan = FakeCkan(datasets=[ds])
    http = FakeHttp(bodies={csv_url: b"a\n1\n", xml_url: b"<root/>"})
    gcs = FakeGcs()

    with RunLogWriter(path=runlog_path) as runlog:
        run(
            settings=settings, sources=sources,
            request=RunRequest(subject="x", formats=("csv",)),
            ckans={"ckan-opencanada": ckan}, http=http, gcs=gcs, runlog=runlog,
        )

    entries = _read_runlog(runlog_path)
    assert len(entries) == 1
    assert entries[0]["source_url"] == csv_url


def test_french_sibling_skipped_when_english_present(
    settings: Settings, sources: SourcesConfig, runlog_path: Path
) -> None:
    en_url = "https://x.example/a-en.csv"
    fr_url = "https://x.example/a-fr.csv"
    ds = _make_dataset(
        id_="d1",
        org="fin",
        subjects=["x"],
        resources=[
            {"id": "en", "url": en_url, "format": "CSV", "language": ["en"]},
            {"id": "fr", "url": fr_url, "format": "CSV", "language": ["fr"]},
        ],
    )
    ckan = FakeCkan(datasets=[ds])
    http = FakeHttp(bodies={en_url: b"x\n", fr_url: b"y\n"})
    gcs = FakeGcs()

    with RunLogWriter(path=runlog_path) as runlog:
        summary = run(
            settings=settings, sources=sources,
            request=RunRequest(subject="x", formats=("csv",)),
            ckans={"ckan-opencanada": ckan}, http=http, gcs=gcs, runlog=runlog,
        )

    assert summary.success == 1
    assert summary.skipped_by_pairing == 1
    entries = _read_runlog(runlog_path)
    assert entries[0]["language"] == "en"


def test_limit_orgs_filters_out_unrequested(
    settings: Settings, sources: SourcesConfig, runlog_path: Path
) -> None:
    ckan = FakeCkan(datasets=[])
    with RunLogWriter(path=runlog_path) as runlog:
        summary = run(
            settings=settings, sources=sources,
            request=RunRequest(subject="x", limit_orgs=("statcan",)),
            ckans={"ckan-opencanada": ckan}, http=FakeHttp({}), gcs=FakeGcs(), runlog=runlog,
        )
    assert summary.datasets_seen == 0
    assert summary.success == 0
    assert ckan.calls == []
    assert _read_runlog(runlog_path) == []
