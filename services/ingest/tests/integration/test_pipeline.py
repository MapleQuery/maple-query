"""Integration test for the pipeline using in-memory fakes.

Covers: happy path, GCS-based dedup on re-run, dry-run, format filter,
EN/FR pairing, limit-orgs filter, and the JSONL run-log writer.
"""
from __future__ import annotations

import json
import os
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
    discover_calls: list[dict] = field(default_factory=list)

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

    def discover_organizations(
        self, *, subject, formats=None, since=None
    ) -> list[str]:
        self.discover_calls.append(
            {"subject": subject, "formats": formats, "since": since}
        )
        # Stable order for assertions.
        return sorted({d.organization_code for d in self.datasets})


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
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    for key in list(os.environ):
        if key.startswith("INGEST_"):
            monkeypatch.delenv(key)
    return Settings(
        gcp_project_id="test-project",
        gcs_bucket="test-bucket",
        runlog_dir=tmp_path / "runlog",
        _env_file=None,
    )


@pytest.fixture
def sources() -> SourcesConfig:
    return SourcesConfig.model_validate(
        [
            {
                "country": "ca",
                "source": "ckan-opencanada",
                "api_base": "https://x.example/api/3/action",
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
            "metadata_created": "2020-01-01T00:00:00",
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


def test_rerun_dedups_when_resource_lacks_last_modified_and_dataset_was_touched(
    settings: Settings, sources: SourcesConfig, runlog_path: Path
) -> None:
    """Regression: partition must be stable across runs when
    `resource.last_modified` is absent and the dataset's
    `metadata_modified` bumps between runs (e.g. sibling edited).
    Falls back to `metadata_created`, which is immutable.
    """
    csv_url = "https://x.example/data/report-en.csv"
    csv_body = b"a,b,c\n1,2,3\n"

    # Note: resource has no `last_modified` — forces fallback to a
    # dataset-level field.
    ds_v1 = _make_dataset(
        id_="d1",
        org="fin",
        subjects=["x"],
        resources=[{"id": "r1", "url": csv_url, "format": "CSV", "language": ["en"]}],
    )
    # Second pass: same bytes, same resource, but the dataset was
    # edited (sibling resource added, title fixed, anything) — so
    # `metadata_modified` is newer. `metadata_created` is unchanged.
    ds_v2 = Dataset.model_validate(
        {
            **ds_v1.model_dump(by_alias=True, mode="json"),
            "metadata_modified": "2026-06-15T00:00:00",
        }
    )
    assert ds_v2.metadata_modified != ds_v1.metadata_modified
    assert ds_v2.metadata_created == ds_v1.metadata_created

    http = FakeHttp(bodies={csv_url: csv_body})
    gcs = FakeGcs()

    with RunLogWriter(path=runlog_path) as runlog:
        run(
            settings=settings, sources=sources,
            request=RunRequest(subject="x", formats=("csv",)),
            ckans={"ckan-opencanada": FakeCkan(datasets=[ds_v1])},
            http=http, gcs=gcs, runlog=runlog,
        )

    uploads_after_first = dict(gcs.uploads)

    with RunLogWriter(path=runlog_path) as runlog:
        summary2 = run(
            settings=settings, sources=sources,
            request=RunRequest(subject="x", formats=("csv",)),
            ckans={"ckan-opencanada": FakeCkan(datasets=[ds_v2])},
            http=http, gcs=gcs, runlog=runlog,
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


def test_limit_orgs_pins_set_and_skips_discovery(
    settings: Settings, sources: SourcesConfig, runlog_path: Path
) -> None:
    """`--limit-orgs` overrides discovery — only the listed orgs are
    iterated, and `discover_organizations` isn't called at all."""
    csv_url = "https://x.example/data/report-en.csv"
    ds_fin = _make_dataset(
        id_="d_fin", org="fin", subjects=["x"],
        resources=[{"id": "r_fin", "url": csv_url, "format": "CSV", "language": ["en"]}],
    )
    ds_stat = _make_dataset(
        id_="d_stat", org="statcan", subjects=["x"],
        resources=[{"id": "r_stat", "url": csv_url + "?2", "format": "CSV", "language": ["en"]}],
    )
    ckan = FakeCkan(datasets=[ds_fin, ds_stat])
    http = FakeHttp(bodies={csv_url: b"a\n", csv_url + "?2": b"b\n"})

    with RunLogWriter(path=runlog_path) as runlog:
        summary = run(
            settings=settings, sources=sources,
            request=RunRequest(subject="x", limit_orgs=("statcan",)),
            ckans={"ckan-opencanada": ckan}, http=http, gcs=FakeGcs(), runlog=runlog,
        )

    # Only statcan was iterated — fin's dataset is invisible.
    assert summary.datasets_seen == 1
    assert summary.success == 1
    assert ckan.discover_calls == []
    assert {c["organization"] for c in ckan.calls} == {"statcan"}


def test_default_discovers_orgs_from_ckan_facet(
    settings: Settings, sources: SourcesConfig, runlog_path: Path
) -> None:
    """With no --limit-orgs, the pipeline calls `discover_organizations`
    and iterates whatever it returns."""
    csv_url_a = "https://x.example/data/a.csv"
    csv_url_b = "https://x.example/data/b.csv"
    ds_fin = _make_dataset(
        id_="d_fin", org="fin", subjects=["x"],
        resources=[{"id": "r_fin", "url": csv_url_a, "format": "CSV", "language": ["en"]}],
    )
    ds_dfo = _make_dataset(
        id_="d_dfo", org="dfo-mpo", subjects=["x"],
        resources=[{"id": "r_dfo", "url": csv_url_b, "format": "CSV", "language": ["en"]}],
    )
    ckan = FakeCkan(datasets=[ds_fin, ds_dfo])
    http = FakeHttp(bodies={csv_url_a: b"a\n", csv_url_b: b"b\n"})

    with RunLogWriter(path=runlog_path) as runlog:
        summary = run(
            settings=settings, sources=sources,
            request=RunRequest(subject="x", formats=("csv",)),
            ckans={"ckan-opencanada": ckan}, http=http, gcs=FakeGcs(), runlog=runlog,
        )

    # Discovery ran exactly once for the source.
    assert len(ckan.discover_calls) == 1
    # Both orgs got iterated and ingested.
    assert summary.success == 2
    assert {c["organization"] for c in ckan.calls} == {"fin", "dfo-mpo"}


def test_sniff_skips_when_bytes_dont_match_format_filter(
    settings: Settings, sources: SourcesConfig, runlog_path: Path
) -> None:
    """Resources declared CSV but whose bytes sniff as HTML (landing pages),
    ZIP (bundles), or unknown must be skipped under `-f csv` — no upload,
    no run-log row."""
    html_url = "https://x.example/data/landing.csv"
    zip_url = "https://x.example/data/bundle.csv"
    unknown_url = "https://x.example/data/opaque"  # no extension → no URL fallback
    real_csv_url = "https://x.example/data/real.csv"

    ds = _make_dataset(
        id_="d1", org="fin", subjects=["x"],
        resources=[
            {"id": "html", "url": html_url, "format": "CSV", "language": ["en"]},
            {"id": "zip", "url": zip_url, "format": "CSV", "language": ["en"]},
            {"id": "unk", "url": unknown_url, "format": "CSV", "language": ["en"]},
            {"id": "csv", "url": real_csv_url, "format": "CSV", "language": ["en"]},
        ],
    )
    http = FakeHttp(bodies={
        html_url: b"<!DOCTYPE html><html><body>not a csv</body></html>",
        zip_url: b"PK\x03\x04" + b"\x00" * 32,  # ZIP magic
        unknown_url: b"\x00\x01\x02\x03opaque-bytes",
        real_csv_url: b"a,b,c\n1,2,3\n",
    })
    gcs = FakeGcs()

    with RunLogWriter(path=runlog_path) as runlog:
        summary = run(
            settings=settings, sources=sources,
            request=RunRequest(subject="x", formats=("csv",)),
            ckans={"ckan-opencanada": FakeCkan(datasets=[ds])},
            http=http, gcs=gcs, runlog=runlog,
        )

    # Only the real csv lands; three resources skipped at the sniff gate.
    assert summary.success == 1
    assert summary.skipped_by_format == 3
    assert summary.quarantined == 0
    assert summary.failed == 0
    assert len(gcs.uploads) == 1
    entries = _read_runlog(runlog_path)
    assert len(entries) == 1
    assert entries[0]["source_url"] == real_csv_url
