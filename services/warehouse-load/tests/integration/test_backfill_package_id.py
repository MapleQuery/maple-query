"""Integration tests for the package_id backfill script.

Uses `FakeBqClient` from `tests/integration/conftest.py` and the
`requests-mock` adapter to stub the CKAN HTTP surface. The script
lives outside `src/` so it is imported via an explicit sys.path
insertion (matches how the script wires itself in production).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import requests_mock

from tests.integration.conftest import FakeBqClient

# tests/integration/.. -> tests/.. -> services/warehouse-load/scripts
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import backfill_package_id as bf  # noqa: E402

_CKAN_BASE = "https://ckan.test/api/3/action"
_TABLE = "test-project.raw.documents"


# --- helpers --------------------------------------------------------------


def _seed_target_row(
    bq: FakeBqClient,
    *,
    document_id: str,
    source_url: str,
    package_id: str | None = None,
) -> None:
    bq.target_rows[document_id] = {
        "document_id": document_id,
        "source_url": source_url,
        "package_id": package_id,
    }


def _seed_query_result(bq: FakeBqClient, *, rows: list[dict[str, str]]) -> None:
    """Pre-seed the FIFO query result the script's `fetch_targets` consumes."""
    bq.query_results.append(rows)


def _mock_package_list(m: requests_mock.Mocker, package_ids: list[str]) -> None:
    m.get(
        f"{_CKAN_BASE}/package_list",
        json={"success": True, "result": package_ids},
    )


def _mock_package_show(
    m: requests_mock.Mocker,
    *,
    package_id: str,
    resources: list[dict[str, str]],
    status_code: int = 200,
) -> None:
    if status_code == 404:
        m.get(
            f"{_CKAN_BASE}/package_show",
            status_code=404,
            json={"success": False, "error": {"__type": "Not Found Error"}},
        )
        return
    m.get(
        f"{_CKAN_BASE}/package_show",
        json={
            "success": True,
            "result": {"id": package_id, "resources": resources},
        },
    )


# --- tests ----------------------------------------------------------------


def test_backfill_writes_package_ids_from_walk() -> None:
    """Happy path: every NULL doc gets the matching package_id."""
    bq = FakeBqClient()
    _seed_target_row(bq, document_id="d1", source_url="https://ckan.test/a.csv")
    _seed_target_row(bq, document_id="d2", source_url="https://ckan.test/b.csv")
    _seed_target_row(bq, document_id="d3", source_url="https://ckan.test/c.csv")
    _seed_query_result(
        bq,
        rows=[
            {"document_id": "d1", "source_url": "https://ckan.test/a.csv"},
            {"document_id": "d2", "source_url": "https://ckan.test/b.csv"},
            {"document_id": "d3", "source_url": "https://ckan.test/c.csv"},
        ],
    )

    with requests_mock.Mocker() as m:
        _mock_package_list(m, ["pkg-a", "pkg-b"])
        m.get(
            f"{_CKAN_BASE}/package_show",
            [
                {
                    "json": {
                        "success": True,
                        "result": {
                            "id": "pkg-a",
                            "resources": [
                                {"url": "https://ckan.test/a.csv"},
                                {"url": "https://ckan.test/b.csv"},
                            ],
                        },
                    },
                },
                {
                    "json": {
                        "success": True,
                        "result": {
                            "id": "pkg-b",
                            "resources": [{"url": "https://ckan.test/c.csv"}],
                        },
                    },
                },
            ],
        )

        summary = bf.run(
            bq=bq,
            table=_TABLE,
            ckan_base=_CKAN_BASE,
            inter_request_delay=0,
            dry_run=False,
            reset=False,
            limit_packages=None,
            batch_size=500,
        )

    assert summary.packages_walked == 2
    assert summary.resources_indexed == 3
    assert summary.docs_seen == 3
    assert summary.docs_updated == 3
    assert summary.docs_missed == 0
    assert bq.target_rows["d1"]["package_id"] == "pkg-a"
    assert bq.target_rows["d2"]["package_id"] == "pkg-a"
    assert bq.target_rows["d3"]["package_id"] == "pkg-b"


def test_backfill_misses_unknown_urls() -> None:
    """A source_url absent from the walk is counted as a miss, left NULL."""
    bq = FakeBqClient()
    _seed_target_row(bq, document_id="d1", source_url="https://ckan.test/a.csv")
    _seed_target_row(bq, document_id="d2", source_url="https://other.test/elsewhere.csv")
    _seed_query_result(
        bq,
        rows=[
            {"document_id": "d1", "source_url": "https://ckan.test/a.csv"},
            {"document_id": "d2", "source_url": "https://other.test/elsewhere.csv"},
        ],
    )

    with requests_mock.Mocker() as m:
        _mock_package_list(m, ["pkg-a"])
        _mock_package_show(
            m,
            package_id="pkg-a",
            resources=[{"url": "https://ckan.test/a.csv"}],
        )

        summary = bf.run(
            bq=bq,
            table=_TABLE,
            ckan_base=_CKAN_BASE,
            inter_request_delay=0,
            dry_run=False,
            reset=False,
            limit_packages=None,
            batch_size=500,
        )

    assert summary.docs_updated == 1
    assert summary.docs_missed == 1
    assert bq.target_rows["d1"]["package_id"] == "pkg-a"
    assert bq.target_rows["d2"]["package_id"] is None


def test_backfill_idempotent_rerun() -> None:
    """Second run with no NULL rows updates zero."""
    bq = FakeBqClient()
    _seed_target_row(
        bq,
        document_id="d1",
        source_url="https://ckan.test/a.csv",
        package_id="pkg-a",  # already populated
    )
    # `fetch_targets` filters WHERE package_id IS NULL, so the seeded
    # query result is empty for the second run.
    _seed_query_result(bq, rows=[])

    with requests_mock.Mocker() as m:
        _mock_package_list(m, ["pkg-a"])
        _mock_package_show(
            m,
            package_id="pkg-a",
            resources=[{"url": "https://ckan.test/a.csv"}],
        )

        summary = bf.run(
            bq=bq,
            table=_TABLE,
            ckan_base=_CKAN_BASE,
            inter_request_delay=0,
            dry_run=False,
            reset=False,
            limit_packages=None,
            batch_size=500,
        )

    assert summary.docs_seen == 0
    assert summary.docs_updated == 0
    assert summary.docs_missed == 0
    assert bq.target_rows["d1"]["package_id"] == "pkg-a"


def test_backfill_handles_404_on_package_show() -> None:
    """A package that 404s on `package_show` is logged and skipped."""
    bq = FakeBqClient()
    _seed_target_row(bq, document_id="d1", source_url="https://ckan.test/a.csv")
    _seed_query_result(
        bq,
        rows=[{"document_id": "d1", "source_url": "https://ckan.test/a.csv"}],
    )

    with requests_mock.Mocker() as m:
        _mock_package_list(m, ["pkg-deleted", "pkg-a"])
        m.get(
            f"{_CKAN_BASE}/package_show",
            [
                {
                    "status_code": 404,
                    "json": {"success": False, "error": {"__type": "Not Found Error"}},
                },
                {
                    "json": {
                        "success": True,
                        "result": {
                            "id": "pkg-a",
                            "resources": [{"url": "https://ckan.test/a.csv"}],
                        },
                    },
                },
            ],
        )

        summary = bf.run(
            bq=bq,
            table=_TABLE,
            ckan_base=_CKAN_BASE,
            inter_request_delay=0,
            dry_run=False,
            reset=False,
            limit_packages=None,
            batch_size=500,
        )

    # Walk continued past the 404 and the surviving package matched d1.
    assert summary.packages_walked == 2
    assert summary.docs_updated == 1
    assert bq.target_rows["d1"]["package_id"] == "pkg-a"


def test_backfill_url_collision_logs_warning(capsys: pytest.CaptureFixture[str]) -> None:
    """Same URL in two packages: first wins, second logged.

    structlog is wired to PrintLoggerFactory(stdout), so the audit-
    trail event surfaces via capsys, not caplog.
    """
    with requests_mock.Mocker() as m:
        _mock_package_list(m, ["pkg-first", "pkg-second"])
        m.get(
            f"{_CKAN_BASE}/package_show",
            [
                {
                    "json": {
                        "success": True,
                        "result": {
                            "id": "pkg-first",
                            "resources": [{"url": "https://ckan.test/shared.csv"}],
                        },
                    },
                },
                {
                    "json": {
                        "success": True,
                        "result": {
                            "id": "pkg-second",
                            "resources": [{"url": "https://ckan.test/shared.csv"}],
                        },
                    },
                },
            ],
        )

        url_to_pkg, walked, indexed = bf.build_url_to_package_map(
            _CKAN_BASE,
            delay=0,
        )

    assert walked == 2
    assert indexed == 1  # second occurrence not counted
    assert url_to_pkg["https://ckan.test/shared.csv"] == "pkg-first"
    captured = capsys.readouterr()
    assert "backfill_url_collision" in captured.out


def test_backfill_dry_run_skips_update() -> None:
    """`--dry-run` walks CKAN and counts targets but issues no UPDATE."""
    bq = FakeBqClient()
    _seed_target_row(bq, document_id="d1", source_url="https://ckan.test/a.csv")
    _seed_query_result(
        bq,
        rows=[{"document_id": "d1", "source_url": "https://ckan.test/a.csv"}],
    )

    with requests_mock.Mocker() as m:
        _mock_package_list(m, ["pkg-a"])
        _mock_package_show(
            m,
            package_id="pkg-a",
            resources=[{"url": "https://ckan.test/a.csv"}],
        )

        summary = bf.run(
            bq=bq,
            table=_TABLE,
            ckan_base=_CKAN_BASE,
            inter_request_delay=0,
            dry_run=True,
            reset=False,
            limit_packages=None,
            batch_size=500,
        )

    assert summary.docs_seen == 1
    assert summary.docs_updated == 0
    # Target row unchanged.
    assert bq.target_rows["d1"]["package_id"] is None
    # No param-bound UPDATE issued.
    assert bq.execute_with_params_calls == []


def test_backfill_reset_nulls_then_refetches() -> None:
    """`--reset` clears every package_id before walking + updating."""
    bq = FakeBqClient()
    _seed_target_row(
        bq,
        document_id="d1",
        source_url="https://ckan.test/a.csv",
        package_id="pkg-stale",  # wrong from a prior bad backfill
    )
    # After reset, fetch_targets returns the row again (NULL).
    _seed_query_result(
        bq,
        rows=[{"document_id": "d1", "source_url": "https://ckan.test/a.csv"}],
    )

    with requests_mock.Mocker() as m:
        _mock_package_list(m, ["pkg-correct"])
        _mock_package_show(
            m,
            package_id="pkg-correct",
            resources=[{"url": "https://ckan.test/a.csv"}],
        )

        summary = bf.run(
            bq=bq,
            table=_TABLE,
            ckan_base=_CKAN_BASE,
            inter_request_delay=0,
            dry_run=False,
            reset=True,
            limit_packages=None,
            batch_size=500,
        )

    assert summary.docs_updated == 1
    assert bq.target_rows["d1"]["package_id"] == "pkg-correct"
    # The reset issued an unparameterised UPDATE before the walk.
    reset_calls = [q for q in bq.query_calls if "SET package_id = NULL" in q]
    assert len(reset_calls) == 1


def test_backfill_absolutises_path_only_resource_urls() -> None:
    """Datastore-hosted resources (path-only URLs) match raw.documents.source_url
    after urljoin against the portal origin."""
    bq = FakeBqClient()
    absolute_url = "https://ckan.test/dataset/x/resource/y/download/data.csv"
    _seed_target_row(bq, document_id="d1", source_url=absolute_url)
    _seed_query_result(
        bq,
        rows=[{"document_id": "d1", "source_url": absolute_url}],
    )

    with requests_mock.Mocker() as m:
        _mock_package_list(m, ["pkg-a"])
        _mock_package_show(
            m,
            package_id="pkg-a",
            resources=[{"url": "/dataset/x/resource/y/download/data.csv"}],
        )

        summary = bf.run(
            bq=bq,
            table=_TABLE,
            ckan_base=_CKAN_BASE,
            inter_request_delay=0,
            dry_run=False,
            reset=False,
            limit_packages=None,
            batch_size=500,
        )

    assert summary.docs_updated == 1
    assert bq.target_rows["d1"]["package_id"] == "pkg-a"


def test_backfill_rejects_reset_and_dry_run_together() -> None:
    """`--reset` and `--dry-run` are mutually exclusive — the reset would
    mutate state that the dry-run is supposed to leave alone."""
    bq = FakeBqClient()
    with pytest.raises(ValueError, match="mutually exclusive"):
        bf.run(
            bq=bq,
            table=_TABLE,
            ckan_base=_CKAN_BASE,
            inter_request_delay=0,
            dry_run=True,
            reset=True,
            limit_packages=None,
            batch_size=500,
        )


def test_backfill_rejects_malicious_table_identifier() -> None:
    """Defense in depth: the table id is interpolated into the UPDATE SQL,
    so a hostile identifier must be rejected before binding parameters."""
    bq = FakeBqClient()
    with pytest.raises(ValueError, match="invalid BQ identifier"):
        bf.update_package_ids(
            bq,
            table="proj`; DROP TABLE x; --.raw.documents",
            pairs=[("d1", "pkg-a")],
            dry_run=False,
        )
