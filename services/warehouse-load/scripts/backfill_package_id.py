"""Periodic re-attachment of raw.documents.package_id.

Walks every CKAN package via package_search pagination, builds a
{resource_url: package_id} map, joins it against (SELECT document_id,
source_url FROM raw.documents WHERE package_id IS NULL), and writes
the matching package_id back via batched parameter-bound UPDATEs.

`raw.documents.package_id` is a permanently NULLABLE best-effort link
to the CKAN parent: a row's NULL is healed automatically by re-running
this script whenever a previously-deleted package reappears in CKAN.
Run on demand after suspected CKAN changes; safe to re-run any time
(idempotent — only un-linked rows are touched).

Operator-grade, lives under scripts/ (not src/) so it stays out of the
import-linter graph and the mypy --strict surface of the
warehouse_load package proper.

Run shape:

    uv run python services/warehouse-load/scripts/backfill_package_id.py \\
        [--dry-run] [--reset] [--limit-packages N] [--ckan-base URL] \\
        [--inter-request-delay 0.5] [--batch-size 500]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from google.cloud import bigquery
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

# Make the warehouse_load package importable when running this script
# directly via `uv run python services/warehouse-load/scripts/...`. The
# script lives outside `src/` on purpose (see module docstring), so the
# package import path needs to be wired by hand.
_HERE = Path(__file__).resolve()
_SRC = _HERE.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from warehouse_load.clients.bq import BqClient, RealBqClient  # noqa: E402
from warehouse_load.config.settings import Settings  # noqa: E402
from warehouse_load.providers.logging import configure_logging, get_logger  # noqa: E402

log = get_logger(__name__)

_DEFAULT_CKAN_BASE = "https://open.canada.ca/data/api/3/action"
_DEFAULT_INTER_REQUEST_DELAY = 0.5
_DEFAULT_BATCH_SIZE = 500

# CKAN is the slower, less-reliable side (open.canada.ca returns 502s
# under load), so we tolerate more retries here than the BQ side. The
# warehouse loader's bq_retry_policy() keeps its 3-attempt budget; this
# is intentionally a separate constant.
_CKAN_RETRY = Retrying(
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=2.0, max=30.0, jitter=0.25),
    retry=retry_if_exception_type(
        (requests.ConnectionError, requests.Timeout, requests.HTTPError),
    ),
    before_sleep=before_sleep_log(
        logging.getLogger("warehouse_load.backfill.ckan_retry"),
        logging.WARNING,
    ),
    reraise=True,
)


@dataclass(frozen=True)
class BackfillSummary:
    """End-of-run roll-up. Printed as JSON; attached to the promotion PR."""

    packages_walked: int
    resources_indexed: int
    docs_seen: int
    docs_updated: int
    docs_missed: int
    duration_ms: int


_PACKAGE_SEARCH_PAGE_SIZE = 1000


def build_url_to_package_map(
    ckan_base: str,
    *,
    delay: float,
    limit_packages: int | None = None,
) -> tuple[dict[str, str], int, int]:
    """Walk every package via `package_search` pagination; return
    ({resource_url: package_id}, packages, resources).

    Uses `package_search?q=*:*&rows=1000` to fetch ~1000 datasets per
    HTTP call, each carrying its full `resources` array. For a 47K-
    package portal this is ~48 calls vs. ~47K per-package calls —
    minutes instead of hours.

    Path-only resource URLs (CKAN datastore-hosted files) are
    absolutised against `portal_origin` so they line up byte-for-byte
    with `raw.documents.source_url` — the ingest CKAN client does the
    same normalisation in `_absolutize_resource_urls`. Without this,
    every datastore-hosted resource would be a miss.

    First-writer-wins on URL collisions: a URL that appears in two
    packages keeps the first owner and logs `backfill_url_collision`
    for the second. CKAN-side URL sharing across packages does happen
    in practice, and without a stronger signal than "same URL in two
    packages" automated arbitration would just pick a side at random.
    """
    portal_origin = _portal_origin(ckan_base)
    url = f"{ckan_base.rstrip('/')}/package_search"

    url_to_pkg: dict[str, str] = {}
    packages_walked = 0
    resources_indexed = 0
    start = 0
    total: int | None = None

    while True:
        if limit_packages is not None and packages_walked >= limit_packages:
            break

        params = {
            "q": "*:*",
            "rows": str(_PACKAGE_SEARCH_PAGE_SIZE),
            "start": str(start),
        }

        payload: Any = None
        for attempt in _CKAN_RETRY:
            with attempt:
                resp = requests.get(url, params=params, timeout=120)
                resp.raise_for_status()
                payload = resp.json()
        if not payload.get("success"):
            raise RuntimeError(
                f"CKAN package_search returned success=false: {payload.get('error')}",
            )

        result = payload["result"]
        results = result.get("results", []) or []
        if total is None:
            total = int(result.get("count", 0))

        if not results:
            break

        for pkg in results:
            pkg_id = pkg.get("id")
            if not isinstance(pkg_id, str) or not pkg_id:
                # Defensive — should never happen on open.canada.ca.
                continue
            for resource in pkg.get("resources", []) or []:
                raw_url = resource.get("url")
                if not isinstance(raw_url, str) or not raw_url:
                    continue
                resolved = (
                    urljoin(portal_origin, raw_url) if raw_url.startswith("/") else raw_url
                )
                existing = url_to_pkg.get(resolved)
                if existing is None:
                    url_to_pkg[resolved] = pkg_id
                    resources_indexed += 1
                elif existing != pkg_id:
                    log.warning(
                        "backfill_url_collision",
                        url=resolved,
                        kept_package_id=existing,
                        dropped_package_id=pkg_id,
                    )
            packages_walked += 1
            if limit_packages is not None and packages_walked >= limit_packages:
                break

        log.info(
            "backfill_walk_progress",
            packages_walked=packages_walked,
            resources_indexed=resources_indexed,
            total_packages=total,
        )

        start += len(results)
        if total is not None and start >= total:
            break
        if delay > 0:
            time.sleep(delay)

    log.info(
        "backfill_walk_finish",
        packages_walked=packages_walked,
        resources_indexed=resources_indexed,
        total_packages=total,
    )
    return url_to_pkg, packages_walked, resources_indexed


def fetch_targets(bq: BqClient, table: str) -> list[tuple[str, str]]:
    """Return (document_id, source_url) pairs for rows with package_id NULL."""
    sql = (
        f"SELECT document_id, source_url FROM `{table}` "
        f"WHERE package_id IS NULL"
    )
    return [(r["document_id"], r["source_url"]) for r in bq.query_rows(sql)]


def update_package_ids(
    bq: BqClient,
    *,
    table: str,
    pairs: list[tuple[str, str]],
    dry_run: bool,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> int:
    """Write package_id back. Returns the number of rows it claimed to update.

    Batched parameter-bound UPDATE pattern: a single statement joins
    against an inline `UNNEST([STRUCT(...), ...])` literal so a 15K-row
    backfill becomes ~30 BQ statements rather than 15K. The
    `t.package_id IS NULL` guard is belt-and-braces against a
    hypothetical concurrent writer (none today).

    `dry_run=True` short-circuits without touching BQ.
    """
    if not pairs:
        return 0
    if dry_run:
        log.info("backfill_dry_run_skip_update", would_update=len(pairs))
        return 0

    _validate_table_id(table)

    total = 0
    for batch_start in range(0, len(pairs), batch_size):
        batch = pairs[batch_start : batch_start + batch_size]
        struct_literals: list[str] = []
        params: list[bigquery.ScalarQueryParameter] = []
        for idx, (doc_id, pkg_id) in enumerate(batch):
            struct_literals.append(
                f"STRUCT(@d_{idx} AS document_id, @p_{idx} AS package_id)",
            )
            params.append(bigquery.ScalarQueryParameter(f"d_{idx}", "STRING", doc_id))
            params.append(bigquery.ScalarQueryParameter(f"p_{idx}", "STRING", pkg_id))

        sql = (
            f"UPDATE `{table}` t\n"
            "SET package_id = src.package_id\n"
            "FROM UNNEST([\n  "
            + ",\n  ".join(struct_literals)
            + "\n]) AS src\n"
            "WHERE t.document_id = src.document_id\n"
            "  AND t.package_id IS NULL"
        )
        bq.execute_with_params(sql, params=params)
        total += len(batch)
        log.info(
            "backfill_update_batch_finish",
            batch_index=batch_start // batch_size,
            batch_size=len(batch),
            cumulative=total,
        )
    return total


def reset_package_ids(bq: BqClient, *, table: str) -> None:
    """NULL every package_id. Use only before re-running on a bad backfill."""
    _validate_table_id(table)
    sql = f"UPDATE `{table}` SET package_id = NULL WHERE TRUE"
    log.warning("backfill_reset_start", table=table)
    bq.execute(sql)
    log.warning("backfill_reset_finish", table=table)


def run(
    *,
    bq: BqClient,
    table: str,
    ckan_base: str,
    inter_request_delay: float,
    dry_run: bool,
    reset: bool,
    limit_packages: int | None,
    batch_size: int,
) -> BackfillSummary:
    """Run one backfill pass. Pure orchestration; no argv / no exit codes."""
    start = time.monotonic()

    if reset:
        if dry_run:
            raise ValueError("--reset and --dry-run are mutually exclusive")
        reset_package_ids(bq, table=table)

    url_to_pkg, packages_walked, resources_indexed = build_url_to_package_map(
        ckan_base,
        delay=inter_request_delay,
        limit_packages=limit_packages,
    )

    targets = fetch_targets(bq, table)
    pairs: list[tuple[str, str]] = []
    misses = 0
    for doc_id, source_url in targets:
        pkg_id = url_to_pkg.get(source_url)
        if pkg_id is None:
            misses += 1
            log.info("backfill_miss", document_id=doc_id, source_url=source_url)
            continue
        pairs.append((doc_id, pkg_id))

    docs_updated = update_package_ids(
        bq,
        table=table,
        pairs=pairs,
        dry_run=dry_run,
        batch_size=batch_size,
    )

    return BackfillSummary(
        packages_walked=packages_walked,
        resources_indexed=resources_indexed,
        docs_seen=len(targets),
        docs_updated=docs_updated,
        docs_missed=misses,
        duration_ms=int((time.monotonic() - start) * 1000),
    )


def _portal_origin(ckan_base: str) -> str:
    parsed = urlparse(ckan_base)
    return f"{parsed.scheme}://{parsed.netloc}"


def _validate_table_id(table_id: str) -> None:
    """Reject any segment with characters outside `[A-Za-z0-9_-]`.

    Same posture as warehouse_load.core.documents_merge._validate_table_id:
    backticks don't escape in BQ, and `table` is interpolated into the
    UPDATE SQL.
    """
    parts = table_id.split(".")
    if len(parts) != 3:
        raise ValueError(f"expected project.dataset.table, got {table_id!r}")
    for part in parts:
        if not part or any(c for c in part if not (c.isalnum() or c in "_-")):
            raise ValueError(f"invalid BQ identifier segment {part!r} in {table_id!r}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill raw.documents.package_id from a CKAN walk.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Walk CKAN and build the map, but do not UPDATE raw.documents.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="NULL every existing package_id before refetching. Use after a bad backfill.",
    )
    parser.add_argument(
        "--limit-packages",
        type=int,
        default=None,
        help="Stop the CKAN walk after N packages. Smoke-test knob.",
    )
    parser.add_argument(
        "--ckan-base",
        default=_DEFAULT_CKAN_BASE,
        help=f"CKAN API base URL. Default: {_DEFAULT_CKAN_BASE}",
    )
    parser.add_argument(
        "--inter-request-delay",
        type=float,
        default=_DEFAULT_INTER_REQUEST_DELAY,
        help=f"Seconds between CKAN calls. Default: {_DEFAULT_INTER_REQUEST_DELAY}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_DEFAULT_BATCH_SIZE,
        help=f"Document_id rows per UPDATE statement. Default: {_DEFAULT_BATCH_SIZE}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging()

    settings = Settings()  # type: ignore[call-arg]
    project_id = settings.gcp_project_id
    table = f"{project_id}.{settings.bq_dataset_raw}.{settings.bq_documents_table}"

    log.info(
        "backfill_start",
        table=table,
        ckan_base=args.ckan_base,
        dry_run=args.dry_run,
        reset=args.reset,
        limit_packages=args.limit_packages,
        inter_request_delay=args.inter_request_delay,
        batch_size=args.batch_size,
    )

    bq = RealBqClient.for_project(project_id)
    summary = run(
        bq=bq,
        table=table,
        ckan_base=args.ckan_base,
        inter_request_delay=args.inter_request_delay,
        dry_run=args.dry_run,
        reset=args.reset,
        limit_packages=args.limit_packages,
        batch_size=args.batch_size,
    )

    print(json.dumps(asdict(summary), indent=2))
    log.info("backfill_finish", **asdict(summary))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
