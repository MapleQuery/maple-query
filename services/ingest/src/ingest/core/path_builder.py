"""Build canonical GCS object keys.

Canonical raw path:

    raw/country=<cc>/source=<src>/organization=<org>
        /resource_last_modified=<YYYY-MM-DD>
        /fmt=<ext>__id=<doc_id12>__<safe_filename>

Canonical quarantine path:

    quarantine/country=<cc>/source=<src>
        /resource_last_modified=<YYYY-MM-DD>/reason=<reason>
        /__id=<doc_id12>__<safe_filename>

The partition is keyed on a stable property of the resource: its own
`last_modified` if present, else the dataset's `metadata_created` (the
dataset's creation timestamp, immutable per CKAN). Not wallclock, and
not `metadata_modified` — that one floats on any dataset edit and
would break dedup for resources without their own `last_modified`.
Because the partition is stable, re-ingesting the same unchanged
resource lands on the same key, and the GCS md5-match dedup fires
across days. See `_resource_partition_date` in `core/pipeline.py`.

`doc_id12` is the first 12 hex chars of the full sha256 document_id —
defence-in-depth against accidental collisions; the actual uniqueness
guarantee comes from the write-time existence check in `clients/gcs.py`.
"""
from __future__ import annotations

import re
from datetime import date

from ingest.core.slugify import slugify
from ingest.types import QuarantineReason

ALLOWED_COUNTRIES: frozenset[str] = frozenset({"ca", "uk", "us", "fr"})
ALLOWED_QUARANTINE_REASONS: frozenset[str] = frozenset({
    "download_failed",
    "oversize",
    "truncated_body",
    "unreadable_encoding",
    "path_collision",
})

_SOURCE_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,38}[a-z0-9]$")
_ORG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
_FMT_RE = re.compile(r"^[a-z0-9]{1,10}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_KEY_BYTES = 1024
_DOC_ID12_LEN = 12


class PathValidationError(ValueError):
    """Raised when build_*_path inputs violate the storage-layer contract."""


def build_raw_path(
    *,
    country: str,
    source: str,
    organization: str,
    resource_last_modified: date,
    fmt: str,
    document_id: str,
    resource_url: str,
) -> str:
    """Build a canonical `raw/...` object key.

    Returns a bucket-relative key (no `gs://`, no bucket name).
    """
    _validate_country(country)
    _validate_source(source)
    _validate_organization(organization)
    _validate_fmt(fmt)
    _validate_document_id(document_id)

    doc_id12 = document_id[:_DOC_ID12_LEN]
    safe_filename = slugify(resource_url=resource_url, fmt=fmt)

    key = (
        f"raw/country={country}/source={source}/organization={organization}"
        f"/resource_last_modified={resource_last_modified.isoformat()}"
        f"/fmt={fmt}__id={doc_id12}__{safe_filename}"
    )
    _validate_key_length(key)
    return key


def build_quarantine_path(
    *,
    country: str,
    source: str,
    resource_last_modified: date,
    reason: QuarantineReason,
    document_id: str,
    resource_url: str,
) -> str:
    """Build a `quarantine/...` object key."""
    _validate_country(country)
    _validate_source(source)
    _validate_reason(reason)
    _validate_document_id(document_id)

    doc_id12 = document_id[:_DOC_ID12_LEN]
    # Quarantine has no `fmt=` segment; slug skips extension enforcement.
    safe_filename = slugify(resource_url=resource_url, fmt="unknown")

    key = (
        f"quarantine/country={country}/source={source}"
        f"/resource_last_modified={resource_last_modified.isoformat()}/reason={reason}"
        f"/__id={doc_id12}__{safe_filename}"
    )
    _validate_key_length(key)
    return key


def _validate_country(country: str) -> None:
    if country not in ALLOWED_COUNTRIES:
        raise PathValidationError(
            f"country {country!r} not in allow-list {sorted(ALLOWED_COUNTRIES)}"
        )


def _validate_source(source: str) -> None:
    if not _SOURCE_RE.match(source):
        raise PathValidationError(f"invalid source code: {source!r}")


def _validate_organization(organization: str) -> None:
    if not _ORG_RE.match(organization):
        raise PathValidationError(f"invalid organization code: {organization!r}")


def _validate_fmt(fmt: str) -> None:
    if not _FMT_RE.match(fmt):
        raise PathValidationError(f"invalid fmt: {fmt!r}")


def _validate_document_id(document_id: str) -> None:
    if not _SHA256_RE.match(document_id):
        raise PathValidationError(
            f"document_id must be 64 lower-case hex chars, got {document_id!r}"
        )


def _validate_reason(reason: str) -> None:
    if reason not in ALLOWED_QUARANTINE_REASONS:
        raise PathValidationError(
            f"quarantine reason {reason!r} not in allowed set "
            f"{sorted(ALLOWED_QUARANTINE_REASONS)}"
        )


def _validate_key_length(key: str) -> None:
    n = len(key.encode("utf-8"))
    if n > _MAX_KEY_BYTES:
        raise PathValidationError(f"object key exceeds {_MAX_KEY_BYTES} bytes ({n})")
