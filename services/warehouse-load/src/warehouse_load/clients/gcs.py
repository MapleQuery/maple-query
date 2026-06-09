"""GCS read client for runlog JSONL files.

Phase B-lite: same code path as local-disk reading, just yields
`(name, line_iterator)` pairs from a `gs://` prefix. Used by the
runlog reader (core/runlog_reader.py); not used until 2.2 (or a
follow-up) uploads runlogs to `gs://maplequery-raw/runlog/`.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from google.cloud import storage  # type: ignore[attr-defined]


@runtime_checkable
class GcsClient(Protocol):
    """The slice of GCS the loader uses."""

    def list_jsonl(self, gcs_prefix: str) -> Iterator[tuple[str, Iterator[str]]]:
        """Yield `(object_name, line_iterator)` for every *.jsonl under prefix."""


class RealGcsClient:
    def __init__(self, client: storage.Client) -> None:
        self._client = client

    @classmethod
    def for_project(cls, project_id: str) -> RealGcsClient:
        return cls(storage.Client(project=project_id))

    def list_jsonl(self, gcs_prefix: str) -> Iterator[tuple[str, Iterator[str]]]:
        bucket_name, prefix = _split_gs_uri(gcs_prefix)
        blobs = self._client.list_blobs(bucket_name, prefix=prefix)
        for blob in blobs:
            if not blob.name.endswith(".jsonl"):
                continue
            text = blob.download_as_text(encoding="utf-8")
            yield blob.name, iter(text.splitlines())


def _split_gs_uri(uri: str) -> tuple[str, str]:
    """`gs://bucket/path/to/` → `("bucket", "path/to/")`."""
    if not uri.startswith("gs://"):
        raise ValueError(f"expected gs:// URI, got {uri!r}")
    without_scheme = uri[len("gs://"):]
    bucket, _, prefix = without_scheme.partition("/")
    if not bucket:
        raise ValueError(f"missing bucket in gs:// URI: {uri!r}")
    return bucket, prefix
