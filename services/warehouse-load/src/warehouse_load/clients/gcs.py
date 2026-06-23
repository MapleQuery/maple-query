"""GCS read client.

`list_jsonl` yields `(object_name, line_iterator)` pairs from a
`gs://` prefix — mirrors the local-disk path for runlog reads.

`list_existing` walks a prefix and returns the full `gs://bucket/object`
URI for every blob found. The documents loader uses this as the
source of truth for blob existence so a bucket clean is self-healing
on the next load.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from google.cloud import storage  # type: ignore[attr-defined]


@runtime_checkable
class GcsClient(Protocol):
    def list_jsonl(self, gcs_prefix: str) -> Iterator[tuple[str, Iterator[str]]]:
        """Yield `(gs://bucket/object, line_iterator)` for every *.jsonl under prefix."""

    def list_existing(self, gcs_prefix: str) -> set[str]:
        """Return the set of full `gs://bucket/object` URIs under prefix."""


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
            yield f"gs://{bucket_name}/{blob.name}", iter(text.splitlines())

    def list_existing(self, gcs_prefix: str) -> set[str]:
        """One pass over the prefix; materialize the full URI set.

        Pagination is handled transparently by `list_blobs`. Network
        errors and auth/bucket-missing conditions propagate to the
        caller, which decides whether to fail the run.
        """
        bucket_name, prefix = _split_gs_uri(gcs_prefix)
        return {
            f"gs://{bucket_name}/{blob.name}"
            for blob in self._client.list_blobs(bucket_name, prefix=prefix)
        }


def _split_gs_uri(uri: str) -> tuple[str, str]:
    """`gs://bucket/path/to/` → `("bucket", "path/to/")`."""
    if not uri.startswith("gs://"):
        raise ValueError(f"expected gs:// URI, got {uri!r}")
    without_scheme = uri[len("gs://"):]
    bucket, _, prefix = without_scheme.partition("/")
    if not bucket:
        raise ValueError(f"missing bucket in gs:// URI: {uri!r}")
    return bucket, prefix
