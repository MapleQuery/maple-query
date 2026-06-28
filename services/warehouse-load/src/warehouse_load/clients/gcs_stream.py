"""Streaming GCS download for CSV bodies.

Yields the blob into a caller-supplied file handle (typically a
`tempfile.NamedTemporaryFile`) so polars can read from the path
without buffering the whole file in memory. A byte counter aborts
the download once `max_bytes` is exceeded — bounded blast radius
for a pathological file.

`BlobMissingError` is raised on a 404; the per-doc orchestrator maps
that to `load_status='blob_missing'`. All other GCS errors propagate.

Retry policy: same tenacity config as `RealBqClient` (3 attempts,
exp backoff + jitter, retries on `ServiceUnavailable` /
`InternalServerError` / `TooManyRequests` / `GatewayTimeout`). PRD
§7.3 — extend `_RETRYABLE` in `providers/retry.py` if a
GCS-specific exception class needs adding.
"""
from __future__ import annotations

from typing import BinaryIO, Protocol, runtime_checkable

from google.api_core import exceptions as gax
from google.cloud import storage  # type: ignore[attr-defined]
from tenacity import Retrying

from warehouse_load.providers.retry import bq_retry_policy


class BlobMissingError(Exception):
    """Raised by `download_blob_to_file` on a 404. Distinct from generic
    errors so the orchestrator can map it to `load_status='blob_missing'`
    without inspecting the exception type."""


class BytesCapExceededError(Exception):
    """Raised when the stream exceeds `max_bytes`. The orchestrator
    maps this to `load_status='parse_failed'` with
    `load_error='exceeded max_bytes_per_doc=<n>'`.
    """


@runtime_checkable
class GcsStreamClient(Protocol):
    def download_blob_to_file(
        self,
        *,
        gcs_uri: str,
        sink: BinaryIO,
        max_bytes: int,
    ) -> int:
        """Stream `gcs_uri` into `sink`. Returns bytes written.

        Raises `BlobMissingError` on 404; `BytesCapExceededError` if
        the stream exceeds `max_bytes`. Other GCS errors propagate.
        """


class RealGcsStreamClient:
    def __init__(self, client: storage.Client, *, retry: Retrying | None = None) -> None:
        self._client = client
        self._retry = retry if retry is not None else bq_retry_policy()

    @classmethod
    def for_project(cls, project_id: str) -> RealGcsStreamClient:
        return cls(storage.Client(project=project_id))

    def download_blob_to_file(
        self,
        *,
        gcs_uri: str,
        sink: BinaryIO,
        max_bytes: int,
    ) -> int:
        bucket_name, name = _split_gs_uri(gcs_uri)
        # `restart=False` is critical: tenacity retries an aborted
        # download from byte 0, but the sink may have partial bytes
        # from the prior attempt. Truncate-on-retry rather than
        # appending lets the retry succeed cleanly.
        for attempt in self._retry:
            with attempt:
                sink.seek(0)
                sink.truncate(0)
                try:
                    blob = self._client.bucket(bucket_name).blob(name)
                    counted = _CountingWriter(sink, max_bytes=max_bytes)
                    blob.download_to_file(counted)
                    return counted.bytes_written
                except gax.NotFound as exc:
                    raise BlobMissingError(f"gcs 404: {gcs_uri}") from exc
        # Unreachable: tenacity raises on exhaustion (reraise=True).
        raise RuntimeError("retry loop exited without returning or raising")


class _CountingWriter:
    """Wrap a file-like object, counting bytes written and aborting
    when the cap is exceeded. The blob iterator stops calling `write`
    as soon as we raise.
    """

    def __init__(self, sink: BinaryIO, *, max_bytes: int) -> None:
        self._sink = sink
        self._max = max_bytes
        self.bytes_written = 0

    def write(self, data: bytes) -> int:
        self.bytes_written += len(data)
        if self.bytes_written > self._max:
            raise BytesCapExceededError(
                f"exceeded max_bytes_per_doc={self._max} "
                f"(read {self.bytes_written})",
            )
        return self._sink.write(data)

    def flush(self) -> None:
        self._sink.flush()


def _split_gs_uri(uri: str) -> tuple[str, str]:
    """`gs://bucket/path/to/file.csv` → `("bucket", "path/to/file.csv")`."""
    if not uri.startswith("gs://"):
        raise ValueError(f"expected gs:// URI, got {uri!r}")
    without_scheme = uri[len("gs://"):]
    bucket, _, name = without_scheme.partition("/")
    if not bucket or not name:
        raise ValueError(f"gs:// URI must include bucket and object name: {uri!r}")
    return bucket, name
