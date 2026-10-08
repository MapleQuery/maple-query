"""GCS surface for curate: find the newest landed archive and open it
for streaming (a seekable reader, which zipfile needs)."""

from __future__ import annotations

from typing import IO, Protocol, runtime_checkable

from google.cloud import storage  # type: ignore[attr-defined]


@runtime_checkable
class GcsClient(Protocol):
    def newest_object(self, *, prefix: str, contains: str) -> str | None:
        """Name of the newest object under `prefix` whose name contains
        `contains` (by the resource_last_modified partition, then update
        time), or None."""

    def open_binary(self, name: str) -> IO[bytes]:
        """Seekable binary reader over an object."""


class RealGcsClient:
    def __init__(self, *, project_id: str, bucket: str) -> None:
        self._bucket = storage.Client(project=project_id).bucket(bucket)
        self.bucket_name = bucket

    def newest_object(self, *, prefix: str, contains: str) -> str | None:
        blobs = [b for b in self._bucket.list_blobs(prefix=prefix) if contains in b.name]
        if not blobs:
            return None
        newest = max(blobs, key=lambda b: (b.name.split("resource_last_modified=")[-1][:10], b.updated))
        return str(newest.name)

    def open_binary(self, name: str) -> IO[bytes]:
        reader: IO[bytes] = self._bucket.blob(name).open("rb")
        return reader
