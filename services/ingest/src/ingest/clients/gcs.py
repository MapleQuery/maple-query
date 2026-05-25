"""GCS client wrapping google-cloud-storage.

Write-time collision protocol — never silently overwrite an existing object:
1. HEAD the target object.
2. If absent → upload with `if_generation_match=0` (precondition: must not exist).
   On 412 Precondition Failed (a TOCTOU race), restart from step 1.
3. If present and md5 matches → idempotent skip (re-run safe).
4. If present and md5 differs → signal PathCollision; the caller routes
   the body to a `quarantine/.../reason=path_collision/` prefix instead.
"""
from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass

from google.api_core import exceptions as gcp_exceptions
from google.cloud import storage


@dataclass(frozen=True)
class Uploaded:
    gcs_uri: str
    generation: int


@dataclass(frozen=True)
class IdempotentSkip:
    gcs_uri: str
    existing_md5_b64: str


@dataclass(frozen=True)
class PathCollision:
    gcs_uri: str
    existing_md5_b64: str
    attempted_md5_b64: str


UploadResult = Uploaded | IdempotentSkip | PathCollision


class GcsClient:
    def __init__(self, *, client: storage.Client, bucket: str, max_collision_retries: int = 3) -> None:
        self._client = client
        self._bucket = client.bucket(bucket)
        self._bucket_name = bucket
        self._max_collision_retries = max_collision_retries

    def upload(
        self,
        *,
        object_name: str,
        body: bytes,
        content_type: str | None = None,
    ) -> UploadResult:
        attempted_md5 = _md5_b64(body)
        gcs_uri = f"gs://{self._bucket_name}/{object_name}"

        for attempt in range(self._max_collision_retries + 1):
            blob = self._bucket.blob(object_name)
            existing = _reload_or_none(blob)

            if existing is not None:
                existing_md5 = blob.md5_hash or ""
                if existing_md5 == attempted_md5:
                    return IdempotentSkip(gcs_uri=gcs_uri, existing_md5_b64=existing_md5)
                return PathCollision(
                    gcs_uri=gcs_uri,
                    existing_md5_b64=existing_md5,
                    attempted_md5_b64=attempted_md5,
                )

            try:
                blob.upload_from_string(
                    body,
                    content_type=content_type or "application/octet-stream",
                    if_generation_match=0,
                )
                return Uploaded(gcs_uri=gcs_uri, generation=blob.generation or 0)
            except gcp_exceptions.PreconditionFailed:
                # TOCTOU: someone wrote the object between our HEAD and PUT.
                # Loop restarts from step 1 (existence check).
                if attempt == self._max_collision_retries:
                    raise
                continue

        raise RuntimeError("unreachable: upload loop exhausted without returning")

    def exists(self, object_name: str) -> bool:
        blob = self._bucket.blob(object_name)
        return _reload_or_none(blob) is not None


def _reload_or_none(blob: storage.Blob) -> storage.Blob | None:
    try:
        blob.reload()
        return blob
    except gcp_exceptions.NotFound:
        return None


def _md5_b64(body: bytes) -> str:
    """GCS exposes md5Hash as a base64-encoded digest, not hex."""
    return base64.b64encode(hashlib.md5(body).digest()).decode("ascii")
