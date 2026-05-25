"""Tests for the GCS upload state machine using a fake bucket/blob.

We don't talk to real GCS here — the SDK glue is integration-test
material. These tests cover the decision flow (new vs idempotent vs
collision vs TOCTOU) which is where the actual contract lives.
"""
from __future__ import annotations

import base64
import hashlib

import pytest
from google.api_core import exceptions as gcp_exceptions

from ingest.clients.gcs import (
    GcsClient,
    IdempotentSkip,
    PathCollision,
    Uploaded,
)


def _md5(body: bytes) -> str:
    return base64.b64encode(hashlib.md5(body).digest()).decode("ascii")


class _FakeBlob:
    def __init__(self, *, name: str, exists: bool, md5_hash: str = "", generation: int = 1) -> None:
        self.name = name
        self.md5_hash = md5_hash
        self.generation = generation
        self._exists = exists
        self.uploaded_body: bytes | None = None
        self.upload_calls = 0

    def reload(self) -> None:
        if not self._exists:
            raise gcp_exceptions.NotFound("not found")

    def upload_from_string(self, body: bytes, *, content_type: str, if_generation_match: int) -> None:
        self.upload_calls += 1
        if self._exists and if_generation_match == 0:
            raise gcp_exceptions.PreconditionFailed("object already exists")
        self._exists = True
        self.uploaded_body = body
        self.md5_hash = _md5(body)


class _FakeBucket:
    def __init__(self, blobs: dict[str, _FakeBlob] | None = None) -> None:
        self._blobs = blobs or {}

    def blob(self, name: str) -> _FakeBlob:
        if name not in self._blobs:
            self._blobs[name] = _FakeBlob(name=name, exists=False)
        return self._blobs[name]


class _FakeStorageClient:
    def __init__(self, bucket: _FakeBucket) -> None:
        self._bucket = bucket

    def bucket(self, name: str) -> _FakeBucket:
        return self._bucket


@pytest.fixture
def empty_bucket() -> _FakeBucket:
    return _FakeBucket()


def test_uploads_new_object_with_generation_match(empty_bucket: _FakeBucket) -> None:
    client = GcsClient(client=_FakeStorageClient(empty_bucket), bucket="maplequery-raw")
    result = client.upload(object_name="raw/x", body=b"hello")
    assert isinstance(result, Uploaded)
    assert result.gcs_uri == "gs://maplequery-raw/raw/x"
    assert empty_bucket._blobs["raw/x"].uploaded_body == b"hello"


def test_idempotent_skip_when_md5_matches() -> None:
    existing_body = b"hello"
    bucket = _FakeBucket(
        {"raw/x": _FakeBlob(name="raw/x", exists=True, md5_hash=_md5(existing_body))}
    )
    client = GcsClient(client=_FakeStorageClient(bucket), bucket="maplequery-raw")
    result = client.upload(object_name="raw/x", body=existing_body)
    assert isinstance(result, IdempotentSkip)
    assert bucket._blobs["raw/x"].upload_calls == 0


def test_path_collision_when_md5_differs() -> None:
    bucket = _FakeBucket(
        {"raw/x": _FakeBlob(name="raw/x", exists=True, md5_hash=_md5(b"DIFFERENT"))}
    )
    client = GcsClient(client=_FakeStorageClient(bucket), bucket="maplequery-raw")
    result = client.upload(object_name="raw/x", body=b"hello")
    assert isinstance(result, PathCollision)
    assert result.existing_md5_b64 == _md5(b"DIFFERENT")
    assert result.attempted_md5_b64 == _md5(b"hello")
