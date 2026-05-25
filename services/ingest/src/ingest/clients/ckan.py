"""CKAN client. See PRD 2.2 §6.

Single endpoint used by default: `package_search`. `package_show` is
exposed as an escape hatch but not called by the pipeline — empirical
verification on `open.canada.ca` (2026-05-24) showed `package_search`
returns every field we need.
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ingest.clients.http import HttpClient


class Resource(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    url: str
    name: str | None = None
    format_declared: str | None = Field(default=None, alias="format")
    mimetype_declared: str | None = Field(default=None, alias="mimetype")
    size_declared: int | None = Field(default=None, alias="size")
    languages_declared: list[str] = Field(default_factory=list, alias="language")
    last_modified: datetime | None = None

    @field_validator("languages_declared", mode="before")
    @classmethod
    def _coerce_languages(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            # Some CKAN deployments return a single string instead of a list.
            return [v]
        return list(v)

    @field_validator("last_modified", mode="after")
    @classmethod
    def _ensure_utc_optional(cls, v: datetime | None) -> datetime | None:
        if v is not None and v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class Dataset(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    name: str
    title: str
    organization_code: str
    metadata_modified: datetime
    subjects: list[str] = Field(default_factory=list, alias="subject")
    resources: list[Resource] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _flatten_organization(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "organization_code" in data:
            return data
        org = data.get("organization")
        if isinstance(org, dict) and org.get("name"):
            return {**data, "organization_code": org["name"]}
        if isinstance(org, str):
            return {**data, "organization_code": org}
        return data

    @field_validator("subjects", mode="before")
    @classmethod
    def _coerce_subjects(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)

    @field_validator("metadata_modified", mode="after")
    @classmethod
    def _ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class CkanError(RuntimeError):
    """Raised when the CKAN API returns `success: false`."""


class CkanClient:
    def __init__(
        self,
        *,
        http: HttpClient,
        api_base: str,
        inter_request_delay_seconds: float = 0.5,
    ) -> None:
        self._http = http
        self._api_base = api_base.rstrip("/")
        self._delay = inter_request_delay_seconds

    def search(
        self,
        *,
        subject: str,
        formats: list[str] | None = None,
        organization: str | None = None,
        since: datetime | None = None,
        page_size: int = 200,
    ) -> Iterator[Dataset]:
        """Yield datasets matching the filter, sorted by metadata_modified asc.

        Pagination is internal. See PRD §6.2.
        """
        fq = self._build_fq(
            subject=subject, formats=formats, organization=organization, since=since
        )
        url = f"{self._api_base}/package_search"

        start = 0
        while True:
            payload = self._http.get_json(
                url,
                params={
                    "fq": fq,
                    "sort": "metadata_modified asc",
                    "rows": str(page_size),
                    "start": str(start),
                },
            )
            if not payload.get("success"):
                raise CkanError(f"CKAN returned success=false: {payload.get('error')}")

            result = payload["result"]
            results = result.get("results", [])
            total = result.get("count", 0)

            for raw in results:
                yield Dataset.model_validate(raw)

            if len(results) < page_size:
                return
            start += len(results)
            if start >= total:
                return
            if self._delay > 0:
                time.sleep(self._delay)

    def show(self, dataset_id: str) -> Dataset:
        """Escape-hatch fetch of a single dataset by id. Not used by the pipeline."""
        payload = self._http.get_json(
            f"{self._api_base}/package_show",
            params={"id": dataset_id},
        )
        if not payload.get("success"):
            raise CkanError(f"CKAN returned success=false: {payload.get('error')}")
        return Dataset.model_validate(payload["result"])

    @staticmethod
    def _build_fq(
        *,
        subject: str,
        formats: list[str] | None,
        organization: str | None,
        since: datetime | None,
    ) -> str:
        parts = [f"subject:{subject}"]
        if formats:
            # CKAN convention: `res_format` is upper-case (e.g. "CSV", "XLSX").
            uppercase = [f.upper() for f in formats]
            if len(uppercase) == 1:
                parts.append(f"res_format:{uppercase[0]}")
            else:
                clause = " OR ".join(f"res_format:{f}" for f in uppercase)
                parts.append(f"({clause})")
        if organization:
            parts.append(f"organization:{organization}")
        if since is not None:
            since_utc = since.astimezone(UTC) if since.tzinfo else since.replace(tzinfo=UTC)
            since_iso = since_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
            parts.append(f"metadata_modified:[{since_iso} TO *]")
        return " AND ".join(parts)
