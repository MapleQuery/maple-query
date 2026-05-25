"""Source / organization config loaded from YAML.

The YAML at `infra/ingest_sources.yaml` is the canonical instance; this
module validates it eagerly on startup and exits non-zero on schema
failure.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, HttpUrl, RootModel


class OrganizationConfig(BaseModel):
    code: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,39}$")
    display_name: str | None = None


class SourceConfig(BaseModel):
    country: str = Field(pattern=r"^[a-z]{2}$")
    # 3-40 chars, [a-z0-9-], no leading/trailing dash. Matches the same
    # rule path_builder enforces, so an invalid value fails at config-load
    # rather than later when we try to build an object key with it.
    source: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,38}[a-z0-9]$")
    api_base: HttpUrl
    organizations: list[OrganizationConfig]
    api_kind: Literal["ckan"] = "ckan"
    page_size: int = Field(default=200, ge=1, le=1000)


class SourcesConfig(RootModel[list[SourceConfig]]):
    def __iter__(self) -> Iterator[SourceConfig]:  # type: ignore[override]
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)


def load_sources(path: Path) -> SourcesConfig:
    """Load and validate `infra/ingest_sources.yaml`.

    Raises `pydantic.ValidationError` on schema failure; the CLI catches
    and exits non-zero.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return SourcesConfig.model_validate(raw)
