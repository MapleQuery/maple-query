"""Source / organization config loaded from YAML.

Per PRD 2.2 §5.2 — the YAML at `infra/ingest_sources.yaml` is the
canonical instance; this module validates it eagerly.
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
    # Matches PRD 2.1 §3.1 SOURCE_CODE grammar (no leading/trailing dash).
    # Stricter than the original PRD 2.2 §5.2 sketch — aligned with path_builder
    # so an invalid value fails at config-load, not later at path-build time.
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
    and exits non-zero per PRD §5.2.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return SourcesConfig.model_validate(raw)
