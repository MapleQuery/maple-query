from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ingest.config.sources import SourceConfig, SourcesConfig, load_sources

VALID_YAML = """
- country: ca
  source: ckan-opencanada
  api_base: https://open.canada.ca/data/api/3/action
"""


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "sources.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def test_loads_valid_yaml(tmp_path: Path) -> None:
    cfg = load_sources(_write(tmp_path, VALID_YAML))
    assert len(cfg) == 1
    src = next(iter(cfg))
    assert src.country == "ca"
    assert src.source == "ckan-opencanada"
    assert src.api_kind == "ckan"
    assert src.page_size == 200


def test_defaults_apply() -> None:
    src = SourceConfig.model_validate(
        {
            "country": "ca",
            "source": "ckan-opencanada",
            "api_base": "https://open.canada.ca/data/api/3/action",
        }
    )
    assert src.api_kind == "ckan"
    assert src.page_size == 200


def test_legacy_organizations_block_is_ignored(tmp_path: Path) -> None:
    """Old YAMLs with an `organizations` block must still load — the field
    is no longer consumed (discovery + --limit-orgs handle org selection
    now) but we don't want stale configs to fail loudly."""
    legacy = """
- country: ca
  source: ckan-opencanada
  api_base: https://open.canada.ca/data/api/3/action
  organizations:
    - code: fin
    - code: statcan
"""
    cfg = load_sources(_write(tmp_path, legacy))
    assert len(cfg) == 1


@pytest.mark.parametrize("country", ["CA", "c", "canada", ""])
def test_rejects_bad_country(country: str) -> None:
    with pytest.raises(ValidationError):
        SourceConfig.model_validate(
            {
                "country": country,
                "source": "ckan-opencanada",
                "api_base": "https://x.example/api",
            }
        )


@pytest.mark.parametrize("source", ["ab", "a" * 41, "UPPER", "-leading", "trailing-"])
def test_rejects_bad_source_code(source: str) -> None:
    with pytest.raises(ValidationError):
        SourceConfig.model_validate(
            {
                "country": "ca",
                "source": source,
                "api_base": "https://x.example/api",
            }
        )


def test_rejects_unknown_api_kind() -> None:
    with pytest.raises(ValidationError):
        SourceConfig.model_validate(
            {
                "country": "ca",
                "source": "some-source",
                "api_base": "https://x.example/api",
                "api_kind": "socrata",
            }
        )


def test_rejects_page_size_out_of_range() -> None:
    with pytest.raises(ValidationError):
        SourceConfig.model_validate(
            {
                "country": "ca",
                "source": "ckan-opencanada",
                "api_base": "https://x.example/api",
                "page_size": 1001,
            }
        )


def test_load_sources_rejects_invalid_file(tmp_path: Path) -> None:
    bad = _write(tmp_path, "- country: CA\n  source: x\n  api_base: notaurl\n")
    with pytest.raises(ValidationError):
        load_sources(bad)


def test_load_sources_round_trips(tmp_path: Path) -> None:
    cfg = load_sources(_write(tmp_path, VALID_YAML))
    reserialized = [src.model_dump(mode="json") for src in cfg]
    again = SourcesConfig.model_validate(reserialized)
    assert len(again) == len(cfg)


def test_yaml_roundtrip_smoke() -> None:
    parsed = yaml.safe_load(VALID_YAML)
    assert isinstance(parsed, list)
    assert parsed[0]["country"] == "ca"
