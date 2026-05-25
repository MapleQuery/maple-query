from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ingest.config.sources import OrganizationConfig, SourceConfig, SourcesConfig, load_sources

VALID_YAML = """
- country: ca
  source: ckan-opencanada
  api_base: https://open.canada.ca/data/api/3/action
  organizations:
    - code: fin
    - code: statcan
      display_name: Statistics Canada
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
    assert [o.code for o in src.organizations] == ["fin", "statcan"]


def test_defaults_apply() -> None:
    src = SourceConfig.model_validate(
        {
            "country": "ca",
            "source": "ckan-opencanada",
            "api_base": "https://open.canada.ca/data/api/3/action",
            "organizations": [{"code": "fin"}],
        }
    )
    assert src.api_kind == "ckan"
    assert src.page_size == 200


@pytest.mark.parametrize("country", ["CA", "c", "canada", ""])
def test_rejects_bad_country(country: str) -> None:
    with pytest.raises(ValidationError):
        SourceConfig.model_validate(
            {
                "country": country,
                "source": "ckan-opencanada",
                "api_base": "https://x.example/api",
                "organizations": [{"code": "fin"}],
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
                "organizations": [{"code": "fin"}],
            }
        )


@pytest.mark.parametrize("org", ["", "-bad", "UPPER", "a" * 41])
def test_rejects_bad_org_code(org: str) -> None:
    with pytest.raises(ValidationError):
        OrganizationConfig.model_validate({"code": org})


def test_rejects_unknown_api_kind() -> None:
    with pytest.raises(ValidationError):
        SourceConfig.model_validate(
            {
                "country": "ca",
                "source": "some-source",
                "api_base": "https://x.example/api",
                "organizations": [{"code": "fin"}],
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
                "organizations": [{"code": "fin"}],
                "page_size": 1001,
            }
        )


def test_load_sources_rejects_invalid_file(tmp_path: Path) -> None:
    bad = _write(tmp_path, "- country: CA\n  source: x\n  api_base: notaurl\n  organizations: []\n")
    with pytest.raises(ValidationError):
        load_sources(bad)


def test_load_sources_round_trips(tmp_path: Path) -> None:
    cfg = load_sources(_write(tmp_path, VALID_YAML))
    reserialized = [src.model_dump(mode="json") for src in cfg]
    # Re-parsing the serialized form yields equivalent config.
    again = SourcesConfig.model_validate(reserialized)
    assert len(again) == len(cfg)


def test_yaml_roundtrip_smoke() -> None:
    parsed = yaml.safe_load(VALID_YAML)
    assert isinstance(parsed, list)
    assert parsed[0]["country"] == "ca"
