from __future__ import annotations

import re
from datetime import date

import pytest

from ingest.core.path_builder import (
    PathValidationError,
    build_quarantine_path,
    build_raw_path,
)

VALID_DOC_ID = "a" * 64
RESOURCE_LAST_MODIFIED = date(2026, 5, 19)

_RAW_GRAMMAR = re.compile(
    r"^raw/country=[a-z]{2}"
    r"/source=[a-z0-9][a-z0-9-]{1,38}[a-z0-9]"
    r"/organization=[a-z0-9][a-z0-9-]{0,39}"
    r"/resource_last_modified=\d{4}-\d{2}-\d{2}"
    r"/fmt=[a-z0-9]{1,10}__id=[0-9a-f]{12}__[a-z0-9][a-z0-9._-]*$"
)

_QUARANTINE_GRAMMAR = re.compile(
    r"^quarantine/country=[a-z]{2}"
    r"/source=[a-z0-9][a-z0-9-]{1,38}[a-z0-9]"
    r"/resource_last_modified=\d{4}-\d{2}-\d{2}"
    r"/reason=(download_failed|oversize|truncated_body|unreadable_encoding|path_collision)"
    r"/__id=[0-9a-f]{12}__[a-z0-9][a-z0-9._-]*$"
)


def _build_raw(**overrides: object) -> str:
    kwargs: dict[str, object] = {
        "country": "ca",
        "source": "ckan-opencanada",
        "organization": "fin",
        "resource_last_modified": RESOURCE_LAST_MODIFIED,
        "fmt": "csv",
        "document_id": VALID_DOC_ID,
        "resource_url": "https://open.canada.ca/data/dataset/x/report.csv",
    }
    kwargs.update(overrides)
    return build_raw_path(**kwargs)  # type: ignore[arg-type]


def _build_quarantine(**overrides: object) -> str:
    kwargs: dict[str, object] = {
        "country": "ca",
        "source": "ckan-opencanada",
        "resource_last_modified": RESOURCE_LAST_MODIFIED,
        "reason": "oversize",
        "document_id": VALID_DOC_ID,
        "resource_url": "https://example.gov/foo.pdf",
    }
    kwargs.update(overrides)
    return build_quarantine_path(**kwargs)  # type: ignore[arg-type]


def test_build_raw_path_matches_canonical_template() -> None:
    key = _build_raw()
    assert key == (
        "raw/country=ca/source=ckan-opencanada/organization=fin"
        "/resource_last_modified=2026-05-19/fmt=csv__id=aaaaaaaaaaaa__report.csv"
    )


def test_build_raw_path_matches_grammar() -> None:
    assert _RAW_GRAMMAR.match(_build_raw())


def test_build_quarantine_path_matches_canonical_template() -> None:
    key = _build_quarantine()
    assert key == (
        "quarantine/country=ca/source=ckan-opencanada"
        "/resource_last_modified=2026-05-19/reason=oversize"
        "/__id=aaaaaaaaaaaa__foo.pdf"
    )


def test_build_quarantine_path_matches_grammar() -> None:
    assert _QUARANTINE_GRAMMAR.match(_build_quarantine())


@pytest.mark.parametrize("country", ["de", "CA", "c", "canada", ""])
def test_rejects_country_outside_allow_list(country: str) -> None:
    with pytest.raises(PathValidationError):
        _build_raw(country=country)


@pytest.mark.parametrize(
    "source",
    [
        "ab",                              # too short
        "a" * 41,                          # too long
        "-leading-dash",
        "trailing-dash-",
        "UPPER",
        "has_underscore",
    ],
)
def test_rejects_invalid_source(source: str) -> None:
    with pytest.raises(PathValidationError):
        _build_raw(source=source)


@pytest.mark.parametrize("org", ["", "a" * 41, "-bad", "UPPER", "with_underscore"])
def test_rejects_invalid_organization(org: str) -> None:
    with pytest.raises(PathValidationError):
        _build_raw(organization=org)


@pytest.mark.parametrize("fmt", ["", "a" * 11, "CSV", "ab.cd"])
def test_rejects_invalid_fmt(fmt: str) -> None:
    with pytest.raises(PathValidationError):
        _build_raw(fmt=fmt)


@pytest.mark.parametrize(
    "doc_id",
    ["", "abc", "a" * 63, "a" * 65, "g" * 64, "A" * 64],
)
def test_rejects_invalid_document_id(doc_id: str) -> None:
    with pytest.raises(PathValidationError):
        _build_raw(document_id=doc_id)


def test_quarantine_rejects_unknown_reason() -> None:
    with pytest.raises(PathValidationError):
        _build_quarantine(reason="not_a_reason")


def test_uses_first_twelve_hex_chars_of_doc_id() -> None:
    doc_id = "0123456789abcdef" + "0" * 48
    key = _build_raw(document_id=doc_id)
    assert "__id=0123456789ab__" in key


def test_each_allowed_country_accepted() -> None:
    for country in ["ca", "uk", "us", "fr"]:
        _build_raw(country=country)


def test_each_allowed_quarantine_reason_accepted() -> None:
    for reason in [
        "download_failed",
        "oversize",
        "truncated_body",
        "unreadable_encoding",
        "path_collision",
    ]:
        _build_quarantine(reason=reason)
