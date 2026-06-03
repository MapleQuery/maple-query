from __future__ import annotations

from typing import Any

import pytest

from ingest.clients.ckan import Dataset, Resource
from ingest.core.language import detect_language, filter_resources_by_pairing


def _resource(**overrides: Any) -> Resource:
    base: dict[str, Any] = {
        "id": "r1",
        "url": "https://example.gov/data/file.csv",
        "language": [],
    }
    base.update(overrides)
    return Resource.model_validate(base)


def _dataset(resources: list[Resource]) -> Dataset:
    return Dataset.model_validate(
        {
            "id": "d1",
            "name": "ds",
            "title": "Ds",
            "organization": {"name": "fin"},
            "metadata_created": "2020-01-01T00:00:00",
            "metadata_modified": "2026-05-01T00:00:00",
            "subject": ["x"],
            "resources": [r.model_dump(by_alias=True) for r in resources],
        }
    )


# --- detect_language ------------------------------------------------------

@pytest.mark.parametrize(("langs", "expected"), [
    (["en"], "en"),
    (["eng"], "en"),
    (["fr"], "fr"),
    (["fra"], "fr"),
    (["en", "fr"], "en"),       # English wins ties
    (["fr", "en"], "en"),       # Order-independent
    (["EN"], "en"),             # Case-insensitive
])
def test_detect_language_from_declared(langs: list[str], expected: str) -> None:
    assert detect_language(_resource(language=langs)) == expected


@pytest.mark.parametrize(("url", "expected"), [
    ("https://example.gov/data/report-fra.csv", "fr"),
    ("https://example.gov/data/report-fr.csv", "fr"),
    ("https://example.gov/fr/report.csv", "fr"),
    # Mid-path `-fr-` deliberately doesn't match — too prone to false positives.
    ("https://example.gov/data-fr-2024/file.csv", "unknown"),
    ("https://example.gov/data/report-eng.csv", "en"),
    ("https://example.gov/data/report-en.csv", "en"),
    ("https://example.gov/en/report.csv", "en"),
    ("https://example.gov/data/plain.csv", "unknown"),
])
def test_detect_language_from_url(url: str, expected: str) -> None:
    assert detect_language(_resource(url=url, language=[])) == expected


def test_detect_language_filename_when_url_neutral() -> None:
    r = _resource(url="https://example.gov/data/file.csv", name="rapport-fra.csv", language=[])
    assert detect_language(r) == "fr"


def test_detect_language_declared_beats_url() -> None:
    r = _resource(url="https://example.gov/data/report-fra.csv", language=["en"])
    assert detect_language(r) == "en"


# --- filter_resources_by_pairing ------------------------------------------

def test_pairing_en_only() -> None:
    en = _resource(id="a", language=["en"])
    pairs = filter_resources_by_pairing(_dataset([en]))
    assert [(r.id, lang) for r, lang in pairs] == [("a", "en")]


def test_pairing_fr_only() -> None:
    fr = _resource(id="a", language=["fr"])
    pairs = filter_resources_by_pairing(_dataset([fr]))
    assert [(r.id, lang) for r, lang in pairs] == [("a", "fr")]


def test_pairing_mixed_drops_french() -> None:
    en = _resource(id="a", language=["en"])
    fr = _resource(id="b", language=["fr"])
    pairs = filter_resources_by_pairing(_dataset([en, fr]))
    assert [(r.id, lang) for r, lang in pairs] == [("a", "en")]


def test_pairing_unknown_treated_as_english_for_pairing() -> None:
    unknown = _resource(id="a", url="https://example.gov/no-lang-hint", language=[])
    fr = _resource(id="b", language=["fr"])
    pairs = filter_resources_by_pairing(_dataset([unknown, fr]))
    # Unknown is in the EN bucket → French sibling is skipped.
    assert [(r.id, lang) for r, lang in pairs] == [("a", "unknown")]


def test_pairing_en_plus_unknown_both_kept() -> None:
    en = _resource(id="a", language=["en"])
    unknown = _resource(id="b", url="https://example.gov/raw", language=[])
    pairs = filter_resources_by_pairing(_dataset([en, unknown]))
    assert {(r.id, lang) for r, lang in pairs} == {("a", "en"), ("b", "unknown")}


def test_pairing_only_unknown_ingests_all() -> None:
    u1 = _resource(id="a", url="https://example.gov/x", language=[])
    u2 = _resource(id="b", url="https://example.gov/y", language=[])
    pairs = filter_resources_by_pairing(_dataset([u1, u2]))
    assert {r.id for r, _ in pairs} == {"a", "b"}


def test_pairing_empty_dataset() -> None:
    assert filter_resources_by_pairing(_dataset([])) == []
