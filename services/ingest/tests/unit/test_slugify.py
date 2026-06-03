from __future__ import annotations

import pytest

from ingest.core.slugify import slugify


@pytest.mark.parametrize(
    ("url", "fmt", "expected"),
    [
        # Canonical examples covering the main slugify transforms.
        (
            "https://open.canada.ca/data/dataset/abc/Budget%202023%20%E2%80%93%20EN.xlsx",
            "xlsx",
            "budget-2023-en.xlsx",
        ),
        ("https://example.gov/data/Some%20File%20Name.pdf", "pdf", "some-file-name.pdf"),
        ("https://example.gov/RAW", "unknown", "raw"),
        ("https://example.gov/$$$.csv", "csv", "file.csv"),
    ],
)
def test_prd_pinned_examples(url: str, fmt: str, expected: str) -> None:
    assert slugify(resource_url=url, fmt=fmt) == expected


def test_truncates_to_150_chars_preserving_extension() -> None:
    url = "https://example.gov/" + "a" * 300 + ".csv"
    result = slugify(resource_url=url, fmt="csv")
    assert len(result) == 150
    assert result.endswith(".csv")


def test_empty_segment_falls_back_to_file() -> None:
    result = slugify(resource_url="https://example.gov/", fmt="csv")
    assert result == "file.csv"


def test_accents_stripped_via_nfkd() -> None:
    result = slugify(resource_url="https://example.gov/R%C3%A9sum%C3%A9.pdf", fmt="pdf")
    assert result == "resume.pdf"


def test_unknown_fmt_skips_extension() -> None:
    result = slugify(resource_url="https://example.gov/somefile", fmt="unknown")
    assert result == "somefile"


def test_unknown_fmt_preserves_url_extension_chars() -> None:
    # fmt=unknown means we didn't sniff a format; we don't strip the URL's
    # extension-looking suffix because we have no canonical fmt to substitute.
    result = slugify(resource_url="https://example.gov/foo.pdf", fmt="unknown")
    assert result == "foo.pdf"


def test_appends_extension_when_url_has_none() -> None:
    result = slugify(resource_url="https://example.gov/dataset", fmt="csv")
    assert result == "dataset.csv"


def test_appends_extension_when_url_extension_differs() -> None:
    # Declared/sniffed fmt overrides URL suffix; both end up in the filename.
    result = slugify(resource_url="https://example.gov/foo.txt", fmt="csv")
    assert result == "foo.txt.csv"


def test_collapses_dashes_and_strips_edges() -> None:
    result = slugify(resource_url="https://example.gov/--a___b--c--.csv", fmt="csv")
    assert result == "a___b-c.csv"


def test_emoji_replaced_with_dash() -> None:
    # %F0%9F%92%A9 = pile of poo emoji
    result = slugify(resource_url="https://example.gov/data%F0%9F%92%A9file.csv", fmt="csv")
    assert result == "data-file.csv"
