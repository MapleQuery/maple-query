from __future__ import annotations

import pytest

from curate.core.names import normalize, variants


@pytest.mark.parametrize(
    "raw",
    [
        "Pierre Poilievre",
        "POILIEVRE, Pierre",
        "Hon. Pierre Poilievre",
        "Pierre Poilievre, M.P.",
        "Poilièvre, Pierre J.",
        "The Honourable Pierre  Poilievre",
    ],
)
def test_one_person_many_spellings(raw: str) -> None:
    assert normalize(raw) == "pierre poilievre"


def test_hyphens_apostrophes_and_accents() -> None:
    assert normalize("Béchard, Bruno-Marie") == "bruno marie bechard"
    assert normalize("Seamus O'Regan") == "seamus o'regan"


def test_middle_initial_dropped_but_not_end_words() -> None:
    assert normalize("John A. Macdonald") == "john macdonald"
    assert normalize("A. Smith") == "a smith"


def test_variants_include_first_given_name() -> None:
    v = variants("Mary Jane Smith", given="Mary Jane", family="Smith")
    assert {"mary jane smith", "mary smith"} <= v


def test_empty_is_empty() -> None:
    assert normalize("") == ""
    assert variants("") == set()


def test_post_nominals_after_a_comma_are_not_a_flip() -> None:
    assert normalize("Pierre Poilievre, M.P., P.C.") == "pierre poilievre"
    assert normalize("Poilievre, Pierre, M.P.") == "pierre poilievre"
