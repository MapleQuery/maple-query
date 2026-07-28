"""Header detection, checked against spreadsheets that really exist.

Every case in `ACCEPT_CASES` and `DECLINE_CASES` is a real document: the
rows come from the warehouse verbatim, and the expected outcome comes
from reading those rows, not from running the detector. A
test written from the same misunderstanding as the code preserves the
bug instead of catching it, so the header row index of each accept was
established by looking at the spreadsheet and asking what a person would
call the header — then the detector was made to agree.

The decline cases carry the same weight as the accepts, and there are
more of them. A wrong name is worse than no name, so the interesting
question is not "how much did we recover" but "did we ever name a column
something it is not".
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from semantic_enrich.core.header_recovery import (
    HEADER_SCAN_ROWS,
    detect_header,
    explain_header,
)

_FIXTURE = (
    Path(__file__).parent.parent / "fixtures" / "header_recovery_documents.json"
)


def _documents() -> dict[str, dict[str, Any]]:
    raw = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    return {doc["document_id"]: doc for doc in raw}


DOCUMENTS = _documents()


def _doc(prefix: str) -> dict[str, Any]:
    matches = [d for k, d in DOCUMENTS.items() if k.startswith(prefix)]
    assert len(matches) == 1, f"{prefix} matched {len(matches)} fixture documents"
    return matches[0]


def _detect(prefix: str, **kwargs: Any) -> Any:
    doc = _doc(prefix)
    return detect_header(doc["rows"], doc["generated_columns"], **kwargs)


# (document prefix, expected header row, what the spreadsheet looks like)
ACCEPT_CASES = [
    (
        "43d968b125",
        1,
        "housing benefit table 1: one blank row, then "
        "Province/Territory | Number of Unique Applicants | Total Amount ($000)",
    ),
    (
        "cdf5065c01",
        2,
        "housing benefit table 2: blank row, an 'Age Group' banner, then the "
        "real header of age bands",
    ),
    ("3d5f92333b", 2, "housing benefit table 3: same shape, dollar amounts"),
    ("4144936e1f", 2, "housing benefit table 4: same shape, gender columns"),
    ("b5fe239d7a", 2, "housing benefit table 5: same shape, gender amounts"),
    ("f0ddafb474", 1, "housing benefit table 6: by adjusted family net income"),
    ("86390ed722", 1, "housing benefit table 7: by family type"),
    (
        "e6fe36c3fa",
        4,
        "immigration overview: three note lines and a blank above the header",
    ),
    (
        "0810ec3604",
        1,
        "cyber centre contact stats: a bilingual title line above the header",
    ),
    (
        "561a96eb02",
        0,
        "federal tax expenditures: header on row 0, and eight of its fourteen "
        "columns are named after years",
    ),
    (
        "95816e1374",
        0,
        "Alberta deaths by cause: header on row 0, twenty-five unnamed columns",
    ),
    (
        "fdf3e12873",
        0,
        "CSIS pay scale: header on row 0, but only two of the four unnamed "
        "columns carry a name in it",
    ),
    ("3b6f7bc41f", 0, "temporary foreign worker LMIAs: header on row 0"),
    (
        "5f797ad093",
        0,
        "briefing notes (ISC): an all-text header whose contrast comes from "
        "the date column below it",
    ),
    ("9ce63876ed", 0, "briefing notes (CIRNAC): the same shape, other department"),
    (
        "18930d3aea",
        2,
        "legal aid applications: merged 'Criminal' and 'Civil' labels one "
        "row above the leaf names they span — recovered by composing the "
        "two tiers, so the header row is the leaf",
    ),
]

# (document prefix, expected decline reason, why the document declines)
DECLINE_CASES = [
    (
        "57f2ef5417",
        "tier_split",
        "legal aid expenditures: a genuine THREE-tier header. Composition "
        "handles two tiers; three is where telling a unit row from a "
        "banner starts, so this stays declined",
    ),
    (
        "9d9358fe99",
        "no_generated_names",
        "cadet applications: a French second header row, and every one of "
        "its twenty-three unnamed columns is empty in every row — there "
        "is nothing there to name",
    ),
    (
        "deba8847a0",
        "no_data_rows_in_window",
        "an HTML error page ingested as a one-column CSV — all text, no data",
    ),
    (
        "dfe3ec6549",
        "no_data_rows_in_window",
        "survey report prose: question text in column 0, nothing else filled",
    ),
    (
        "759ce236f4",
        "data_starts_at_row_0",
        "election results: the header never made it into the file",
    ),
    ("235723ded4", "data_starts_at_row_0", "health dashboard: data from row 0"),
    ("b933807b1d", "data_starts_at_row_0", "wholesale prices: data from row 0"),
    ("7e563f6e2f", "data_starts_at_row_0", "NRC publications: data from row 0"),
    ("db4482cbd6", "data_starts_at_row_0", "small business financing: data from row 0"),
    (
        "61fbbfbc45",
        "data_starts_at_row_0",
        "radon survey: named columns plus a ragged tail of empty ones",
    ),
    ("762f7b5f3f", "data_starts_at_row_0", "ACROSS error codes: ragged empty tail"),
    ("a447f09d43", "data_starts_at_row_0", "occupational classification: ragged tail"),
    ("37f2027303", "data_starts_at_row_0", "municipal voting areas: ragged tail"),
    (
        "7da766a390",
        "data_starts_at_row_0",
        "telemetry: the single unnamed column is a row counter",
    ),
]


@pytest.mark.parametrize(("prefix", "expected_index", "shape"), ACCEPT_CASES)
def test_accepts_real_documents(
    prefix: str, expected_index: int, shape: str
) -> None:
    recovery = _detect(prefix)
    assert recovery is not None, f"expected recovery for {shape}"
    assert recovery.header_row_index == expected_index
    assert recovery.preamble_rows == expected_index


# Documents recovered by composing two header tiers rather than reading
# one row. Their names are joins of two cells, so the single-row verbatim
# invariant does not apply — a stricter one is asserted for them below.
COMPOSED = {"18930d3aea"}


@pytest.mark.parametrize(("prefix", "expected_index", "shape"), ACCEPT_CASES)
def test_accepted_names_come_verbatim_from_the_header_row(
    prefix: str, expected_index: int, shape: str
) -> None:
    """Every name is a cell of the human-identified header row.

    This is the one hard gate expressed structurally: the row
    index is established by reading the spreadsheet, so tying the names to
    that row leaves the detector no room to invent one.
    """
    if prefix in COMPOSED:
        pytest.skip("composed from two tiers; see the composition test")
    doc = _doc(prefix)
    recovery = _detect(prefix)
    assert recovery is not None
    header = doc["rows"][expected_index]
    for column, name in recovery.names.items():
        assert column in doc["generated_columns"]
        cell = header.get(column)
        assert cell is not None
        assert name == " ".join(str(cell).split())
        assert name


@pytest.mark.parametrize("prefix", sorted(COMPOSED))
def test_composed_names_are_built_only_from_the_two_tier_cells(
    prefix: str,
) -> None:
    """Composition is the one place a name is not a single cell, so the
    invariant is weakened by exactly one step and no further: every part
    of a composed name is a cell of one of the two tiers, at that same
    column. Nothing is synthesised, nothing is carried in from a third
    row, and nothing is reworded."""
    doc = _doc(prefix)
    recovery = _detect(prefix)
    assert recovery is not None
    leaf = doc["rows"][recovery.header_row_index]
    for column, name in recovery.names.items():
        parts = [
            " ".join(str(row.get(column) or "").split())
            for row in doc["rows"][: recovery.header_row_index + 1]
        ]
        cells = {p for p in parts if p}
        leaf_cell = " ".join(str(leaf.get(column) or "").split())
        # The name is either one of this column's own cells, or a cell
        # from an upper tier followed by this column's leaf cell.
        assert name in cells or (
            leaf_cell
            and name.endswith(leaf_cell)
            and name[: -len(leaf_cell)].strip() in cells
        ), f"{column}: {name!r} not built from this column's cells {cells}"


def test_housing_table_one_names_the_amount_column() -> None:
    """The column the loop could describe but could not sum."""
    recovery = _detect("43d968b125")
    assert recovery is not None
    assert recovery.names == {
        "__col_1": "Number of Unique Applicants",
        "__col_2": "Total Amount ($000)",
    }


def test_housing_table_two_names_every_age_band() -> None:
    recovery = _detect("cdf5065c01")
    assert recovery is not None
    assert recovery.names == {
        "__col_1": "Under 25",
        "__col_2": "25-34",
        "__col_3": "35-44",
        "__col_4": "45-54",
        "__col_5": "55-64",
        "__col_6": "65+",
        "__col_7": "Total",
    }


def test_year_columns_are_recovered_as_names() -> None:
    """A fiscal table names columns after years. Treating a bare year as a
    value would decline the whole document."""
    recovery = _detect("561a96eb02")
    assert recovery is not None
    assert recovery.names["__col_6"] == "2019"
    assert recovery.names["__col_13"] == "2026"
    assert recovery.names["__col_0"] == "MEASURE"


def test_columns_the_header_row_leaves_blank_stay_unnamed() -> None:
    """CSIS pay scale has four unnamed columns; its header row fills two.
    The other two keep their positional keys rather than getting an empty
    name."""
    doc = _doc("fdf3e12873")
    recovery = _detect("fdf3e12873")
    assert recovery is not None
    assert recovery.names == {"__col_1": "Minimum", "__col_2": "Maximum"}
    assert {"__col_3", "__col_4"} <= set(doc["generated_columns"])


def test_named_columns_are_never_renamed() -> None:
    """The radon survey carries seven real column names. Whatever the
    detector decides, none of them may appear in `names`."""
    for prefix, _index, _shape in ACCEPT_CASES:
        doc = _doc(prefix)
        recovery = _detect(prefix)
        assert recovery is not None
        generated = set(doc["generated_columns"])
        assert set(recovery.names) <= generated


@pytest.mark.parametrize(("prefix", "reason", "why"), DECLINE_CASES)
def test_declines_real_documents(prefix: str, reason: str, why: str) -> None:
    doc = _doc(prefix)
    report = explain_header(doc["rows"], doc["generated_columns"])
    assert report.recovery is None, f"expected decline: {why}"
    assert report.reason == reason


def test_multi_tier_header_declines_rather_than_picking_a_tier() -> None:
    """Legal aid expenditures spreads its header over three rows:
    `Jurisdiction | Total ... | Direct Legal Aid Services Expenditures`,
    then `Criminal matters | Civil matters` under it, then `I&R | All
    other civil` under that. No single row names the columns, so naming
    from any one of them would be wrong."""
    assert _detect("57f2ef5417") is None


def test_two_tier_header_over_repeated_leaf_labels_composes() -> None:
    """The shape from statistical releases: a year tier above a metric
    tier. The lower row fills every column and sits directly above the
    data, but repeats itself — `Applicants | Amount | Applicants |
    Amount` names nothing on its own. The upper tier is what makes each
    column addressable, so the two compose."""
    rows = [
        {"a": "", "__col_1": "2023", "__col_2": "2023", "__col_3": "2024", "__col_4": "2024"},
        {
            "a": "Region",
            "__col_1": "Applicants",
            "__col_2": "Amount",
            "__col_3": "Applicants",
            "__col_4": "Amount",
        },
        {"a": "ON", "__col_1": "5", "__col_2": "10", "__col_3": "6", "__col_4": "12"},
    ]
    generated = ["__col_1", "__col_2", "__col_3", "__col_4"]
    recovery = detect_header(rows, generated)
    assert recovery is not None
    assert recovery.names == {
        "__col_1": "2023 Applicants",
        "__col_2": "2023 Amount",
        "__col_3": "2024 Applicants",
        "__col_4": "2024 Amount",
    }


def test_composition_still_declines_when_it_cannot_disambiguate() -> None:
    """Composing is only worth doing if it separates the columns. Two
    tiers that still collide leave the same wrong name they started
    with."""
    rows = [
        {"__col_0": "2023", "__col_1": "2023"},
        {"__col_0": "Amount", "__col_1": "Amount"},
        {"__col_0": "5", "__col_1": "6"},
    ]
    assert detect_header(rows, ["__col_0", "__col_1"]) is None


def test_composition_needs_a_recoverable_column_order() -> None:
    """Forward-filling a merged label means knowing which columns it
    spans. Row bodies arrive with their keys alphabetised, so position
    survives only inside the `__col_N` names — two columns carrying real
    names leave the layout genuinely ambiguous, and composition declines
    rather than guessing which one a span reaches."""
    rows = [
        {"Alpha": "2023", "Beta": "", "__col_2": "", "__col_3": ""},
        {"Alpha": "", "Beta": "x", "__col_2": "Amount", "__col_3": "Amount"},
        {"Alpha": "ON", "Beta": "y", "__col_2": "5", "__col_3": "6"},
    ]
    assert detect_header(rows, ["__col_2", "__col_3"]) is None


def test_a_fill_never_runs_past_the_leaf_into_the_empty_tail() -> None:
    """A spanning label forward-fills across the columns the leaf names,
    and stops. Letting it run into the ragged empty tail handed two dead
    columns the same name, which the SQL layer would have resolved to
    whichever it saw first."""
    rows = [
        {"__col_0": "2013", "__col_1": "", "__col_2": "", "__col_3": ""},
        {"__col_0": "COUNT", "__col_1": "SUM", "__col_2": "", "__col_3": ""},
        {"__col_0": "5", "__col_1": "10", "__col_2": "", "__col_3": ""},
    ]
    recovery = detect_header(rows, ["__col_0", "__col_1", "__col_2", "__col_3"])
    assert recovery is not None
    assert set(recovery.names) == {"__col_0", "__col_1"}
    assert len(set(recovery.names.values())) == len(recovery.names)


def test_adjacent_qualifying_rows_decline() -> None:
    """Two stacked rows that each look like a header are a two-tier header.
    Either one alone is a wrong name, so neither is used."""
    rows = [
        {"__col_0": "Region", "__col_1": "Population", "__col_2": "Area"},
        {"__col_0": "Province", "__col_1": "Persons", "__col_2": "Square km"},
        {"__col_0": "ON", "__col_1": "15000000", "__col_2": "1076395"},
    ]
    report = explain_header(rows, ["__col_0", "__col_1", "__col_2"])
    assert report.recovery is None
    assert report.reason == "multi_tier"


def test_preamble_deeper_than_the_scan_window_declines() -> None:
    """The immigration overview's header sits on row 4. Told to look at
    three rows, the detector must not settle for a note line."""
    doc = _doc("e6fe36c3fa")
    assert detect_header(doc["rows"], doc["generated_columns"], scan_rows=3) is None
    assert detect_header(doc["rows"], doc["generated_columns"]) is not None


def test_empty_input_returns_none() -> None:
    assert detect_header([], ["__col_1"]) is None


def test_single_row_returns_none() -> None:
    """One row cannot be both the header and the evidence that it is one."""
    assert detect_header([{"__col_0": "Region"}], ["__col_0"]) is None


def test_no_generated_columns_returns_none() -> None:
    doc = _doc("cdf5065c01")
    assert detect_header(doc["rows"], []) is None


def test_rows_of_only_blanks_return_none() -> None:
    rows: list[dict[str, object]] = [{"__col_0": None}, {"__col_0": ""}]
    assert detect_header(rows, ["__col_0"]) is None


def test_header_cell_that_is_itself_a_positional_key_is_not_used() -> None:
    """A recovered name has to be worth more than the key it replaces."""
    rows = [
        {"__col_0": "__col_0", "__col_1": "Region"},
        {"__col_0": "5", "__col_1": "ON"},
    ]
    recovery = detect_header(rows, ["__col_0", "__col_1"])
    assert recovery is not None
    assert recovery.names == {"__col_1": "Region"}


def test_default_scan_window_is_eight_rows() -> None:
    assert HEADER_SCAN_ROWS == 8


def test_fixture_covers_more_declines_than_accepts_across_publishers() -> None:
    """Guard on the fixture itself. The detector is a heuristic over other
    people's spreadsheets, and a fixture drawn from one publisher would
    prove nothing about the next one."""
    publishers = {doc["organization_code"] for doc in DOCUMENTS.values()}
    assert len(publishers) >= 15
    assert len(DECLINE_CASES) >= 10
    assert len(ACCEPT_CASES) + len(DECLINE_CASES) == len(DOCUMENTS)


def test_a_merged_cell_tier_is_composed_rather_than_taken_alone() -> None:
    """The failure this rule was written for, from the document that
    exposed it.

    `Legal Aid in Canada` puts `Criminal` and `Civil` on its top row as
    merged labels spanning two and three columns; the columns' real names
    (`Adult matter applications`, `Youth matter applications`, ...) sit a
    row below. Taking the top row named one column "Criminal" when it
    holds adult matters only and left youth unnamed — a total that looks
    complete and is not.

    Neither of the two gates meant to catch merged cells fired: the row
    sat exactly on the density threshold, and a merged label arrives as
    blanks rather than as a repeated value, so distinctness stayed clean.
    """
    doc = _doc("18930d3aea")
    report = explain_header(doc["rows"], doc["generated_columns"])
    assert report.recovery is not None
    assert report.reason == "accepted_composed"
    names = report.recovery.names
    # The column that used to be named "Criminal" now says which of the
    # two criminal columns it is, and its sibling is no longer nameless.
    assert names["__col_3"] == "Criminal Adult matter applications"
    assert names["__col_4"] == "Youth matter applications"
    assert names["__col_7"] == "Civil Family matter applications"
    # Every unnamed column got a name, and no two share one.
    assert set(names) == set(doc["generated_columns"])
    assert len(set(names.values())) == len(names)


def test_a_banner_above_a_complete_header_still_recovers() -> None:
    """The rule keys on the header leaving a *gap* another row fills, not
    on two rows sharing a column. The housing table has `Age Group`
    spanning the age columns one row above a header that names every
    column itself — nothing is missing, so nothing is ambiguous."""
    recovery = _detect("cdf5065c01")
    assert recovery is not None
    assert recovery.header_row_index == 2
    assert len(recovery.names) == 7


def test_a_section_label_below_the_header_is_not_a_tier() -> None:
    """A row that fills a column the header left blank only matters when
    that column is one we would otherwise leave unnamed. A section label
    under a header whose blank sits over an already-named column is not a
    split header."""
    rows = [
        {"Region": "", "__col_1": "2023", "__col_2": "2024"},
        {"Region": "Ontario", "__col_1": "", "__col_2": ""},
        {"Region": "Toronto", "__col_1": "5", "__col_2": "6"},
    ]
    recovery = detect_header(rows, ["__col_1", "__col_2"])
    assert recovery is not None
    assert recovery.names == {"__col_1": "2023", "__col_2": "2024"}


def test_a_split_header_is_joined_across_its_two_rows() -> None:
    """A split header can put the missing names on either side. Here the
    row nearest the data names two of three columns and the row above
    carries the third, so the pair is read together — and `Region` is
    NOT smeared across the other two, because the leaf labels are already
    unique and need no disambiguating."""
    rows = [
        {"__col_0": "Region", "__col_1": "", "__col_2": ""},
        {"__col_0": "", "__col_1": "Adult", "__col_2": "Youth"},
        {"__col_0": "ON", "__col_1": "5", "__col_2": "6"},
    ]
    recovery = detect_header(rows, ["__col_0", "__col_1", "__col_2"])
    assert recovery is not None
    assert recovery.names == {
        "__col_0": "Region",
        "__col_1": "Adult",
        "__col_2": "Youth",
    }
