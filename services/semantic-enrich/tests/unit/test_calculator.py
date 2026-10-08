from __future__ import annotations

import pytest

from semantic_enrich.core.calculator import CalcError, evaluate


def test_real_change() -> None:
    assert round(evaluate("(42737/31423) / (164.2/126.6) - 1"), 4) == 0.0486


def test_thousands_separators_and_symbols() -> None:
    assert evaluate("1,000 \u00d7 3 \u00f7 2") == 1500


@pytest.mark.parametrize("bad", ["__import__('os')", "a + 1", "2 ** 1000", "1/0", "[1,2]"])
def test_rejects_anything_but_arithmetic(bad: str) -> None:
    with pytest.raises(CalcError):
        evaluate(bad)
