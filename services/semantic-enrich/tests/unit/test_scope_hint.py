"""The scope hint template. Pure, no model, no state.

The hint is a *preference*: it steers the model at the scoped packages
but never forbids a broad search. A stale chip — a notebook block whose
dataset was re-ingested — must degrade to an ordinary turn, not a dead
one, so the wording has to leave the escape hatch open.
"""
from __future__ import annotations

from semantic_enrich.core.agent.scope import (
    render_hint,
    titles_from_records,
)

_ID = "0f3765d1-3375-4423-8fd6-6da7f382fa1a"
_ID_2 = "1a2b3c4d-5566-7788-99aa-bbccddeeff00"


def _record(*packages: tuple[str, str | None]) -> dict[str, object]:
    return {
        "packages": [
            {"package_id": pid, "title": title} for pid, title in packages
        ]
    }


def test_hint_renders_titles_when_known() -> None:
    out = render_hint([_ID], titles={_ID: "Supplementary Estimates B"})
    assert "*Supplementary Estimates B*" in out
    assert f"`{_ID}`" in out


def test_hint_falls_back_to_the_bare_id() -> None:
    out = render_hint([_ID], titles={})
    assert f"`{_ID}`" in out
    assert "**" not in out


def test_hint_keeps_the_broad_search_escape_hatch() -> None:
    # A preference, not a filter. If this sentence disappears a stale
    # scope produces a dead turn instead of a degraded one.
    out = render_hint([_ID], titles={})
    assert "unless these turn up nothing relevant" in out
    assert "list_documents" in out


def test_empty_scope_renders_nothing() -> None:
    assert render_hint([], titles={}) == ""


def test_long_titles_are_truncated() -> None:
    out = render_hint([_ID], titles={_ID: "Estimates " + "X" * 100})
    rendered = out.split("*")[1]
    assert len(rendered) == 70
    assert rendered.endswith("…")


def test_titles_resolve_from_client_records() -> None:
    titles = titles_from_records(
        [_record((_ID, "Estimates B"), (_ID_2, None))]
    )
    assert titles == {_ID: "Estimates B", _ID_2: None}


def test_malformed_records_never_raise() -> None:
    assert titles_from_records(None) == {}
    assert titles_from_records("nope") == {}
    assert titles_from_records([None, 5, {"packages": "bad"}]) == {}
    assert titles_from_records([{"packages": [None, {"title": "t"}]}]) == {}


def test_first_title_wins_across_records() -> None:
    titles = titles_from_records(
        [_record((_ID, "Original")), _record((_ID, "Later"))]
    )
    assert titles[_ID] == "Original"
