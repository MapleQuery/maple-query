"""Scope validation: defensive, non-raising, degrades to unscoped.

The scope arrives from a client, so it is untrusted — but it carries no
authorisation weight either, since it only selects among packages the
user could already reach by searching. That combination is what makes
"drop it silently" the right posture rather than "reject the request".
"""
from __future__ import annotations

from semantic_enrich.core.agent.scope import MAX_SCOPE_PACKAGES, sanitize

_VALID = "0f3765d1-3375-4423-8fd6-6da7f382fa1a"
_VALID_2 = "1a2b3c4d-5566-7788-99aa-bbccddeeff00"


def test_valid_ids_survive_in_order() -> None:
    assert sanitize([_VALID, _VALID_2]) == (_VALID, _VALID_2)


def test_excess_ids_are_capped_not_rejected() -> None:
    many = [f"pkg-{i:04d}-aaaa" for i in range(10)]
    kept = sanitize(many)
    assert len(kept) == MAX_SCOPE_PACKAGES
    assert kept == tuple(many[:MAX_SCOPE_PACKAGES])


def test_malformed_entries_are_dropped_without_raising() -> None:
    kept = sanitize(
        [
            _VALID,
            "",
            "   ",
            "short",  # below the length floor
            "-leading-hyphen-id",
            "has spaces in it",
            "DROP TABLE raw.rows; --",
            None,
            12345,
            {"package_id": _VALID_2},
        ]
    )
    assert kept == (_VALID,)


def test_all_malformed_degrades_to_unscoped() -> None:
    assert sanitize(["nope", "!!", None]) == ()


def test_empty_and_non_sequence_inputs_are_unscoped() -> None:
    assert sanitize([]) == ()
    assert sanitize(()) == ()
    assert sanitize(None) == ()
    assert sanitize("a-single-string-id") == ()
    assert sanitize({"a": 1}) == ()


def test_duplicates_collapse() -> None:
    assert sanitize([_VALID, _VALID, _VALID]) == (_VALID,)


def test_ids_are_normalized_before_matching() -> None:
    assert sanitize([f"  {_VALID.upper()}  "]) == (_VALID,)


def test_cap_counts_kept_ids_not_inspected_ones() -> None:
    """Malformed entries must not consume cap slots — otherwise a
    client sending junk first would silently lose valid ids."""
    mixed = ["bad", "!!", _VALID, "also bad", _VALID_2]
    assert sanitize(mixed) == (_VALID, _VALID_2)
