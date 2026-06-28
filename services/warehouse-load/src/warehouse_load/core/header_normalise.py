"""Header-name → JSON key normalisation.

Whitespace collapse, case preservation, empty-name fallback,
duplicate-suffix. The output keys land verbatim as the JSON object
keys in `raw.rows.row`, so they are also the names the curated
layer will reference.

Case preservation is deliberate: bilingual datasets pair `EN_YEAR`
with `FR_ANNEE` and folding case loses the language signal that
distinguishes them.
"""
from __future__ import annotations

import re

_WHITESPACE_RUN = re.compile(r"\s+")


def normalise_keys(raw_keys: list[str]) -> list[str]:
    """Convert a list of raw header names into JSON keys.

    Steps (in order, per PRD §6.4):

    1. Strip + collapse whitespace runs to a single `_`.
    2. Preserve case.
    3. Empty → `__col_<index>`.
    4. Suffix duplicate normalised names with `__2`, `__3`, ...

    Returns a list of the same length as `raw_keys`. Idempotent
    under re-application: normalising the output again yields the
    same list (the duplicate-suffix is only applied on first pass
    because the index space stays disjoint).
    """
    intermediate: list[str] = []
    for index, raw in enumerate(raw_keys):
        intermediate.append(_normalise_one(raw, index))

    seen: dict[str, int] = {}
    out: list[str] = []
    for name in intermediate:
        count = seen.get(name, 0)
        if count == 0:
            out.append(name)
        else:
            # `__2` for the first repeat (the original is `__1`-implicit),
            # `__3` next, and so on — first appearance keeps the bare name.
            out.append(f"{name}__{count + 1}")
        seen[name] = count + 1
    return out


def _normalise_one(raw: str, index: int) -> str:
    """Whitespace-collapse + empty-fallback. Case preserved."""
    stripped = raw.strip() if raw else ""
    if not stripped:
        return f"__col_{index}"
    return _WHITESPACE_RUN.sub("_", stripped)
