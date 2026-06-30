"""Deterministic sample-row + representative-resource selection.

Two pure functions, both invoked by `dataset_extract` per package:

- `pick_representative(resources)` — median-row-count resource, lower-
  `document_id` tiebreak.
- `derive_indices(document_id, row_count, k)` — k sample-row indices
  seeded from sha256(document_id), so two extracts of the same
  document against the same `raw.*` snapshot pick the same rows.

`hash()` is not used because Python's per-process hash salt would
re-randomise the selection between runs.
"""
from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable

from semantic_enrich.types import PackageResource


def pick_representative(resources: Iterable[PackageResource]) -> PackageResource:
    """Return the median-row-count resource.

    Tiebreak: lower `document_id` (lexicographic) wins. Resources with
    `row_count is None` are sorted after resources with a known count
    (so they never win the median in a mixed set) — this is a
    defensive case; the extract candidate query filters
    `load_status='loaded'`, so row_count should be non-null.
    """
    pool = list(resources)
    if not pool:
        raise ValueError("pick_representative requires at least one resource")

    def key(r: PackageResource) -> tuple[int, int, str]:
        # (known-count first, then row_count, then document_id)
        if r.row_count is None:
            return (1, 0, r.document_id)
        return (0, r.row_count, r.document_id)

    pool.sort(key=key)
    return pool[len(pool) // 2]


def derive_indices(*, document_id: str, row_count: int, k: int) -> list[int]:
    """Pick `k` indices from `range(row_count)`, deterministically.

    If `row_count < k`, returns all `row_count` indices. The seed
    derives from sha256(document_id) so the same document always
    produces the same indices across processes.
    """
    if row_count <= 0:
        raise ValueError(
            f"derive_indices requires row_count > 0; got {row_count}"
        )
    if k <= 0:
        raise ValueError(f"derive_indices requires k > 0; got {k}")
    seed = int(hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    return sorted(rng.sample(range(row_count), k=min(k, row_count)))


def truncate_cell(value: object, *, max_chars: int = 200) -> str | None:
    """Render one cell value as a (possibly truncated) string.

    The sample rows feed the prompt's `sample_rows_block` and the JSONL
    stage. Limiting per-cell to 200 chars keeps the prompt bounded for
    pathological wide-text columns (e.g. embedded HTML, multi-paragraph
    notes) and keeps the JSONL human-inspectable in a terminal.
    """
    if value is None:
        return None
    s = str(value)
    if len(s) > max_chars:
        return s[: max_chars - 1] + "…"
    return s
