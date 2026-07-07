"""Deterministic sample-row + representative-resource selection.

Pure functions, invoked by `dataset_extract` per package:

- `pick_representative(resources)` — median-row-count resource, lower-
  `document_id` tiebreak; resources whose headers look like a data
  dictionary are demoted out of the pool when per-doc headers are
  supplied.
- `looks_like_dictionary(columns)` — header-vocabulary heuristic for
  schema/dictionary CSVs that ship alongside the data CSV they
  describe.
- `derive_indices(document_id, row_count, k)` — k sample-row indices
  seeded from sha256(document_id), so two extracts of the same
  document against the same `raw.*` snapshot pick the same rows.

`hash()` is not used because Python's per-process hash salt would
re-randomise the selection between runs.
"""
from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Mapping, Sequence

from semantic_enrich.types import PackageResource

# Fixed vocabulary of header names that describe a schema rather than
# data. A dictionary CSV's headers are drawn almost entirely from this
# set ("COLUMN_NAME, DATA_TYPE, DESCRIPTION"); a data CSV's headers are
# domain terms. Threshold and column cap below are starting points —
# tuned against observed dictionary CSVs, revisit if the picker logs
# show false demotions.
_DICTIONARY_HEADER_SET = frozenset({
    "column_name", "column", "field_name", "field",
    "data_type", "type", "datatype",
    "description", "definition", "notes",
    "example", "sample", "sample_value",
    "constraint", "constraints", "nullable",
})
_DICTIONARY_HEADER_RATIO = 0.6
_DICTIONARY_MAX_COLUMNS = 8


def looks_like_dictionary(columns: Sequence[str]) -> bool:
    """True when the header set is dictionary-shaped: <=8 columns and
    >=60% of them drawn from the schema-description vocabulary.

    Empty/unknown headers return False — without evidence the resource
    is treated as data, never demoted."""
    if not columns or len(columns) > _DICTIONARY_MAX_COLUMNS:
        return False
    normalized = [c.strip().lower().replace(" ", "_") for c in columns]
    matches = sum(1 for c in normalized if c in _DICTIONARY_HEADER_SET)
    return matches / len(normalized) >= _DICTIONARY_HEADER_RATIO


def pick_representative(
    resources: Iterable[PackageResource],
    *,
    columns_by_doc: Mapping[str, Sequence[str]] | None = None,
) -> PackageResource:
    """Return the median-row-count resource.

    When `columns_by_doc` (document_id → header names) is provided,
    dictionary-shaped resources are excluded from the pool first; they
    only remain eligible when *every* resource is dictionary-shaped
    (a package publishing nothing but a dictionary still deserves a
    representative for browseability).

    Tiebreak: lower `document_id` (lexicographic) wins. Resources with
    `row_count is None` are sorted after resources with a known count
    (so they never win the median in a mixed set) — this is a
    defensive case; the extract candidate query filters
    `load_status='loaded'`, so row_count should be non-null.
    """
    pool = list(resources)
    if not pool:
        raise ValueError("pick_representative requires at least one resource")

    if columns_by_doc:
        non_dict = [
            r for r in pool
            if not looks_like_dictionary(columns_by_doc.get(r.document_id, ()))
        ]
        pool = non_dict or pool

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
