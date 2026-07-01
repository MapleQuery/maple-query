"""Pydantic + JSON Schema constants this package owns.

The DatasetCard pydantic model lives in `types.py` (it's a data shape
imported by both the on-disk row and the staged-disk row); this module
keeps the matching JSON Schema constant that outlines hands the
decoder, plus the smoke-test schema used by 4.3.
"""
from __future__ import annotations

from typing import Any

import pydantic


class SmokeOutput(pydantic.BaseModel):
    """Tight closed-shape schema used by the smoke test.

    `extra="forbid"` means the constrained decoder cannot smuggle in
    unknown keys; if outlines's schema handling regresses, the smoke
    test trips immediately instead of silently passing.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    package_id: str = pydantic.Field(min_length=1)
    summary: str = pydantic.Field(min_length=1, max_length=500)


# JSON Schema the outlines decoder is constrained against. Mirrors the
# `DatasetCard` pydantic model in `types.py`. Both must stay in lock-
# step; `additionalProperties: false` + `extra="forbid"` is belt and
# suspenders.
DATASET_CARD_GUIDED_JSON: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["package_id", "summary"],
    "properties": {
        "package_id": {"type": "string"},
        "summary": {"type": "string", "minLength": 50, "maxLength": 1200},
        "grain": {"type": "string"},
        "measures": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 20,
        },
        "dimensions": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 20,
        },
        "date_range_start": {"type": ["string", "null"], "format": "date"},
        "date_range_end": {"type": ["string", "null"], "format": "date"},
    },
}


# Per-chunk columns response. The prompt asks the model for one JSON
# array; outlines constrains the decoder against this schema. Mirrors
# the `ColumnOutput` pydantic model in `types.py`.
COLUMNS_GUIDED_JSON_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["column_name", "description"],
        "properties": {
            "column_name": {"type": "string"},
            "semantic_type": {"type": "string"},
            "description": {"type": "string", "minLength": 20, "maxLength": 600},
            "sample_values": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 10,
            },
        },
    },
}
