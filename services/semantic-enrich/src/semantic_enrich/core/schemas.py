"""Pydantic schemas this package owns.

Only the smoke-test schema lives here — the real dataset and column
schemas are owned by the downstream enrichment pipeline, which passes
them to `generate_json` directly.
"""
from __future__ import annotations

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
