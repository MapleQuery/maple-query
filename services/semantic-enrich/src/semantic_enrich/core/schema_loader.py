"""Load BQ schemas from `infra/terraform/schemas/*.json`.

Same JSON is consumed by Terraform's `google_bigquery_table` — one
source of truth, so schema diffs read as ordinary JSON diffs and no
Python literal can drift from the HCL. Identical contract to
warehouse-load/core/schema_loader.py; kept duplicated rather than
extracted into a shared package because cross-service Python imports
are an explicit non-goal of the layout.
"""
from __future__ import annotations

import json
from pathlib import Path

from google.cloud import bigquery


def load_schema(path: Path) -> list[bigquery.SchemaField]:
    """Parse a BQ schema JSON file into a list of `SchemaField` objects."""
    with path.open() as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(
            f"schema file {path} must be a JSON array, got {type(raw).__name__}"
        )
    return [bigquery.SchemaField.from_api_repr(field) for field in raw]


def assert_datasets_schema(schema: list[bigquery.SchemaField]) -> None:
    """Drift guard for `semantic.datasets`.

    Raises at startup of any subcommand that touches the load path if
    `embedding` has moved away from `ARRAY<FLOAT64>`. Catches a 4.2
    schema change that 4.4 hasn't been updated for, before any load
    job runs.
    """
    if not any(
        f.name == "embedding"
        and f.field_type == "FLOAT64"
        and f.mode == "REPEATED"
        for f in schema
    ):
        raise AssertionError(
            "semantic_datasets.json must declare embedding ARRAY<FLOAT64>"
        )
