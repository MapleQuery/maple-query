"""Load BQ schemas from `infra/terraform/schemas/*.json`.

Same JSON is consumed by Terraform's `google_bigquery_table` — one
source of truth, so schema diffs read as ordinary JSON diffs and no
Python literal can drift from the HCL.
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
        raise ValueError(f"schema file {path} must be a JSON array, got {type(raw).__name__}")
    return [bigquery.SchemaField.from_api_repr(field) for field in raw]
