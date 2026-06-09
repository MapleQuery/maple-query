"""Load BQ schemas from `infra/terraform/schemas/*.json`.

One source of truth shared with Terraform's `google_bigquery_table`
resource (3.1 §4.5). Keeping it in JSON means schema diffs read as
ordinary JSON diffs in PR review, and there's no Python-side schema
literal that can drift from the HCL.
"""
from __future__ import annotations

import json
from pathlib import Path

from google.cloud import bigquery


def load_schema(path: Path) -> list[bigquery.SchemaField]:
    """Parse a BQ schema JSON file into a list of `SchemaField` objects.

    Raises whatever `from_api_repr` raises on malformed input — that's
    what we want the §11.1 CI check to assert against.
    """
    with path.open() as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"schema file {path} must be a JSON array, got {type(raw).__name__}")
    return [bigquery.SchemaField.from_api_repr(field) for field in raw]
