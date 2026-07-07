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
    if not any(
        f.name == "representative_document_id"
        and f.field_type == "STRING"
        and f.mode == "NULLABLE"
        for f in schema
    ):
        raise AssertionError(
            "semantic_datasets.json must declare representative_document_id "
            "STRING NULLABLE"
        )


def assert_columns_schema(schema: list[bigquery.SchemaField]) -> None:
    """Drift guard for `semantic.columns`. Same posture as 4.4's
    `assert_datasets_schema` — catches a 4.2 schema change before any
    load job runs."""
    fields = {f.name: f for f in schema}
    embed = fields.get("embedding")
    if embed is None or embed.field_type != "FLOAT64" or embed.mode != "REPEATED":
        raise AssertionError(
            "semantic_columns.json must declare embedding ARRAY<FLOAT64>"
        )
    for required in ("package_id", "column_name", "description", "generated_at"):
        f = fields.get(required)
        if f is None or f.mode != "REQUIRED":
            raise AssertionError(
                f"semantic_columns.json must declare {required} REQUIRED"
            )
