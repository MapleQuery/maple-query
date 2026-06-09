# `services/warehouse-load/` — agent notes

Loader that reads ingest JSONL runlogs (from `services/ingest/runlog/`
or `gs://maplequery-raw/runlog/`) and writes per-file rows into
BigQuery's `raw.documents` table via a MERGE-on-`document_id` pattern.

Spec: `docs/product-specs/milestone-2/3.2-documents-loader.md` (PRD,
gitignored). Schemas: `infra/terraform/schemas/raw_documents.json`
(single source of truth shared with Terraform).

## Layering

`entrypoint → core → clients → providers → config → types`

Enforced by `import-linter`. `core` accepts client instances by
parameter — it does not import `google.cloud.*` directly.

## Common commands

- `uv sync --extra dev` — install + dev tools.
- `uv run warehouse-load documents --dry-run` — smoke against the local
  runlog dir.
- `uv run pytest` — unit + integration + e2e.
- `uv run ruff check .` / `uv run mypy .` / `uv run lint-imports`.

## Things to know

- The MERGE in `core/documents_merge.py` **must not touch** 3.3-owned
  columns (`preamble_rows`, `header_confidence`, `load_status`,
  `load_attempted_at`, `load_error`, `row_count`). This is asserted in
  the integration test by regex over the UPDATE clause.
- Filter (§6.1) runs **before** dedupe (§6.2). Reversing the order
  lets a quarantined-csv row shadow a real success row at the same
  `source_url`. See the `bq_loader_format_filter` memory.
- Dedupe key is `source_url`, not `document_id`. CKAN URL-sharing
  and failed-row placeholder `document_id`s produce within-run
  `source_url` dupes by design. See the `runlog_failed_rows` memory.
- Schema is loaded from `infra/terraform/schemas/raw_documents.json`
  via `core/schema_loader.py`. Adding a column means editing that
  file and updating the model in `types.py`. The §11.2 / §11.3 CI
  checks catch drift in the other direction.
