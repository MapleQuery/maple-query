# services/warehouse-load

Reads ingest JSONL runlogs and writes rows into BigQuery's `raw.documents` table via a `MERGE` keyed on `document_id`.

Inputs:
- `services/ingest/runlog/*.jsonl` on local disk (default), or
- `gs://maplequery-raw/runlog/*.jsonl` when `WHLOAD_RUNLOG_GCS_PREFIX` is set.

Output: `raw.documents` in BigQuery, populated against the schema at `infra/terraform/schemas/raw_documents.json` (the same JSON that `google_bigquery_table` consumes — single source of truth shared with Terraform).

## Layering

Imports flow forward only, enforced by `import-linter`:

```
types → config → providers → clients → core → entrypoint
```

`core/` is pure logic. Concrete clients (`google.cloud.bigquery`, `google.cloud.storage`) live in `clients/` behind `Protocol`s (`BqClient`, `GcsClient`) and are passed into `core` by the entrypoint. Tests substitute fakes against those protocols instead of monkeypatching.

## Pipeline

1. **Read** — stream rows from local + GCS runlogs (`core/runlog_reader.py`). Parse errors are surfaced as a `ParseError` event and counted; one bad line does not halt the run because the runlog is immutable history.
2. **Filter** — keep only `file_format == "csv"` AND `ingestion_status == "success"`. Filter runs *before* dedupe so a quarantined row never shadows a real success at the same `source_url`.
3. **Dedupe by `source_url`** — not by `document_id`. CKAN URL-sharing and failed-row placeholder `document_id`s produce within-run `source_url` dupes by design. Winner: latest `ingested_at`; tie-break `document_id` ASC.
4. **Stage + MERGE** — write the kept rows to a TTL-bounded staging table, then `MERGE INTO raw.documents` keyed on `document_id`. Staging table auto-deletes after the configured TTL (default 1h).

## Column ownership in the MERGE

`core/documents_merge.py` splits `raw.documents` columns into two sets:

- **Written by this service** (`DOCUMENTS_OWNED_BY_LOADER`): `country_code`, `source_code`, `organization_code`, `source_url`, `gcs_uri`, `checksum`, `etag`, `http_last_modified`, `resource_last_modified`, `file_format`, `declared_format`, `language`, `title`, `document_type`, `subjects`, `published_date`, `metadata_modified`, `ingested_at`, `ingestion_status`, `quarantine_reason`, `run_id`.
- **Written by the downstream content loader** (`DOCUMENTS_OWNED_BY_CONTENT_LOADER`): `preamble_rows`, `header_confidence`, `load_status`, `load_attempted_at`, `load_error`, `row_count`.

Two invariants the MERGE must satisfy (asserted by an integration test that regexes the generated SQL):

1. The `UPDATE` clause **never** touches the content-loader columns. Once that loader marks a doc `load_status='loaded'`, re-running this service must not reset it to `'pending'`.
2. The `INSERT` clause sets content-loader columns to their initial values (`load_status='pending'`, rest `NULL`) — and only on first insert.

The `UPDATE` triggers when `s.metadata_modified > t.metadata_modified OR s.ingested_at > t.ingested_at`. Otherwise the row is a no-op and BigQuery skips the write.

## Schema source of truth

Schemas are loaded from `infra/terraform/schemas/*.json` via `core/schema_loader.py`. The same JSON is consumed by Terraform's `google_bigquery_table` resource, so schema diffs read as ordinary JSON diffs in PR review and there is no Python-side schema literal that can drift from the HCL. Adding a column means editing the JSON and updating the pydantic model in `types.py` — CI key- and value-drift checks catch the latter.

## Running locally

```sh
cd services/warehouse-load
uv sync --extra dev
```

Set `WHLOAD_GCP_PROJECT_ID` (or `GCP_PROJECT_ID`), then:

```sh
# Dry-run against the local ingest runlog directory.
uv run warehouse-load documents --dry-run

# Real load.
uv run warehouse-load documents

# Ad-hoc reload, filtered to recent runlogs.
uv run warehouse-load documents --since 2026-06-01

# Restrict to a subset of org codes.
uv run warehouse-load documents --limit-orgs fin --limit-orgs statcan
```

Default runlog source is `services/ingest/runlog/*.jsonl` (override with `--runlog-local-dir` or `WHLOAD_RUNLOG_LOCAL_DIR`). To read from GCS, set `WHLOAD_RUNLOG_GCS_PREFIX=gs://maplequery-raw/runlog/` or pass `--runlog-gcs-prefix`.

## Tests

```sh
uv run pytest                  # all
uv run pytest tests/unit       # fast feedback loop
uv run pytest tests/integration # FakeBqClient against the MERGE
uv run pytest tests/e2e        # dry-run against real runlogs
uv run ruff check . && uv run mypy . && uv run lint-imports
```

`uv.lock` is committed; don't regenerate casually.
