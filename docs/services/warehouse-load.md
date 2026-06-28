# services/warehouse-load

Three commands share this package:

- **`warehouse-load documents`** — reads ingest JSONL runlogs and writes rows into `raw.documents` via a MERGE keyed on `document_id`. The catalog loader.
- **`warehouse-load rows`** — reads candidate docs from `raw.documents`, streams each CSV body from GCS, detects the header, and writes per-row JSON into `raw.rows` via a batch MERGE on `document_id`. The body loader. Refreshes `raw.column_index` after each run by default.
- **`warehouse-load column-index`** — standalone refresh of `raw.column_index` from `raw.rows`. Idempotent (`CREATE OR REPLACE TABLE`).

Inputs:
- `services/ingest/runlog/*.jsonl` on local disk (default), or `gs://maplequery-raw/runlog/*.jsonl` when `WHLOAD_RUNLOG_GCS_PREFIX` is set (documents loader).
- `raw.documents` rows where `file_format IN ('csv','tsv')`, `ingestion_status='success'`, and `load_status=<filter>` (rows loader).
- CSV bytes at `raw.documents.gcs_uri` (rows loader; default bucket `gs://maplequery-raw/raw/`).

Outputs:
- `raw.documents` — per-file catalog. Schema at `infra/terraform/schemas/raw_documents.json` (the same JSON that `google_bigquery_table` consumes — single source of truth shared with Terraform).
- `raw.rows` — one BQ row per CSV body row. The `row` column is a BQ `JSON` value whose keys are the normalised header names.
- `raw.column_index` — materialised `(col_name, file_count, document_ids)` index over `raw.rows`.

## Layering

Imports flow forward only, enforced by `import-linter`:

```
types → config → providers → clients → core → entrypoint
```

`core/` is pure logic. Concrete clients (`google.cloud.bigquery`, `google.cloud.storage`) live in `clients/` behind `Protocol`s (`BqClient`, `GcsClient`) and are passed into `core` by the entrypoint. Tests substitute fakes against those protocols instead of monkeypatching.

## Documents pipeline (`warehouse-load documents`)

1. **Read** — stream rows from local + GCS runlogs (`core/runlog_reader.py`). Parse errors are surfaced as a `ParseError` event and counted; one bad line does not halt the run because the runlog is immutable history.
2. **Filter** — keep only `file_format == "csv"` AND `ingestion_status == "success"`. Filter runs *before* dedupe so a quarantined row never shadows a real success at the same `source_url`.
3. **Dedupe by `source_url`** — not by `document_id`. CKAN URL-sharing and failed-row placeholder `document_id`s produce within-run `source_url` dupes by design. Winner: latest `ingested_at`; tie-break `document_id` ASC.
4. **Bucket-existence intersection** — list `WHLOAD_BUCKET_PREFIX` once per run (default `gs://maplequery-raw/raw/`), then drop any deduped row whose `gcs_uri` is absent from the bucket-truth set with reason `blob_missing`. Makes a bucket clean self-healing on the next load. Pass `--no-bucket-check` to opt out (logs a loud warning); a missing/unreachable bucket in a real run fails the run rather than silently polluting the warehouse.
5. **Mass-blob-missing guardrail** — if at least 100 rows AND at least 50% of the deduped set would drop as `blob_missing`, the run aborts before the MERGE. Before aborting, the guardrail samples three of the "missing" URIs and HEADs them on the bucket — if any actually exist, the error names URI-format drift as the likely cause instead of a real bucket clean. Override with `--allow-mass-blob-missing` for an intentional full-clean reload.
6. **Stage + MERGE** — write the kept rows to a TTL-bounded staging table, then `MERGE INTO raw.documents` keyed on `document_id`. An empty kept set short-circuits (no staging table, no MERGE). Staging table auto-deletes after the configured TTL (default 1h).

The loader assumes a quiescent corpus: do not run while `ingest` is actively writing, since a runlog row written between the runlog read and the bucket listing could resolve to a blob that isn't yet visible to the listing and would be misclassified as a zombie.

## Column ownership in the MERGE

`core/documents_merge.py` splits `raw.documents` columns into two sets:

- **Written by this service** (`DOCUMENTS_OWNED_BY_LOADER`): `country_code`, `source_code`, `organization_code`, `source_url`, `gcs_uri`, `checksum`, `etag`, `http_last_modified`, `resource_last_modified`, `file_format`, `declared_format`, `language`, `title`, `document_type`, `subjects`, `published_date`, `metadata_modified`, `ingested_at`, `ingestion_status`, `quarantine_reason`, `run_id`.
- **Written by the downstream content loader** (`DOCUMENTS_OWNED_BY_CONTENT_LOADER`): `preamble_rows`, `header_confidence`, `load_status`, `load_attempted_at`, `load_error`, `row_count`.

Two invariants the MERGE must satisfy (asserted by an integration test that regexes the generated SQL):

1. The `UPDATE` clause **never** touches the content-loader columns. Once that loader marks a doc `load_status='loaded'`, re-running this service must not reset it to `'pending'`.
2. The `INSERT` clause sets content-loader columns to their initial values (`load_status='pending'`, rest `NULL`) — and only on first insert.

The `UPDATE` triggers when `s.metadata_modified > t.metadata_modified OR s.ingested_at > t.ingested_at`. Otherwise the row is a no-op and BigQuery skips the write.

One pair of content-loader columns IS written by the UPDATE: `load_status='pending'` and `load_error=NULL` are reset whenever the catalog UPDATE fires (an ingest re-pull moved the source). That's how a re-ingested doc re-enters the rows-loader candidate queue without a separate watermark column. The set is captured in `DOCUMENTS_RESET_ON_REFRESH` and the integration test pins both that the reset columns are present in the UPDATE AND that no other content-loader column is.

## Rows pipeline (`warehouse-load rows`)

1. **Staging precondition** — `raw.rows_staging` must be empty. If not, the runner emits `staging_precondition_violated` and `sys.exit(2)` (distinct exit code so wrapper scripts can branch on it). This is the single-runner-at-a-time guard; concurrent runners would race on the shared staging table and the MERGE would land an arbitrary mixture.
2. **Candidate query** — `SELECT ... FROM raw.documents WHERE file_format IN ('csv','tsv') AND ingestion_status='success' AND load_status=@status` with optional `limit_orgs` / `limit_documents` parameter arrays. Default `--status pending`; `--force` is required to replay `--status loaded`.
3. **Per-doc work** (parallel, `rows_concurrency=4` by default):
   - Mark in-flight: `UPDATE raw.documents SET load_status='pending', load_attempted_at=NOW(), load_error=NULL`.
   - Download the GCS blob into a temp file, byte-counting against `max_bytes_per_doc=600 MB`.
   - Sniff delimiter + encoding from the first 8 KiB (`core/csv_sniff.py`). Disagreement between sniffed delimiter and declared `file_format` is logged via `csv_delimiter_disagrees_with_declared` but the catalog row is NOT rewritten — `file_format` is 3.2-owned.
   - If the sniffed encoding isn't utf-8, decode + rewrite the blob as utf-8 (polars only natively reads utf-8).
   - Detect the header (`core/header_detect.py`). 25-row lookahead, modal-column-count + stability + type-mix signal. Returns `single` / `multi_row` / `low` confidence.
   - Stream the body via `pl.scan_csv(...).collect_batches()` with an explicit `schema` pinning the column count. Each cell becomes a JSON string (empty → `null`, NUL bytes stripped).
   - Append a per-doc JSONL temp file to `raw.rows_staging` (`WRITE_APPEND`).
4. **Batch flush** — when a batch of docs completes AND staging row count crosses `rows_staging_flush_threshold=500_000`, OR at the end of the run, issue a multi-statement script: `MERGE WHEN MATCHED THEN DELETE` + `INSERT INTO ... SELECT FROM staging` + `TRUNCATE TABLE staging`. The DELETE-then-INSERT pattern makes per-doc replay correct.
5. **Per-doc UPDATE** — after the MERGE lands, `UPDATE raw.documents SET preamble_rows=..., header_confidence=..., load_status='loaded', row_count=..., load_error=NULL` for each doc in the batch. Failures (blob_missing, parse_failed) record their outcome immediately within the per-doc worker since they don't append to staging.
6. **Column-index refresh** — `CREATE OR REPLACE TABLE raw.column_index AS SELECT ... FROM raw.rows, UNNEST(JSON_KEYS(row))`. Atomic at the BQ level; readers either see the prior table or the new one. Skip via `--no-refresh-column-index`.
7. **Post-run invariant** — `docs_loaded + docs_blob_missing + docs_parse_failed + docs_skipped_already_loaded == candidate_count`. A mismatch raises `RuntimeError` — that's a worker that raised an uncaught exception the orchestrator didn't map to a `load_status`, which would otherwise silently leak as "fewer docs loaded than expected."

Encoding fallback: when polars raises `UnicodeDecodeError` mid-stream under the sniffed encoding, the worker retries the doc once with `latin-1` (always decodes). Mojibake may survive into `raw.rows` for the latin-1-mismatched cells, but the doc loads — the curated layer can re-decode. Failures other than encoding (truncated bytes, malformed quoting) still mark the doc `parse_failed`.

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

Rows loader:

```sh
# Dry-run: full sniff+detect+stream pipeline, no BQ writes. Useful for plumbing.
uv run warehouse-load rows --limit-orgs fin --dry-run

# Real load: fin org only. Refreshes raw.column_index by default at end of run.
uv run warehouse-load rows --limit-orgs fin

# Retry only previously failed docs.
uv run warehouse-load rows --status parse_failed

# Replay a specific already-loaded doc end-to-end.
uv run warehouse-load rows --limit-documents <doc_id> --status loaded --force

# Standalone column-index refresh (no rows reload).
uv run warehouse-load column-index
```

Concurrency knobs (env-tunable, no code change): `WHLOAD_ROWS_CONCURRENCY` (parallel docs per batch, default 4), `WHLOAD_MAX_BYTES_PER_DOC` (default 600 MiB), `WHLOAD_MAX_ROWS_PER_DOC` (default 50M), `WHLOAD_PER_DOC_TIMEOUT_SECONDS` (default 900 s), `WHLOAD_ROWS_STAGING_FLUSH_THRESHOLD` (default 500K rows). Header thresholds: `WHLOAD_BODY_MIN_RUN`, `WHLOAD_HEADER_LOOKBACK`, `WHLOAD_BODY_MODAL_MATCH_RATIO`, `WHLOAD_HEADER_MAX_CELL_CHARS`.

Default runlog source is `services/ingest/runlog/*.jsonl` (override with `--runlog-local-dir` or `WHLOAD_RUNLOG_LOCAL_DIR`). To read from GCS, set `WHLOAD_RUNLOG_GCS_PREFIX=gs://maplequery-raw/runlog/` or pass `--runlog-gcs-prefix`.

The bucket-existence step uses `WHLOAD_BUCKET_PREFIX` (default `gs://maplequery-raw/raw/`). In a real (non-dry-run) load the bucket must be reachable; pass `--no-bucket-check` to opt out for debugging. Dry-runs without a configured bucket simply skip the step (and report `bucket_check_skipped=true`).

## Tests

```sh
uv run pytest                  # all
uv run pytest tests/unit       # fast feedback loop
uv run pytest tests/integration # FakeBqClient against the MERGE
uv run pytest tests/e2e        # dry-run against real runlogs
uv run ruff check . && uv run mypy . && uv run lint-imports
```

`uv.lock` is committed; don't regenerate casually.
