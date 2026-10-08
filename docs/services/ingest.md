# services/ingest

The ingestion job: pulls qualifying resources from configured CKAN sources, writes raw bytes to `gs://maplequery-raw/raw/...`, and appends a per-resource JSONL record to `runlog/<run_id>.jsonl`. A separate follow-up task loads the JSONL into BigQuery's `raw.documents` table.

The canonical GCS object key shape is:

```
raw/country=<cc>/source=<src>/organization=<org>/resource_last_modified=<YYYY-MM-DD>/fmt=<ext>__id=<doc_id12>__<safe_filename>
```

The partition is the resource's own `last_modified` (CKAN), falling back to the dataset's `metadata_created` — *not* wallclock ingest time, and *not* `metadata_modified` (which bumps on any dataset edit and would silently break dedup for resources lacking their own `last_modified`). `metadata_created` is set at dataset creation and never updated, so the partition is stable; re-ingesting an unchanged resource hits the same key, and GCS md5-match dedup fires across days. Path building, slug rules, and the write-time collision contract (HEAD → `if_generation_match=0` → on existing object, compare md5) live in `core/path_builder.py`, `core/slugify.py`, and `clients/gcs.py`.

## Layering

Imports flow forward only, enforced by `import-linter` (config in `services/ingest/pyproject.toml`):

```
types → config → providers → clients → core → entrypoint
```

`core/` is pure logic. Concrete clients (`google.cloud.storage`, `httpx`) live in `clients/` and are passed into `core.pipeline.run(...)` by callers — `core` never constructs them.

## Layout

Subpackages exist as they're implemented; missing modules aren't stubs — they don't exist yet. Walk `src/ingest/` to see the current shape.

## Running locally

```bash
cd services/ingest
uv sync --extra dev
uv run pytest
uv run ruff check
uv run lint-imports

# Dry-run against live CKAN (no GCS writes, no run-log)
INGEST_GCP_PROJECT_ID=<your-project> \
  uv run ingest -s government_and_politics -f csv --limit-orgs fin --dry-run

# Real run
INGEST_GCP_PROJECT_ID=<your-project> \
  uv run ingest -s government_and_politics -f csv --limit-orgs fin
```

`uv.lock` is committed; don't regenerate casually.

## Scope of this service

GCS writes + JSONL run log. BigQuery loading is not done here: the run
log is read by [`services/warehouse-load`](warehouse-load.md), which
writes `raw.documents` and `raw.rows`.

Known limits:

- **Archives land unchanged, on request.** A resource declared in a
  requested format whose bytes are a ZIP (Elections Canada's
  contributions "CSV") is skipped by default. With `--accept-archives`
  it lands as-is with `fmt=zip`: raw stays the source's bytes, and the
  consumer extracts (see [`services/curate`](curate.md)). warehouse-load
  only loads `csv`/`tsv`, so zips never reach `raw.rows`.
- **No per-package selection.** A run takes everything matching
  `subject` + format (+ `--limit-orgs`).
- **Size cap.** Downloads over 512 MiB fail and are quarantined as
  `download_failed`.
- **CKAN only.** `api_kind` is `Literal["ckan"]`; scraped HTML sources
  (e.g. ourcommons.ca expenditure reports) need a new source kind.
