# services/ingest

The ingestion job: pulls qualifying resources from configured CKAN sources, writes raw bytes to `gs://maplequery-raw/raw/...`, and appends a per-resource JSONL record to `runlog/<run_id>.jsonl`. A separate follow-up task loads the JSONL into BigQuery's `raw.documents` table.

The canonical GCS object key shape is:

```
raw/country=<cc>/source=<src>/organization=<org>/resource_last_modified=<YYYY-MM-DD>/fmt=<ext>__id=<doc_id12>__<safe_filename>
```

The partition is the resource's own `last_modified` (CKAN), falling back to the dataset's `metadata_modified` — *not* wallclock ingest time. This makes the path content-addressed: re-ingesting an unchanged resource hits the same key, so GCS md5-match dedup fires across days. Path building, slug rules, and the write-time collision contract (HEAD → `if_generation_match=0` → on existing object, compare md5) live in `core/path_builder.py`, `core/slugify.py`, and `clients/gcs.py`.

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

Today: GCS writes + JSONL run log. **No BigQuery.**

Follow-up: a loader reads JSONL files and inserts rows into `raw.documents`; the metadata- and content-hash-based dedup ladder lands at the same time. The JSONL shape matches the eventual table schema so the follow-up is load-only — no re-ingest from CKAN.
