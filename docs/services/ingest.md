# services/ingest

The ingestion job: pulls qualifying resources from configured CKAN sources, writes raw bytes to `gs://maplequery-raw/raw/...`, and appends a per-resource JSONL record to `runlog/<run_id>.jsonl` (the Phase A2 BQ-catalog task loads this into `raw.documents`).

**Spec:** [`docs/product-specs/milestone-1/2.2-ingestion-pipeline.md`](../product-specs/milestone-1/2.2-ingestion-pipeline.md). Read it before changing anything in `services/ingest/`.

The storage-layer contracts this service writes against (path grammar, slugify rules, write-time collision check) live in [`2.1-gcs-storage-layer.md`](../product-specs/milestone-1/2.1-gcs-storage-layer.md).

## Layering

Imports flow forward only, enforced by `import-linter` (config in `services/ingest/pyproject.toml`):

```
types → config → providers → clients → core → entrypoint
```

`core/` is pure logic. Concrete clients (`google.cloud.storage`, `httpx`) live in `clients/` and are passed into `core.pipeline.run(...)` by callers — `core` never constructs them.

## Layout

See spec §3 for the canonical tree. Subpackages exist as they're implemented; missing modules aren't stubs — they don't exist yet.

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

## Phase A1 vs A2

Phase A1 (current): GCS writes + JSONL run log. **No BigQuery.**
Phase A2 (future task): load JSONL → `raw.documents`, wire Layer 1 / Layer 3 dedup against BQ.

The JSONL shape matches the eventual `raw.documents` schema so A2 is a load job, not a re-ingest.
