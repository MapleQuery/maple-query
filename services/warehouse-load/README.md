# warehouse-load

Reads ingest JSONL runlogs and writes `raw.documents` in BigQuery.

## Setup

```sh
cd services/warehouse-load
uv sync --extra dev
```

Set `WHLOAD_GCP_PROJECT_ID` (or `GCP_PROJECT_ID`) and run:

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

By default the loader reads `services/ingest/runlog/*.jsonl` (override
with `--runlog-local-dir` or `WHLOAD_RUNLOG_LOCAL_DIR`). To read
runlogs from GCS, set `WHLOAD_RUNLOG_GCS_PREFIX=gs://maplequery-raw/runlog/`
or pass `--runlog-gcs-prefix`.

## Tests

```sh
uv run pytest                  # all
uv run pytest tests/unit       # fast feedback loop
uv run pytest tests/e2e        # dry-run against real runlogs
```

See `AGENTS.md` for layering and architectural notes.
