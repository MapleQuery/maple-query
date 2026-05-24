# Architecture

MapleQuery's job is to take open government data, land it durably in
GCS, catalog it in BigQuery, then make it queryable via an LLM agent.
The pipeline is **staged**: each stage owns one transformation and
hands off through versioned storage (GCS prefixes or BigQuery tables),
never through direct calls.

---

## Stages

```
       ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
sources│  Ingest  │ →  │ Extract  │ →  │Normalize │ →  │  Agent   │
       │  (M1)    │    │  (M2)    │    │  (M3)    │    │  (M4)    │
       └──────────┘    └──────────┘    └──────────┘    └──────────┘
            │               │                │                │
            ▼               ▼                ▼                ▼
        gs://maplequery-raw     bq.raw.*  →   bq.curated.*   answers
```

Only the Ingest stage is in scope today. Everything to the right of
Extract is out of scope for now.

---

## Storage layers (the contracts between stages)

The storage layer is the public interface between stages. A stage may
be reimplemented entirely as long as its outputs in this layer remain
backward-compatible. Contracts for future stages are listed because
they constrain how earlier stages must shape their output.

| Layer | Owner | Contract |
| -- | -- | -- |
| `gs://maplequery-raw/raw/...` | Ingest | Immutable raw source bytes. Path scheme is the public contract. |
| `gs://maplequery-raw/quarantine/...` | Ingest | Files that failed safety checks; 30-day TTL. |
| `gs://maplequery-raw/sandbox/...` | (any) | Ad-hoc experiments; 7-day TTL. Never read by production code. |
| `bq.raw.documents` | Ingest | One row per ingested file. |
| `bq.raw.ingest_watermark` | Ingest | Per-org incremental cursor. |
| `bq.curated.*` | Normalize (M3) | TBD. |
