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

### Live sources (no ingest)

Not every public dataset should be mirrored. Statistics Canada already
serves its ~8,000 official tables through a public API (the Web Data
Service), current to the day, with full metadata. The agent reads those
tables at question time (`clients/statcan.py`, `core/statcan_tools.py`
in `services/semantic-enrich`) instead of copying them into BigQuery:

- **Warehouse** (ingest → BigQuery): open.canada.ca CSVs that have no
  query API: grants, contracts, travel and hospitality, departmental
  program data. Row-level records, queried with SQL.
- **Live** (StatCan WDS): economy- and population-wide statistics: CPI,
  GDP, income, jobs, population components, housing starts, trade by
  country, government finance. Fetched per series, cited to the table.
- **Live** (open.canada.ca CKAN DataStore): the full proactive-disclosure
  tables (grants and contributions, contracts over $10K, travel,
  hospitality), including the multi-GB files the warehouse never held.
  CKAN filters server-side; grouping, sums and amendment de-duplication
  run in `core/opencanada_tools.py` over at most 30,000 matching rows.

- **Live** (openparliament.ca API): the House of Commons record —
  MPs, recorded votes with per-party breakdowns, bills and their status,
  Hansard speeches. A volunteer-run mirror of ourcommons.ca / LEGISinfo,
  so the client identifies itself, caches slow-changing lists and caps
  detail fetches per call.

Rule of thumb for a new source: mirror it only when the publisher has no
query API, or when answering needs joins across its raw rows. Otherwise
read it live.

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
| `bq.curated.*` | Normalize (M3) | TBD. First proposed tables: [people spine](docs/design/people-spine.md). |
