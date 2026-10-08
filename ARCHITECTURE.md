# Architecture

MapleQuery's job is to take open government data, land it durably in
GCS, catalog it in BigQuery, then make it queryable via an LLM agent.
The pipeline is **staged**: each stage owns one transformation and
hands off through versioned storage (GCS prefixes or BigQuery tables),
never through direct calls.

---

## Stages

```
            ┌─────────┐   ┌────────────────┐   ┌────────────────┐   ┌───────────┐
 CKAN ────▶ │ Ingest  │──▶│ Extract        │──▶│ Enrich         │──▶│  Agent    │──▶ answers
(open.      │ (M1)    │   │ warehouse-load │   │ semantic-enrich│   │ (M4)      │
 canada.ca) └────┬────┘   │ (M2)           │   │                │   │ semantic- │
                 ▼        └───────┬────────┘   └───────┬────────┘   │ enrich +  │
       gs://maplequery-raw        ▼                    ▼            │ agent-    │
                           bq.raw.documents     bq.semantic.datasets│ service   │
                           bq.raw.rows          bq.semantic.columns └─────┬─────┘
                           bq.raw.column_index  (summaries+embeddings)    │ live APIs
                                                                          ▼
                                         StatCan WDS · open.canada.ca DataStore · openparliament.ca

                   ┌──────────────┐
  openparliament ─▶│ Normalize    │ ──▶ bq.curated.people / person_names /
  raw/ EC archive ▶│ (M3) curate  │     person_terms / person_contributions
                   └──────────────┘
```

| Stage | Service | Status |
| -- | -- | -- |
| Ingest (M1) | `services/ingest` | Built. CKAN `package_search` by subject/format/org → GCS + run log. CSV only in practice: there is no archive (ZIP) extraction. |
| Extract (M2) | `services/warehouse-load` | Built. Run log → `raw.documents` (MERGE on `document_id`); CSV bodies → `raw.rows` (one JSON row per CSV row). |
| Enrich | `services/semantic-enrich` | Built. Per-dataset and per-column descriptions + embeddings → `semantic.*`, used for dataset search. |
| Agent (M4) | `services/semantic-enrich` (loop, tools, prompts) served by `services/agent-service` | Built and deployed (Cloud Run, on push to `main`). Reads the warehouse with guarded SQL and three live APIs. |
| Normalize (M3) | `services/curate` | Built: `curate people` (openparliament.ca → `curated.people`, `person_names`, `person_terms`) and `curate contributions` (Elections Canada archive → `curated.person_contributions`). Run by an operator. Design: [people spine](docs/design/people-spine.md). |

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
| `bq.raw.documents` | Extract (warehouse-load) | One row per ingested file, keyed on `document_id` = sha256(source_url ‖ checksum). |
| `bq.raw.rows` | Extract (warehouse-load) | One row per CSV body row: `(document_id, row_index, row JSON)`, clustered by `document_id`, ~200 GB. Header names become JSON keys; every value is a string. |
| `bq.raw.column_index` | Extract (warehouse-load) | `(col_name, file_count, document_ids)` over `raw.rows`. |
| `bq.semantic.datasets` / `bq.semantic.columns` | Enrich (semantic-enrich) | Descriptions + 1536-dim embeddings per package / column; vector search for the agent. |
| `bq.curated.*` | Normalize (M3, `services/curate`) | `people`, `person_names`, `person_terms`, `person_contributions`; schemas `infra/terraform/schemas/curated_*.json`; snapshot-MERGE per table. See [people spine](docs/design/people-spine.md). |
