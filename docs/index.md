# Docs index

## Architecture
- [Top-level architecture](../ARCHITECTURE.md)

## Services
- [`services/ingest`](services/ingest.md) — CKAN ingestion job (Phase A1: GCS + JSONL).
- [`services/warehouse-load`](services/warehouse-load.md) — runlog -> BigQuery loader (`raw.documents`, `raw.rows`, `raw.column_index`).
- [`services/semantic-enrich`](services/semantic-enrich.md) — semantic layer pipeline; currently ships the vLLM model-serving stack.

## Policy
- [Reliability](RELIABILITY.md)
- [Security](SECURITY.md)
