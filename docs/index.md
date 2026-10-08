# Docs index

## Architecture
- [Top-level architecture](../ARCHITECTURE.md)

## Services
- [`services/ingest`](services/ingest.md): CKAN ingestion job (GCS + run log).
- [`services/warehouse-load`](services/warehouse-load.md): GCS → BigQuery `raw.documents` / `raw.rows`.
- [`services/semantic-enrich`](services/semantic-enrich.md): dataset/column enrichment, and the agent (pipeline, tools, live sources, routing, evals).
- [`services/curate`](services/curate.md): Normalize (M3), the curated people, terms and linked contributions tables.
- [`services/agent-service`](services/agent-service.md): FastAPI wrapper serving `/chat` and friends on Cloud Run.
- [`web`](services/web.md): Next.js app (chat, evidence rail, notebook, explorer).

## Design
- [People spine](design/people-spine.md): a politician table in `curated` to join contributions, lobbying and expenses on (proposed).

## Policy
- [Reliability](RELIABILITY.md)
- [Security](SECURITY.md)
