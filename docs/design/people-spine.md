# People spine: one table of politicians to join the record on

Status: **proposed** (2026-10-08). Phase 2 of "hold politicians
accountable". Phase 1, live Parliament tools (votes, bills, Hansard),
shipped without new infrastructure.

## Why a table, when Parliament is already live

Phase 1 answers questions about one record at a time: how an MP
voted, what they said. The accountability questions people actually
ask join a *person* across sources:

- who lobbied MP X, who donated to X, and how X then voted;
- grants that went to X's riding while X held it;
- X's office expenses against the median MP's.

Every source spells names its own way ("Hon. Pierre Poilievre",
"POILIEVRE, Pierre", "Pierre Poilievre, M.P."), and ridings are
renamed and redrawn every decade. The join key has to be a stable
person id with known name variants and terms of office. That is a
table, built offline, not something the agent can work out at
question time. This is the "joins across raw rows" exception in
`ARCHITECTURE.md`'s rule for mirroring sources.

## How it fits the warehouse as built

Mapped against the implementation, not the aspirations (file
references are to the repo at the time of writing).

| Convention | As implemented | This design |
|---|---|---|
| Stages hand off through storage, never calls | ingest → GCS → `raw.*` → `semantic.*` | New **Normalize (M3)** stage writes `curated.*`. Reads `raw.*` and the openparliament.ca API, writes nothing upstream |
| Layer for typed, cleaned data | `curated` dataset exists in `infra/terraform/bigquery.tf`, no tables ("Empty in milestone 2") | All new tables land in `curated` |
| Schema as code | `infra/terraform/schemas/*.json`, read by both Terraform and Python, with drift tests | `curated_people.json`, `curated_person_names.json`, `curated_person_terms.json`, `curated_person_links.json` + `bigquery.tftest.hcl` assertions |
| Idempotency | stage → MERGE with explicit column ownership (`warehouse-load`), newer-`generated_at`-wins (`semantic-enrich`) | Deterministic ids; stage → MERGE on natural keys; a re-run with the same inputs is a no-op |
| Quarantine over drop | named reasons, never silent | Ambiguous matches go to `person_links` with `status = 'ambiguous'` and candidates, never auto-linked, never dropped |
| One service account per service, dataset-scoped IAM | `sa-warehouse-load` edits `raw`; `sa-agent-service` reads `raw` + `semantic` only | New `sa-curate`: editor on `curated`, viewer on `raw`. `sa-agent-service` gets viewer on `curated` |
| Layering lint | import-linter `types → config → providers → clients → core → entrypoint` per service | Same, in the new service |
| Partitioning | none; clustering only | Cluster `person_links` by `source_system, person_id`; the rest are small (thousands of rows) |

### The tables

All live in `curated`. Ids are deterministic so re-runs converge.

- **`people`**: one row per person. `person_id` (`op:<openparliament
  slug>`, e.g. `op:pierre-poilievre`, stable since the slug is), display
  name, given and family name, `source_url`, `first_seen_at`,
  `updated_at`, `run_id`. MERGE on `person_id`.
- **`person_names`**: `(person_id, name_norm)` with `name_raw` and
  `origin` (openparliament alternate names, Hansard attribution, a
  confirmed match from a source). `name_norm` = casefolded,
  accent-stripped, honorifics removed. This is what linking matches on.
- **`person_terms`**: `(person_id, role, start_date)` with `end_date`,
  party, riding name and number, province. From openparliament
  memberships. Answers "who held riding R on date D".
- **`person_links`**: `(source_system, record_key, field, person_id)`
  with `match_method` (`exact_name_term`, `riding_term`, `manual`),
  `confidence`, `status` (`linked` | `ambiguous`), `candidates` (for
  ambiguous), `run_id`, `linked_at`. `record_key` is
  `document_id:row_index` for warehouse rows. Cluster
  `source_system, person_id`.

Riding-level joins (grants by `federal_riding_name`) need **no**
row-level links: `person_terms` already answers "who held this riding
then", so the agent joins on riding and date range at question time.
Row-level links are only for sources that name a person (contributions,
lobbying communications, expense reports).

### What the agent needs to read it

Three blockers in today's agent, all small:

1. **SQL allow-list.** `settings.eval_allowed_datasets = ("raw",
   "semantic")` and `sql_guard._dataset_violation` reject any other
   dataset. Add `curated`.
2. **The `raw.rows` rule.** The guard requires a literal `document_id
   IN (...)` on `raw.rows` and refuses joins through it, so
   "`person_links` → `raw.rows`" cannot run. Links therefore carry the
   values a question needs (date, amount, counterpart name) copied
   from the row at link time, rather than pointing back into
   `raw.rows`. Copying costs little; the guard keeps the 200 GB table
   safe.
3. **IAM.** `sa-agent-service` has no grant on `curated`.

Better than raw SQL for the model: one tool, `person_record(person,
since, sources)`, that returns a politician's terms, votes (live),
linked contributions, lobbying contacts and expenses in one call, built
on these tables plus the live Parliament client.

## Inputs, and what each needs

| Source | Where | Gets in via | Status |
|---|---|---|---|
| MPs, terms, name variants | openparliament.ca API (live, JSON) | The new stage calls it directly (`include=all` + memberships) | Ready |
| Election contributions | Elections Canada, `od_cntrbtn_de_e.zip` (106 MB), listed on open.canada.ca (org `elections`, subject `government_and_politics`) | Existing ingest, **plus ZIP support**: the file is a ZIP labelled CSV, so the format sniff quarantines it today | Needs a small ingest change |
| Lobbying registrations + monthly communications | lobbycanada.gc.ca ZIPs (listed on open.canada.ca, org `ocl-cal`) | — | **Blocked**: the files sit behind a Cloudflare bot challenge (HTTP 403 to any script). We do not work around it. Options: a person downloads monthly into `gs://…/raw/` with provenance, or ask the Commissioner's office for an unchallenged URL |
| MP office expenditures | ourcommons.ca proactive-disclosure pages (HTML, quarterly) | New ingest source kind (`api_kind` is `Literal["ckan"]` today) | Needs a scraper source kind |

Ingest constraints that apply (from `services/ingest`): it queries
by `subject` + format + optional `--limit-orgs`, with no
per-package filter, so a run pulls every dataset in that org and
subject. Downloads over 512 MiB are quarantined; warehouse-load caps
documents at 600 MiB / 50M rows. The contributions file fits.

## Linking rules

Deterministic first, and nothing silent:

1. Normalise both sides the same way (`person_names.name_norm`).
2. **Exact name + term overlap**: the record's date falls inside one
   of the person's terms → `linked`, confidence 1.0.
3. **Exact name, no term overlap, or one riding + date** →
   `linked` at 0.9 when exactly one candidate survives.
4. Two or more candidates → `ambiguous`, candidates recorded. The
   agent states the ambiguity instead of choosing.
5. Fuzzy matching (edit distance) only ever proposes; a fuzzy match is
   `ambiguous` until confirmed (`manual`).

Re-running with the same inputs produces the same links: ids are
deterministic, the MERGE key is `(source_system, record_key, field)`, and
each run's `run_id` records when a link last changed.

## Cost

- Storage: `people`/`names`/`terms` are thousands of rows; links run to
  the low millions (contributions). Cents a month.
- Build: one pass over the contribution documents in `raw.rows`, scoped
  by `document_id` (the table's clustering key), so it scans only those
  documents. Re-runs are incremental by `loaded_at`.
- Agent: queries against `curated` are small and stay under the guard's
  existing byte cap.

## Decisions needed

1. **Bring M3 into scope.** `ARCHITECTURE.md` says everything right of
   Extract is "out of scope for now" and `curated` is "TBD". This
   design is the first M3 work; the doc needs updating when it's approved.
2. **New service (`services/curate`) or a subcommand of
   `warehouse-load`.** Recommendation: a new service. It reads an
   external API, which warehouse-load never does, and it owns a
   different dataset and service account.
3. **Lobbying data**: manual monthly upload, or ask the Commissioner.
4. **Which first**: contributions are ready once ingest reads ZIPs.
   Recommendation: people + terms + contributions first, then expenses
   (needs a scraper), then lobbying.

## Drift found while mapping (fix alongside)

- `ARCHITECTURE.md` lists `bq.raw.ingest_watermark`; no such table
  exists. Ingest's only cursor is `--since`.
- No service-local `AGENTS.md` exists, though the root one requires
  them.
- `ingest` and `warehouse-load` have no CI; only agent-service deploys
  are gated.
- `docs/services/ingest.md` still says "No BigQuery".
