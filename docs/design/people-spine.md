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
- **`ridings`**: see "Riding joins" below.
- **`person_terms`**: `(person_id, role, start_date)` with `end_date`,
  party, riding name and number, province. From openparliament
  memberships. Answers "who held riding R on date D".
- **`person_links`**: `(source_system, record_key, field, person_id)`
  with `match_method` (`exact_name_term`, `riding_term`, `manual`),
  `confidence`, `status` (`linked` | `ambiguous`), `candidates` (for
  ambiguous), `run_id`, `linked_at`. `record_key` is
  `document_id:row_index` for warehouse rows. Cluster
  `source_system, person_id`.

### Riding joins: possible, partial, and easy to get wrong

Checked against live data on 2026-10-08, not assumed:

- **Coverage is about a fifth.** In the 2,000 newest grants and
  contributions rows, 22% carry `federal_riding_name_en` /
  `federal_riding_number`. Grants to individuals (`recipient_type = P`,
  the largest group) never do; organizations mostly do. So "grants to
  X's riding" can only ever mean grants to *organizations* located there,
  and every answer has to say so. The postal code is present on 92% of
  rows, but mapping postal codes to ridings needs StatCan's Postal Code
  Conversion File, which is licensed, not open.
- **Riding numbers are not stable ids.** Ridings were redrawn for the
  2025 election (2023 Representation Order), and numbers were reused. In
  the grants data, `48018` is *Edmonton Riverbend* (2013 order); in
  openparliament.ca, `48018` is *Edmonton Manning* (2023 order), while
  the grants data has Edmonton Manning as `48016`. Joining on the number
  attaches money to the wrong MP, silently.
- **Names differ in punctuation:** `Kanata--Carleton` vs `Kanata—Carleton`.

So the join is **(normalised riding name, representation order inferred
from the agreement date) → riding → term**, never the bare number. That
needs one more small table, **`ridings`**: `(riding_key, order, number,
name_variants, valid_from, valid_to)`, built from Elections Canada's
published electoral-district lists for both the 2013 and 2023 orders
(openparliament's riding ids follow the 2023 order, and it relabels
older memberships onto them, so it cannot supply the 2013 numbering). Grant rows without a riding
stay unlinked, and answers say what share of the money that leaves out.

Row-level person links are only for sources that name a person
(contributions, lobbying communications, expense reports).

### What the agent needs to read it: the SQL guard

Every SQL statement the model writes goes through
`services/semantic-enrich/src/semantic_enrich/core/sql_guard.py` before
BigQuery sees it. The checks, in order, first failure wins:

1. Length 20 B–20 KB; a single statement; no forbidden keyword
   (`INSERT`, `UPDATE`, `DELETE`, `MERGE`, `CREATE`, `DROP`, `ALTER`,
   `GRANT`, `REVOKE`, `TRUNCATE`, `CALL`); the root is a `SELECT`.
2. **Dataset allow-list** (`_dataset_violation`): every table must live
   in a dataset listed in `settings.eval_allowed_datasets`, today
   `("raw", "semantic")`. A query naming `curated.people` is rejected
   with `sql_dataset_not_allowed: curated`. This is the "allow-list".
3. Project: an explicit project id must be ours.
4. **The `raw.rows` rule** (`_document_id_filter_violation`): any query
   that touches `raw.rows` must contain a literal
   `document_id IN ('…','…')`. A subquery IN or a JOIN to find the ids
   is rejected (`sql_no_document_id_filter`).
5. Dry run: BigQuery estimates bytes; over the cap (50 GB) is rejected.
6. A `LIMIT 100` wrapper is added if missing.

**Why `raw.rows` matters at all.** It is where the warehouse's data
actually is: every row of every ingested CSV, ~200 GB, one JSON object
per row. `raw.documents` and `semantic.*` are catalogs *about* that data;
any number the agent reports from the warehouse comes out of `raw.rows`.
Contributions, once ingested, would land there too.

**Why the rule exists.** `raw.rows` is clustered by `document_id`, and
BigQuery can only skip the clusters it doesn't need when the ids are
literals it can read at planning time. A JOIN or subquery that *finds*
the ids makes it scan the whole table: about 200 GB, ~$1.25 a query at
on-demand prices, every time. The rule forces the agent to pick
specific documents first (`list_documents`), then query only those.

**What that means for people links.** The agent will never be able to
write "join `person_links` to `raw.rows`": the ids would come from a
join, which the rule forbids. So the link builder (offline, not the
agent) copies the values a question needs from the row into
`person_links` (date, amount, counterpart name, document title) when it
creates the link. The agent then reads only `curated`, which is small.
`raw.rows` stays protected and is read only by the offline stage, scoped
by `document_id` like everything else.

Changes needed, all small:

1. Add `"curated"` to `eval_allowed_datasets` (one setting) and to the
   table-name normaliser (`core/sql_normalize.py` only rewrites
   `raw.rows` today).
2. `roles/bigquery.dataViewer` on `curated` for `sa-agent-service`
   (Terraform, `agent_service.tf`).
3. Better still, a `person_record` tool so the model rarely writes SQL
   over these tables at all.

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

## Decisions

1. **M3 is not built, so this starts it.** The old `ARCHITECTURE.md`
   said "only Ingest is in scope", which was stale: Extract, Enrich and
   the Agent are all built. Normalize (M3) is the one stage that is
   genuinely missing: `bq.curated` exists in Terraform but has no tables
   and nothing writes to it. "In scope" means building it, starting with
   this. `ARCHITECTURE.md` now shows the stages as they are.
2. **New service: `services/curate`** (decided 2026-10-08). It calls an
   external API, owns a different dataset and gets its own service
   account; warehouse-load does none of those.
3. **Lobbying data**: still open. This is the federal Registry of
   Lobbyists, run by the Office of the Commissioner of Lobbying:
   *registrations* (who is paid to lobby which institutions, on what
   subjects) and *monthly communication reports* (every arranged
   communication between a lobbyist and a designated public office
   holder (an MP, minister or senior official), with the date and
   subject). It is what makes "who lobbied MP X before the vote"
   answerable. The files are published, but lobbycanada.gc.ca serves
   them behind a Cloudflare bot challenge (every scripted request gets a
   403), and we do not work around that. Options: a person downloads
   them monthly into `gs://…/raw/` with provenance, or we ask the
   Commissioner's office for an unchallenged download.
4. **ZIP reading has to be built.** Nothing in ingest or warehouse-load
   opens an archive today: a "CSV" that is really a ZIP is sniffed as
   `zip` and never lands as CSV, and warehouse-load only parses CSV/TSV.
   Both contribution and lobbying files are ZIPs. Proposed: ingest
   extracts CSV members at landing (one GCS object per member, with the
   archive's `document_id` and member name recorded), so everything
   downstream stays CSV-only.
5. **Order**: people + terms + ridings first (no ingest needed, the
   openparliament API is enough), then ZIP extraction + contributions,
   then expenses (needs a scraper), then lobbying once access is sorted.

## Drift found while mapping (fix alongside)

- ~~`ARCHITECTURE.md` lists `bq.raw.ingest_watermark`~~ (removed; no
  such table existed. Ingest's only cursor is `--since`.)
- ~~`ARCHITECTURE.md` says only Ingest is in scope~~ (updated to the
  stages as built).
- ~~`docs/services/ingest.md` says "No BigQuery"~~ (updated: loading is
  warehouse-load's job; known limits listed).
- No service-local `AGENTS.md` exists, though the root one requires
  them.
- `ingest` and `warehouse-load` have no CI; only agent-service deploys
  are gated.
