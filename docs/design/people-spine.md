# People spine: one table of politicians to join the record on

Status: **people + contributions built** (2026-10-08) in
[`services/curate`](../services/curate.md); MP expenses not started.
Phase 2 of "hold politicians accountable". Phase 1, live Parliament
tools (votes, bills, Hansard), shipped without new infrastructure.

What changed from the proposal while building it (each below):
per-source link tables instead of one generic `person_links`; no
"exact name outside any term" link (namesake risk); party breaks ties
between namesakes; ingest lands the ZIP unchanged instead of
extracting; the agent reads `curated` through a tool, not free SQL.

## Why a table, when Parliament is already live

Phase 1 answers questions about one record at a time: how an MP
voted, what they said. The accountability questions people actually
ask join a *person* across sources:

- who donated to X, and how X then voted;
- (context only) grants to organizations located in X's riding while
  X held it;
- X's office expenses against the median MP's.

Every source spells names its own way ("Hon. Pierre Poilievre",
"POILIEVRE, Pierre", "Pierre Poilievre, M.P."), and ridings are
renamed and redrawn every decade. The join key has to be a stable
person id with known name variants and terms of office. That is a
table, built offline, not something the agent can work out at
question time. This is the "joins across raw rows" exception in
`ARCHITECTURE.md`'s rule for mirroring sources.

**Sources must be fully automatable.** Every input is fetched and
refreshed by code, end to end. Anything that needs a person to download
a file, pass a bot challenge, request access, or buy a licence is out of
scope, however useful the data. The same goes for linking: nothing
waits on a human to confirm a match.

## How it fits the warehouse as built

Mapped against the implementation, not the aspirations (file
references are to the repo at the time of writing).

| Convention | As implemented | This design |
|---|---|---|
| Stages hand off through storage, never calls | ingest → GCS → `raw.*` → `semantic.*` | New **Normalize (M3)** stage writes `curated.*`. Reads `raw.*` and the openparliament.ca API, writes nothing upstream |
| Layer for typed, cleaned data | `curated` dataset exists in `infra/terraform/bigquery.tf`, no tables ("Empty in milestone 2") | All new tables land in `curated` |
| Schema as code | `infra/terraform/schemas/*.json`, read by both Terraform and Python, with drift tests | `curated_people.json`, `curated_person_names.json`, `curated_person_terms.json`, `curated_person_contributions.json` + `tests/curate.tftest.hcl` |
| Idempotency | stage → MERGE with explicit column ownership (`warehouse-load`), newer-`generated_at`-wins (`semantic-enrich`) | Deterministic ids; stage → MERGE on natural keys; a re-run with the same inputs is a no-op |
| Quarantine over drop | named reasons, never silent | Ambiguous matches are stored with `status = 'ambiguous'` and their candidates, never auto-linked; unmatched rows are counted in the run summary |
| One service account per service, dataset-scoped IAM | `sa-warehouse-load` edits `raw`; `sa-agent-service` reads `raw` + `semantic` only | New `sa-curate`: editor on `curated`, viewer on `raw`. `sa-agent-service` gets viewer on `curated` |
| Layering lint | import-linter `types → config → providers → clients → core → entrypoint` per service | Same, in the new service |
| Partitioning | none; clustering only | `person_contributions` clusters on `person_id, status`; the rest are small (thousands of rows) |

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
  party, riding name (as published for that term; context only, see
  below), province. From openparliament memberships.
- **One linked table per source**, not a generic link table: each
  source's questions need its own columns. Built so far:
  **`person_contributions`**, Elections Canada contributions to
  candidates and leadership or nomination contestants, aggregated per
  (recipient, return, contributor type, province, and organization name
  for non-individual contributors), with `person_id`, `status`
  (`linked` | `ambiguous`), `candidates`, `match_method`, count and
  totals. Individual donors are never named. Key `contribution_key`.
  Next would be `person_expenses` (MP office expenditures).

### Riding is context, never the join

Links come **only from records that name the person**: contributions to
a candidate, the MP's own expense reports, ministers' travel and hospitality claims, and (live) their
votes and Hansard. Every link is to a record *about that person*.

Riding is an attribute of a term (`person_terms.riding_name`), not a key.
Where another record happens to carry a riding, the agent may add it as
a labelled extra, e.g. "N grants to organizations located in this
riding during their term", worked out at question time from the riding
name, and never counted as the MP's money. A grant is a department's
decision; where it landed is context, not accountability.

Why not a key (checked against live data 2026-10-08):

- **Coverage is about a fifth.** 22% of the 2,000 newest grants and
  contributions rows carry `federal_riding_name_en` /
  `federal_riding_number`; grants to individuals (`recipient_type = P`)
  never do. Postal codes are on 92% of rows, but postal-code-to-riding
  needs StatCan's licensed Postal Code Conversion File, which is out
  of scope (not automatable without a licence).
- **Numbers were reused across electoral maps.** In the grants data
  `48018` is *Edmonton Riverbend* (2013 Representation Order); in
  openparliament.ca it is *Edmonton Manning* (2023 order), which the
  grants data numbers `48016`. A numeric join pins money on the wrong
  MP, silently.
- **Names differ in punctuation:** `Kanata--Carleton` vs `Kanata—Carleton`.

So the riding extra matches on the normalised name *and* the electoral
map in force on the record's date, and when either is ambiguous it is
**omitted, not guessed**. Nothing else depends on it. A `ridings` table
(both orders, from Elections Canada's electoral-district lists) is only
worth building if the extra proves useful.

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
write "join a link table to `raw.rows`": the ids would come from a
join, which the rule forbids. So the link builder (offline, not the
agent) copies the values a question needs from the row into
the link table (date, amount, counterpart, return) when it
creates the link. The agent then reads only `curated`, which is small.
`raw.rows` stays protected and is read only by the offline stage, scoped
by `document_id` like everything else.

As built: the allow-list is **not** widened. The model never writes
SQL against `curated`; the `person_record` tool runs fixed, parameterised
queries over these small tables and returns a politician's terms and
contribution totals per return (never summed across a leadership
campaign's weekly reports and its final return, which overlap). That
keeps the guard's surface unchanged. `sa-agent-service` gets read-only
`curated` access in Terraform (`curate.tf`), and the tool ships behind
`agent_people_enabled` until the tables exist.

## Inputs, and what each needs

| Source | Where | Gets in via | Status |
|---|---|---|---|
| MPs, terms, name variants | openparliament.ca API (live, JSON) | `curate people` (`include=all`, memberships, per-person detail) | Built |
| Election contributions | Elections Canada, `od_cntrbtn_de_e.zip` (212 MB ZIP of one 3.8 GB CSV, 10.85M rows), listed on open.canada.ca (org `elections`, subject `government_and_politics`) | `ingest --accept-archives` lands the ZIP unchanged; `curate contributions` streams the CSV out of it | Built |
| MP office expenditures | ourcommons.ca proactive-disclosure pages (HTML, quarterly) | New ingest source kind (`api_kind` is `Literal["ckan"]` today) | Needs a scraper source kind |

Ingest constraints that apply (from `services/ingest`): it queries
by `subject` + format + optional `--limit-orgs`, with no
per-package filter, so a run pulls every dataset in that org and
subject. Downloads over 512 MiB are quarantined; warehouse-load caps
documents at 600 MiB / 50M rows. The contributions file fits.

## Linking rules

Deterministic first, and nothing silent:

1. Normalise both sides the same way (`person_names.name_norm`).
2. **Exact name + term window**: the record's date is within one year
   of one of the person's terms (a candidate files around the election
   that starts or ends a term) → `linked` (`exact_name_term`).
3. **Several namesakes fit**: if exactly one of them sat for the party
   named on the record in the matching term → `linked`
   (`exact_name_term_party`); otherwise `ambiguous`, candidates
   recorded, nobody chosen.
4. **No fitting person** → not linked, not stored, counted. Dropped
   from the proposal: linking an exact name *outside* every term. On
   real data that is a namesake trap (a "John Smith" who lost in 2004 is
   not the John Smith elected in 2015).
5. Fuzzy matching (edit distance) never links. A fuzzy-only match is
   recorded as `ambiguous` with its candidates and stays that way; no
   human review step.

Re-running with the same inputs produces the same links: ids are
deterministic, each table is written as a whole snapshot (stage, then
MERGE with delete, guarded against shrinking past 50%), and `run_id`
records which run last changed a row.

Measured on the real file (dry run): 10.85M rows in 32 s; 410K of 817K
person rows linked to 791 people; zero surname mismatches; all 101
namesake rows settled by party.

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
3. **ZIPs land unchanged** (built). The contributions "CSV" is a ZIP
   whose single member is 3.8 GB, past warehouse-load's 600 MB cap.
   Rather than extract at ingest, `ingest --accept-archives` lands the
   ZIP as published (raw stays immutable source bytes) and
   `curate contributions` streams the member straight from GCS. The
   rows never enter `raw.rows`.
4. **Order**: people + terms (built), then contributions (built), then
   MP expenses (needs a scraper source kind; not started).
   Riding context comes free from `person_terms`; no ridings table
   unless it earns one.

## Out of scope: not automatable

Ruled out by the automatable-sources rule (2026-10-08), with the reason,
so they are not re-proposed:

- **Registry of Lobbyists** (Office of the Commissioner of Lobbying):
  registrations and monthly communication reports, the data behind "who
  lobbied MP X". The files on lobbycanada.gc.ca sit behind a Cloudflare
  bot challenge; every scripted request gets HTTP 403. Getting them
  would mean a person downloading by hand or asking for access.
- **Postal code → riding** (StatCan Postal Code Conversion File):
  licensed.
- **Human-confirmed name matches**: ambiguous links stay ambiguous.

Revisit only if a source starts serving the data to scripts (e.g. a
plain download URL or an API).

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
