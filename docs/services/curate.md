# services/curate

Normalize (M3): builds the `curated` dataset from automatable sources.
Design and rationale: [people spine](../design/people-spine.md).

| Command | Reads | Writes |
|---|---|---|
| `curate people` | openparliament.ca API (politicians incl. former, memberships, per-person detail) | `curated.people`, `curated.person_names`, `curated.person_terms` |
| `curate contributions` | The Elections Canada contributions archive in `gs://maplequery-raw/raw/…organization=elections/…od_cntrbtn_de_e…zip`, plus `person_names`/`person_terms` | `curated.person_contributions` |
| `curate expenses` | ourcommons.ca members' expenditure reports (one CSV per quarter, read directly, 1 req/s), plus `person_names`/`person_terms` | `curated.person_expenses` |

## Running

```bash
cd services/curate
uv sync --extra dev

# 1. Land the Elections Canada archive (once per refresh; 212 MB ZIP of a 3.8 GB CSV)
cd ../ingest && INGEST_GCP_PROJECT_ID=<project> \
  uv run ingest -s government_and_politics -f csv --limit-orgs elections --accept-archives

# 2. People, then contributions (as sa-curate, or an admin)
cd ../curate
CURATE_GCP_PROJECT_ID=<project> uv run curate people
CURATE_GCP_PROJECT_ID=<project> uv run curate contributions
CURATE_GCP_PROJECT_ID=<project> uv run curate expenses

# Dry runs write JSONL and never touch BigQuery or GCS
uv run curate people --dry-run --out /tmp/curate/people --max-detail 0
uv run curate contributions --dry-run --out /tmp/curate/contrib \
  --source-file od_cntrbtn_de_e.zip --people-from /tmp/curate/people
```

The first `curate people` fetches every politician's detail record
(≈1,300 requests at 2/s, ~11 minutes). Later runs reuse stored terms and
names for former MPs and fetch only new people plus sitting MPs (from
the stable `/politicians/` list).

## Measured on real data (2026-10-08, full dry run)

- **people**: 1,326 politicians, 2,797 name variants, 1,873 Commons
  terms (since 1994). Terms come from each person's detail record:
  openparliament's paged `/politicians/memberships/` list is unstable
  (one read returned 1,873 rows but only 1,239 distinct terms, dropping
  e.g. Marc Garneau's 2015-2023 term), so it is never used.
- **contributions**: 10.85M rows in 32 s, 94 MB of memory; 817K rows to
  candidates or contestants; **464K linked** (16,358 aggregate rows),
  353K unmatched (mostly candidates who never sat), 0 ambiguous.
- **expenses**: 24 quarters (2020-21 Q2 to 2026-27 Q1), 8,891 rows;
  **8,752 linked (99.4%)**, 89 vacant seats, 50 unmatched (former
  members reporting more than a year after leaving), 0 ambiguous.
- Every linked name was checked against its person: the only surname
  differences are real name changes (Rachael Harder / Thomas, Candice
  Hoeppner / Bergen, Michelle Rempel / Rempel Garner, Jessica Fancy /
  Fancy-Landry), carried by openparliament's alternate names.

## Linking rules

1. Names normalised identically on both sides (`core/names.py`).
2. Exact name AND the record's date within one year of one of the
   person's terms -> `linked` (`exact_name_term`). For expenses, a
   quarter links to a term that covers it or ended within a year before
   it (the House reports a former member's wind-down costs).
3. Several namesakes fit: if exactly one sat for the party named on the
   return (or the caucus on the expense report) in the matching term ->
   `linked` (`exact_name_term_party`);
   otherwise `ambiguous` with the candidates, never resolved by hand.
4. No fitting person -> not stored, only counted.

Riding is never consulted. Individual donors are aggregated (count,
totals, province); only organizations (pre-2007 businesses, unions) are
named.

## Writes

Stage (auto-expiring `curated._stage_<table>_<run>`) then MERGE with
snapshot semantics: changed rows update (bookkeeping columns alone never
trigger an update), new rows insert, rows gone from the snapshot delete.
A snapshot under 50% of the stored table aborts (`--allow-shrink` to
override after checking the source).

## Caveat for readers of `person_contributions`

Leadership campaigns file weekly reports *and* a final campaign return;
the weekly reports overlap the return. Never add amounts across
`financial_report` values for one contest. The agent's `person_record`
tool reports per return for this reason.
