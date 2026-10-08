# services/curate

Normalize (M3): builds the `curated` dataset from automatable sources.
Design and rationale: [people spine](../design/people-spine.md).

| Command | Reads | Writes |
|---|---|---|
| `curate people` | openparliament.ca API (politicians incl. former, memberships, per-person detail) | `curated.people`, `curated.person_names`, `curated.person_terms` |
| `curate contributions` | The Elections Canada contributions archive in `gs://maplequery-raw/raw/…organization=elections/…od_cntrbtn_de_e…zip`, plus `person_names`/`person_terms` | `curated.person_contributions` |

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

# Dry runs write JSONL and never touch BigQuery or GCS
uv run curate people --dry-run --out /tmp/curate/people --max-detail 0
uv run curate contributions --dry-run --out /tmp/curate/contrib \
  --source-file od_cntrbtn_de_e.zip --people-from /tmp/curate/people
```

The first `curate people` fetches every politician's detail record
(≈1,300 requests at 2/s, ~11 minutes). Later runs fetch only people not
yet stored plus sitting MPs.

## Measured on real data (2026-10-08, dry run)

- people: 1,326 politicians, 1,873 Commons terms (since 1994), 6 s.
- contributions: 10.85M rows read in 32 s with 94 MB of memory. 817K rows
  were to candidates or contestants; 410K linked to 791 of the 952 people
  with terms (Elections Canada's itemized data starts in 2004); 407K
  unmatched (mostly candidates who never sat). 10,445 aggregate rows.
  Zero surname mismatches between filed recipient and linked person.
  All 101 namesake rows (two MPs named David Anderson) settled by the
  party on the return.

## Linking rules

1. Names normalised identically on both sides (`core/names.py`).
2. Exact name AND the return's date within one year of one of the
   person's terms -> `linked` (`exact_name_term`).
3. Several namesakes fit: if exactly one sat for the party named on the
   return in the matching term -> `linked` (`exact_name_term_party`);
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
