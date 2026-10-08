# services/curate: read before working here

Normalize (M3). Builds `curated.*` (people, name variants, Commons terms,
linked election contributions, linked MP office expenses). Design: `docs/design/people-spine.md`;
operator doc: `docs/services/curate.md`.

Rules specific to this service:

1. **Only automatable sources.** Nothing that needs a person to
   download, request access, pass a bot challenge or buy a licence. No
   human-confirmation steps either: an unclear match stays `ambiguous`.
2. **Never link on riding.** Riding numbers were reused across electoral
   maps; riding is context on a term, never a key.
3. **Never guess a person.** A link needs an exact normalised name AND a
   term within a year of the record's date (party may break a tie
   between namesakes). Anything else is unmatched or `ambiguous`.
4. **Never store individual donors' identity.** Contributions are
   aggregated; individual contributors are counted by province only.
5. **Snapshot writes.** Each table is written whole (stage, then MERGE
   with delete), guarded against shrinking more than 50%. A re-run with
   the same inputs is a no-op.
6. **Terms from detail records only.** openparliament's paged
   `/politicians/memberships/` list skips and repeats rows; never read
   terms from it.
7. Schemas are `infra/terraform/schemas/curated_*.json`; the dataclasses in
   `src/curate/types.py` must match them (`tests/unit/test_schema_drift.py`).

Checks: `uv run ruff check src tests && uv run mypy src && uv run pytest && uv run lint-imports`.
