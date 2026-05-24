Last updated: 2026-05-21

# Reliability bars

MapleQuery is **not** a user-facing realtime system. Bars are set
accordingly.

- **Ingest runs are best-effort.** A single failed run is not an
  incident; the system self-corrects on the next run. Sustained
  failure against the same source *is* — services define their own
  thresholds.
- **Data durability > availability.** Raw source bytes, once landed,
  are immutable. No deletes, no overwrites.
- **Idempotency everywhere.** Every entry point — scheduled run,
  backfill, manual re-run — must be safe to run twice.
- **No silent failures.** Quarantine or log structured; never drop.
  `try/except: pass` and `<field>=unknown/`-style absorbers that
  swallow input are forbidden.
- **Retries are bounded.** No infinite retries. Specific policies live
  with the code that retries.

Specific thresholds (TTLs, retry counts, alert windows) live in the
service that enforces them.
