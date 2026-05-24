# AGENTS.md — read first, every task

MapleQuery ingests open government data into BigQuery so an LLM agent can answer natural-language questions with cited sources. This file is a **map**, not a manual.

## Core beliefs

1. **`docs/` is a context map.** Code is the source of truth. Docs map the system's shape — stages, contracts, deferred decisions. Update a doc when the map changes; don't open doc PRs for routine code edits. If no doc covers a newly-mapped area, add one and link it from [`docs/index.md`](docs/index.md).
2. **Quarantine over drop.** Every error path either retries, quarantines with a named reason, or fails the run. Never `try/except: pass`, never silent absorbers like `year=unknown/`.
3. **Idempotency everywhere.** Every entry point — scheduled run, backfill, manual rerun — must be safe to run twice. Before merging, ask "what happens if this runs twice?" If the answer isn't "same outcome," the design is wrong.
4. **Rules are mechanical or they don't exist.** If a convention matters, ship it with a linter, structural test, or CI check. A rule that lives only in a doc is a wish.

## Operating rules

1. **Read this file before every task.** It points at the right deeper doc; it is not the doc.
2. **Service-local `AGENTS.md` overrides this one.** Each `services/<name>/` ships its own short `AGENTS.md`. Read it before working in that directory.
3. **Don't invent context.** If a fact is not in `docs/`, not in the code, and not in the conversation, ask.

## Where to find things

| Question | Go to |
| -- | -- |
| What's in the repo? | [`docs/index.md`](docs/index.md) |
| How is the system layered? | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Reliability / security bars | [`docs/RELIABILITY.md`](docs/RELIABILITY.md), [`docs/SECURITY.md`](docs/SECURITY.md) |

## What to load when

| Task shape | Load |
| -- | -- |
| Quick question / orientation | This file + [`ARCHITECTURE.md`](ARCHITECTURE.md). Stop there. |
| Small fix in an existing service | The file you're touching + the service's `AGENTS.md`. |
| Touching cross-cutting policy | [`docs/RELIABILITY.md`](docs/RELIABILITY.md) or [`docs/SECURITY.md`](docs/SECURITY.md). |

## How to work

- Use tooling and conventions already present. Don't introduce new libraries or patterns unless a doc authorises it.
- If something is missing (a tool, a lint, a doc), add it — don't work around it.
- Before opening a PR: relevant docs updated; lint and tests pass; PR description lists which docs were touched and why.
