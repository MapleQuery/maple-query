# services/semantic-enrich

A Python package + Typer CLI for two responsibilities:

1. **Library (4.3).** Two functions the pipeline imports directly:
   - `core.generate.generate_json` — guided JSON generation backed by `outlines` + `transformers`. Returns a parsed dict that conforms to the supplied pydantic schema or JSON Schema.
   - `clients.openai.RealOpenAIClient.embed` — batch encoding via OpenAI text-embedding-3-small. Returns 1536-dim vectors. (Pre-4.7 the pipeline used `core.embed.embed_batch` against a local Qwen model; the local path is dead code kept in-tree for one clean re-enrichment cycle before removal.)

2. **Datasets pipeline (4.4 + 4.7).** Four CLI subcommands that materialise one row per CKAN package into `semantic.datasets`. Post-4.7, only `datasets-generate` needs the GPU box:
   - `datasets-extract` (laptop) — reads `raw.documents` + `raw.rows`, writes `stage/<run_id>/inputs/*.jsonl`.
   - `datasets-generate` (GPU box) — reads inputs, calls `generate_json`, writes `stage/<run_id>/datasets/*.jsonl` with `embedding=null`.
   - `datasets-embed` (laptop, post-4.7) — reads datasets, calls OpenAI text-embedding-3-small, atomically rewrites each JSONL with `embedding` populated.
   - `datasets-load` (laptop) — coalesces the JSONL, `bq load`s into a session-scoped staging table, MERGEs into `semantic.datasets`.

3. **Columns pipeline (4.5).** Four more CLI subcommands that materialise one row per `(package_id, column_name)` into `semantic.columns`. Same machine split, same `stage/<run_id>/` contract; reads `semantic.datasets.summary` for cross-pass context.
   - `columns-extract` (laptop) — reads `raw.documents`, `raw.rows`, and `semantic.datasets`, writes `stage/<run_id>/column_inputs/*.jsonl`. One file per package; per-package fan-out at `extract_concurrency=16` workers means a ~3,693-package backfill completes in ~10-20 minutes rather than ~hours.
   - `columns-generate` (GPU box) — reads column_inputs, chunks each package into ≤100-column batches, calls `generate_json_list` once per chunk, validates the 1:1 column-name mapping invariant per chunk and per package, writes `stage/<run_id>/columns/*.jsonl`.
   - `columns-embed` (laptop, post-4.7) — reads columns, embeds the `description` field via OpenAI text-embedding-3-small. Same loop as `datasets-embed`; `embedding_pass._embed_files` is parameterised on artifact + row type.
   - `columns-load` (laptop) — coalesces the JSONL, validates pre-load, `bq load`s into a session-scoped staging table, MERGEs into `semantic.columns` on `(package_id, column_name)`. Failure markers and embedding-null rows are filtered out at coalesce time.

The on-disk `stage/<run_id>/` dir is the only contract between the two machines — `rsync` (or `scp -r`) moves it. The GPU box never speaks to `googleapis.com`.

## Models

| Pass       | Model                              | Where                                | Output                    |
| ---------- | ---------------------------------- | ------------------------------------ | ------------------------- |
| Generation | `Qwen/Qwen2.5-14B-Instruct`        | GPU box (HF, `bfloat16`)             | constrained-JSON dict     |
| Embedding  | `openai:text-embedding-3-small`    | Laptop (OpenAI API, post-4.7)        | 1536-dim vector           |

If a card can't hold 14B-bf16, swap to `Qwen/Qwen2.5-7B-Instruct` via `WHENRICH_GENERATION_MODEL` — the `generate_json` signature is identical.

## Configuration

| Env var                            | Default                       | Purpose                                                  |
| ---------------------------------- | ----------------------------- | -------------------------------------------------------- |
| `WHENRICH_GENERATION_MODEL`        | `Qwen/Qwen2.5-14B-Instruct`   | HF repo for generation                                   |
| `WHENRICH_EMBEDDING_MODEL`         | `Qwen/Qwen3-Embedding-0.6B`   | HF repo for embedding                                    |
| `WHENRICH_DEVICE`                  | `cuda`                        | torch device string                                      |
| `WHENRICH_HF_CACHE_DIR`            | (HF default)                  | Optional cache override                                  |
| `WHENRICH_GCP_PROJECT_ID`          | (or `GCP_PROJECT_ID`)         | Required for `datasets-extract` / `datasets-load`        |
| `WHENRICH_GENERATION_MAX_TOKENS`   | `800`                         | Per-call max-new-tokens for constrained JSON             |
| `WHENRICH_GENERATION_TEMPERATURE`  | `0.0`                         | Greedy by default (deterministic)                        |
| `WHENRICH_GENERATION_DTYPE`        | `bfloat16`                    | Torch dtype string                                       |
| `WHENRICH_EMBEDDING_DIM`           | `1024`                        | Legacy (pre-4.7) local-Qwen dim knob                     |
| `WHENRICH_EMBEDDING_BATCH_SIZE`    | `64`                          | Legacy (pre-4.7) local-Qwen batch size                   |
| `WHENRICH_OPENAI_API_KEY`          | (or `OPENAI_API_KEY`)         | Required for `*-embed` and `*-reembed`                   |
| `WHENRICH_OPENAI_EMBEDDING_MODEL`  | `text-embedding-3-small`      | OpenAI embedding model id                                |
| `WHENRICH_OPENAI_EMBEDDING_DIM`    | `1536`                        | Sanity-check knob asserted per-vector                    |
| `WHENRICH_OPENAI_EMBEDDING_BATCH_SIZE` | `128`                     | Batch size sent to OpenAI                                |
| `WHENRICH_OPENAI_REQUEST_TIMEOUT_S` | `30.0`                       | Per-request timeout                                      |
| `WHENRICH_OPENAI_MAX_RETRIES`      | `3`                           | Tenacity retries on rate-limit + 5xx                     |
| `WHENRICH_SAMPLE_ROWS_PER_PACKAGE` | `10`                          | Sample-row count fed to the prompt                       |
| `WHENRICH_SAMPLE_COLUMN_CAP`       | `40`                          | Column-name list cap (prompt-bounding)                   |
| `WHENRICH_FLUSH_EVERY_N_PACKAGES`  | `500`                         | Stage flush cadence                                      |
| `WHENRICH_STAGING_DIR`             | `services/semantic-enrich/stage` | Where the per-`<run_id>` JSONLs live                  |
| `WHENRICH_RUN_ID`                  | new UUID per process          | Override with `--run-id` to resume                       |
| `WHENRICH_EXTRACT_CONCURRENCY`     | `16`                          | Per-package BQ fan-out for `*-extract`                   |
| `WHENRICH_COLUMN_CHUNK_SIZE`       | `100`                         | Columns per `generate_json_list` call (4.5)              |
| `WHENRICH_COLUMN_CHUNK_MAX_CHUNKS_PER_PACKAGE` | `20`              | Wide-package safety belt (4.5)                           |
| `WHENRICH_COLUMN_SAMPLE_VALUES_CAP` | `10`                         | Per-column sample-value cap (4.5)                        |
| `WHENRICH_COLUMN_NAME_ALLOWLIST_RE` | (see settings.py)            | Column-name allowlist regex (4.5)                        |
| `WHENRICH_COLUMN_CHUNK_RETRY_TEMPERATURE` | `0.2`                  | Temperature for the single retry on chunk invariant violation (4.5) |
| `WHENRICH_BQ_COLUMNS_TABLE`        | `columns`                     | Target table name in `semantic.*`                        |
| `WHENRICH_OPENAI_GENERATION_MODEL` | `gpt-4o`                      | OpenAI model for SQL generation (4.6 harness)            |
| `WHENRICH_OPENAI_GENERATION_TEMPERATURE` | `0.0`                   | Determinism for the prompt-iteration loop                |
| `WHENRICH_OPENAI_GENERATION_MAX_TOKENS` | `1024`                   | Per-call max-tokens for SQL generation                   |
| `WHENRICH_EVAL_K_PACKAGES`         | `5`                           | Top-k for package `VECTOR_SEARCH`                        |
| `WHENRICH_EVAL_K_COLUMNS`          | `15`                          | Top-k for column `VECTOR_SEARCH` (scoped)                |
| `WHENRICH_EVAL_MAX_BYTES_BILLED`   | `53687091200` (50 GB)         | Dry-run cost cap for the SQL guard                       |
| `WHENRICH_EVAL_QUERY_TIMEOUT_MS`   | `30000`                       | Execution timeout                                        |
| `WHENRICH_EVAL_ROW_LIMIT`          | `100`                         | Hard `LIMIT` enforced by the guard                       |

`WHENRICH_DEV=1` swaps the structlog JSON renderer for the console renderer so local runs are readable.

## First-time setup on the GPU box

The GPU box has anaconda only (no `uv`, and the operator can't install one), so the canonical setup is conda:

```sh
cd services/semantic-enrich
conda create -n semantic-enrich python=3.12 -y
conda activate semantic-enrich
pip install -e '.[dev]'
```

On the laptop (which has `uv`), the alternative is:

```sh
cd services/semantic-enrich
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e '.[dev]'
```

Pre-pull both models to the local HF cache (~30 GB):

```sh
huggingface-cli download Qwen/Qwen2.5-14B-Instruct
huggingface-cli download Qwen/Qwen3-Embedding-0.6B
```

Pin the GPU. The reference box has two cards with different CUDA capability levels — torch's default device selection can pick the wrong one:

```sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0   # or 1, depending on which card
```

Smoke-test (with the `semantic-enrich` conda env active), locking the resolved package versions and HF commit SHAs into `MODELS.lock`:

```sh
semantic-enrich smoke-test --write-lock
```

Exit codes:
- `0` — both passes succeeded.
- `2` — a precondition failed (model load error, schema violation, dimension or L2-norm drift).
- `1` — unexpected internal error.

Commit `MODELS.lock` from the GPU box after the first successful run so future runs reproduce against the same versions.

## Memory budget

Sequential, not co-resident:

- Generation peak: ~28 GB (14B-bf16 weights) + ~10 GB (KV cache + CUDA workspace + activations) ≈ 38 GB.
- Embedding peak: ~2 GB (0.6B-fp16 + small KV).

Fits on a 48 GB A6000. Fits on a 24 GB card with `WHENRICH_GENERATION_MODEL=Qwen/Qwen2.5-7B-Instruct`.

## Datasets pipeline — operator runbook

A two-machine flow. The laptop holds ADC and talks to BigQuery; the GPU box holds the weights and never reaches `googleapis.com`. `rsync` ferries `stage/<run_id>/` between them.

### Prereqs

- 4.1 done: `raw.documents.package_id` REQUIRED + populated.
- 4.2 done: `semantic.datasets` table exists with the schema in `infra/terraform/schemas/semantic_datasets.json`.
- 4.3 done: GPU box has the venv installed, `smoke-test --write-lock` passed, `MODELS.lock` committed.
- Laptop has ADC (`gcloud auth application-default login`) and exports `WHENRICH_GCP_PROJECT_ID` in the shell that runs `datasets-extract` / `datasets-load`.

Laptop sections use `uv run`; GPU-box sections call the `semantic-enrich` console script directly with the `semantic-enrich` conda env active.

### Smoke ladder

```sh
# (laptop)
uv run semantic-enrich datasets-extract --run-id smoke-1 --limit-packages 1
rsync -av services/semantic-enrich/stage/smoke-1/ \
  gpu-box:.../services/semantic-enrich/stage/smoke-1/

# (GPU box — `conda activate semantic-enrich` first)
semantic-enrich datasets-generate --run-id smoke-1
semantic-enrich datasets-embed   --run-id smoke-1
rsync -av gpu-box:.../services/semantic-enrich/stage/smoke-1/ \
  services/semantic-enrich/stage/smoke-1/

# (laptop)
uv run semantic-enrich datasets-load --run-id smoke-1
bq query --use_legacy_sql=false \
  "SELECT * FROM \`<proj>.semantic.datasets\`
   WHERE generated_at > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)"

# Repeat with --limit-packages 10, then 100. Human-review each step.
```

### Full backfill

```sh
RUN_ID="m3-datasets-$(date +%Y-%m-%d)"
# (laptop)
uv run semantic-enrich datasets-extract --run-id "$RUN_ID"
rsync -av services/semantic-enrich/stage/"$RUN_ID"/ \
  gpu-box:.../services/semantic-enrich/stage/"$RUN_ID"/

# (GPU box — `conda activate semantic-enrich` first) — ~2-3 hours for ~3,693 packages
semantic-enrich datasets-generate --run-id "$RUN_ID"
semantic-enrich datasets-embed   --run-id "$RUN_ID"
rsync -av gpu-box:.../services/semantic-enrich/stage/"$RUN_ID"/ \
  services/semantic-enrich/stage/"$RUN_ID"/

# (laptop)
uv run semantic-enrich datasets-load --run-id "$RUN_ID"
```

### Recovery

Every subcommand is resumable by re-running with the same `--run-id`:

- `datasets-extract` re-skips packages already in `stage/<run_id>/inputs/`.
- `datasets-generate` re-skips packages already in `stage/<run_id>/datasets/`.
- `datasets-embed` re-skips rows with non-null `embedding`.
- `datasets-load` is idempotent — the MERGE's `s.generated_at > t.generated_at` guard makes a second load against the same stage a no-op.

If a stage dir is corrupted: delete `stage/<RUN_ID>/<artifact>/` for the affected artifact and re-run from the upstream step.

### Fallback (all-on-GPU-box)

If the operator prefers to skip the rsync dance and the GPU box can reach `googleapis.com`, copy `~/.config/gcloud/application_default_credentials.json` to the GPU box and `export GOOGLE_APPLICATION_CREDENTIALS=/path/to/copied/adc.json`. With ADC in place, all four subcommands can run on the GPU box. The laptop-broker model is canonical because the refresh token is personal and 7-day-expiring, and most lab/GPU boxes are firewalled from `googleapis.com`.

## Columns pipeline — operator runbook

Same two-machine flow as the datasets pipeline, on a separate `--run-id`. Run the datasets pipeline first so `semantic.datasets.summary` is populated — the columns prompt picks it up automatically via `columns-extract`. If you run columns before datasets, the prompt falls back to raw-side context only (logged as `package_summary_unavailable`) and column descriptions are strictly weaker; the canonical order below makes the cross-pass lookup hit.

### Prereqs

- 4.1–4.4 done (laptop has ADC + `WHENRICH_GCP_PROJECT_ID`; GPU box has the venv + `MODELS.lock`).
- `semantic.datasets` populated end-to-end (i.e. you've completed the datasets-pipeline runbook above).

### Smoke ladder

```sh
# (laptop) — pick one package id that the datasets pipeline has already loaded
PKG=<some package id>
uv run semantic-enrich columns-extract --run-id smoke-cols-1 --limit-package-ids "$PKG"
rsync -av services/semantic-enrich/stage/smoke-cols-1/ \
  gpu-box:.../services/semantic-enrich/stage/smoke-cols-1/

# (GPU box — `conda activate semantic-enrich` first)
semantic-enrich columns-generate --run-id smoke-cols-1
semantic-enrich columns-embed    --run-id smoke-cols-1
rsync -av gpu-box:.../services/semantic-enrich/stage/smoke-cols-1/ \
  services/semantic-enrich/stage/smoke-cols-1/

# (laptop)
uv run semantic-enrich columns-load --run-id smoke-cols-1
bq query --use_legacy_sql=false \
  "SELECT package_id, column_name, semantic_type, description
   FROM \`<proj>.semantic.columns\`
   WHERE package_id = '$PKG'
   ORDER BY column_name"

# Repeat with --limit-packages 10, then 100, with distinct --run-id values.
# Human-review each rung — verify descriptions read coherently and
# semantic_type values aren't all defaulting to "text".
```

### Full backfill

```sh
RUN_ID="m3-columns-$(date +%Y-%m-%d)"
# (laptop) — ~10-20 min for ~3,693 packages at 16-way fan-out
uv run semantic-enrich columns-extract --run-id "$RUN_ID"
rsync -av services/semantic-enrich/stage/"$RUN_ID"/ \
  gpu-box:.../services/semantic-enrich/stage/"$RUN_ID"/

# (GPU box — `conda activate semantic-enrich` first) — ~2-3 hours
# for ~3,693 packages × ~1.5 median chunks × ~30 s/chunk; wide
# packages (the 1,383-column outlier) take ~7 min each.
semantic-enrich columns-generate --run-id "$RUN_ID"
semantic-enrich columns-embed    --run-id "$RUN_ID"
rsync -av gpu-box:.../services/semantic-enrich/stage/"$RUN_ID"/ \
  services/semantic-enrich/stage/"$RUN_ID"/
# ~900 MB pull. ~30 seconds on a residential uplink.

# (laptop) — one bq load + one MERGE. Seconds to a few minutes.
uv run semantic-enrich columns-load --run-id "$RUN_ID"
```

### Validation queries

```sh
PROJ=<your project>
# Row count in the 100K-200K band (parent expects ~150K).
bq query --use_legacy_sql=false \
  "SELECT COUNT(*) FROM \`$PROJ.semantic.columns\`"
# All embeddings 1024-dim.
bq query --use_legacy_sql=false \
  "SELECT COUNT(*) FROM \`$PROJ.semantic.columns\`
   WHERE ARRAY_LENGTH(embedding) != 1024"
# All descriptions present + ≥20 chars.
bq query --use_legacy_sql=false \
  "SELECT COUNT(*) FROM \`$PROJ.semantic.columns\`
   WHERE description IS NULL OR LENGTH(description) < 20"
# Every dataset row has at least one column row.
bq query --use_legacy_sql=false \
  "SELECT d.package_id FROM \`$PROJ.semantic.datasets\` d
   LEFT JOIN (SELECT package_id, COUNT(*) AS n
              FROM \`$PROJ.semantic.columns\` GROUP BY package_id) c
   USING (package_id)
   WHERE c.n IS NULL OR c.n = 0"
```

### Recovery

Each subcommand is resumable with the same `--run-id`:

- `columns-extract` re-skips packages already in `stage/<run_id>/column_inputs/`.
- `columns-generate` re-skips packages already in `stage/<run_id>/columns/` — including failure-marker packages (so a re-run doesn't reprocess a known-bad package without operator intervention; delete the failure marker line to retry).
- `columns-embed` re-skips rows with non-null `embedding`.
- `columns-load` is idempotent via the MERGE's `s.generated_at > t.generated_at` guard.

### Disk hygiene

The `columns/*.jsonl` for the full backfill is ~900 MB (150K rows × ~6 KB/row with embeddings as JSON). After acceptance, archive:

```sh
tar -czf "$RUN_ID-stage.tar.gz" services/semantic-enrich/stage/"$RUN_ID"/
```

## Reembed runbook (4.7 — one-off OpenAI swap)

Overwrites `semantic.datasets.embedding` and `semantic.columns.embedding` with OpenAI text-embedding-3-small vectors (1536-dim). Source text (`summary`, `description`) is not touched.

Runs entirely on the laptop — no GPU box, no rsync, no `stage/*.jsonl` on disk. Estimated cost for the full corpus: **~$0.34**. Estimated wall clock: datasets < 5 min, columns < 30 min.

### Prereqs

- Laptop has ADC (`gcloud auth application-default login`) and exports `WHENRICH_GCP_PROJECT_ID` (or `GCP_PROJECT_ID`).
- `OPENAI_API_KEY` (or `WHENRICH_OPENAI_API_KEY`) is set.
- `semantic.datasets` and `semantic.columns` already populated (i.e. the 4.4 + 4.5 backfills ran with the old Qwen vectors).

### Fast path — one command

```sh
cd services/semantic-enrich
# dry-run: previews row counts, prints one would_have_reembedded event
# per row, calls no OpenAI, writes no MERGE. Sanity-check first.
scripts/reembed.sh --dry-run

# real run: datasets first, then columns.
scripts/reembed.sh
```

Flags:
- `--dry-run` — skip OpenAI + MERGE.
- `--datasets-only` / `--columns-only` — run one of the two passes.

### Direct CLI (if you want the granular knobs)

```sh
cd services/semantic-enrich
RUN_ID="reembed-$(date +%Y-%m-%d)"

uv run semantic-enrich datasets-reembed --run-id "$RUN_ID"
uv run semantic-enrich columns-reembed  --run-id "$RUN_ID"
```

### Verify

After both reembeds complete, every `embedding` array should be 1536-dim:

```sh
PROJ=<your project>
bq query --use_legacy_sql=false \
  "SELECT COUNT(*) AS n_wrong FROM \`$PROJ.semantic.datasets\`
   WHERE ARRAY_LENGTH(embedding) != 1536"
bq query --use_legacy_sql=false \
  "SELECT COUNT(*) AS n_wrong FROM \`$PROJ.semantic.columns\`
   WHERE ARRAY_LENGTH(embedding) != 1536"
```

Both queries should return `n_wrong = 0`.

### Recovery

- Idempotent: the runners can be run twice — the second run rewrites the same rows to the same vectors (OpenAI embeddings are deterministic).
- A crash before the MERGE leaves the target table untouched. The staging table auto-expires in 24 h.
- To roll back to Qwen vectors, rerun the pre-4.7 datasets-embed + columns-embed pipeline on the GPU box. Same operation, opposite direction.

## Representative picker + backfill runbook

Each package's enrichment (sample rows, column extraction) reads one *representative* resource: the median-row-count CSV, lexicographic `document_id` tiebreak. Resources whose headers are **dictionary-shaped** — ≤8 columns with ≥60% of names drawn from a schema-description vocabulary (`column_name`, `data_type`, `description`, …) — are demoted out of the pool first, so a package that publishes a data CSV plus a schema/dictionary CSV enriches from the data, not the metadata. If *every* resource is dictionary-shaped the full pool is restored (a dictionary-only package still gets a representative for browseability). Every pick emits a `representative_picked` event with `dictionary_candidates` and the chosen doc.

The chosen `representative_document_id` is persisted on `semantic.datasets` (threaded extract → generate → load), along with `title` — the representative resource's `raw.documents.title`, falling back to the first non-null resource title, NULL when no resource has one (the web UI then falls back to `package_id`; no title is ever invented). For rows enriched before the columns existed, or after a picker behaviour change:

```sh
cd services/semantic-enrich

# Dry-run first: logs one representative_picked per package and a
# representative_pick_changed event for each package whose stored
# choice differs. That change list is the re-enrichment scope.
uv run semantic-enrich datasets-backfill-representative --dry-run

# Real run: stages picks and MERGEs representative_document_id +
# title only. generated_at is untouched (it is the always-newer-wins
# clock for datasets-load; a picker-only backfill must not advance it).
uv run semantic-enrich datasets-backfill-representative
```

Laptop-only, no GPU. Idempotent — re-running merges the same picks. Packages that swap representatives still carry `semantic.columns` rows extracted from the old file; re-run the columns pipeline scoped to those package ids (`--limit-package-ids`) to refresh them.

## Retrieval-validation harness — operator runbook

The `semantic-enrich eval` subcommand grades the semantic layer end-to-end against a committed 20-question fixture: embed the NL question, run `VECTOR_SEARCH` over `semantic.datasets` and `semantic.columns`, ask `gpt-4o` to emit one SQL statement via Structured Outputs, dry-run-validate the SQL under hard guardrails, execute it, and score the result. Laptop-side; no GPU box dependency.

### Prereqs

- `semantic.datasets` and `semantic.columns` populated end-to-end.
- Both tables re-embedded via `datasets-reembed` / `columns-reembed` so `ARRAY_LENGTH(embedding) = 1536`. The runner refuses to proceed against a stale 1024-dim warehouse.
- `WHENRICH_GCP_PROJECT_ID` (or `GCP_PROJECT_ID`) exported.
- `WHENRICH_OPENAI_API_KEY` (or `OPENAI_API_KEY`) exported.

### Loop

```sh
# Prompt-iteration loop (no BQ execution, just retrieval + SQL gen + dry-run).
uv run semantic-enrich eval --no-execute

# One question end-to-end.
uv run semantic-enrich eval --question-ids q01

# Full run with a relaxed cost cap for a deep-dive.
uv run semantic-enrich eval --max-bytes-billed 214748364800   # 200 GB

# Harness self-test — zero OpenAI / BQ calls; verifies the fixture parses,
# the prompt template renders under StrictUndefined, and the report writer
# lands its output files.
uv run semantic-enrich eval --dry-run
```

Full 20-question runs complete in under 90 s on the laptop against warm BQ slots; expect ~\$0.15 in gpt-4o + embedding cost per full run.

### Fixture

Committed at `services/semantic-enrich/eval/questions.yaml`. 20 hand-curated questions across `housing`, `taxes`, `fisheries`, `immigration`, `census`, `environment`, `methodology`, `multi_domain`. `expected_packages` UUIDs are populated by the operator against CKAN search + `raw.documents` — the first pass leaves them empty (contributes 1.0 to per-question recall, documented in the report header). Grow the fixture in `+10` increments as a `v2 fixture` file after M3 ships, not by mutating this one.

### Prompt template

`services/semantic-enrich/eval/prompts/sql_generation.j2`. The load-bearing artefact of the harness. Rendered under `StrictUndefined` (a missing variable raises rather than emitting literal `None`). Edit and re-run; the `prompt_template_hash` in the summary confirms two runs used the same prompt.

### Retrieval stages

Retrieval is three-stage: `VECTOR_SEARCH` over `semantic.datasets` → scoped `VECTOR_SEARCH` over `semantic.columns` → a plain `SELECT` from `raw.documents` filtered to `load_status = 'loaded'` and scoped to the candidate packages. The document stage exists so the harness can inline resolved `document_id`s as literals in the SQL-gen prompt. `raw.rows` is clustered by `document_id` and only a literal `WHERE document_id IN ('…', '…')` shape is plan-time-prunable — a subquery IN (even against `raw.documents`) forces the full ~200 GB clustered scan. The `sql_no_document_id_filter` guard rule enforces this at rejection time; it fires if the model regresses to a subquery-IN or a join through `raw.documents`.

### Reports

Written to `services/semantic-enrich/eval/reports/<run_id>.{json,md}`. JSON is machine-diffable; Markdown is the operator's reading surface. The directory is `.gitignore`d by default; commit a specific acceptance report via `git add -f`.

Each question grades into exactly one terminal state:
- `answered` — SQL executed, returned rows (for `must_return_rows=true`).
- `no_rows` — SQL executed, empty result.
- `sql_invalid` — guard rejected (parse, SELECT-only, identifier whitelist, multi-statement, missing literal `document_id` IN-filter on `raw.rows`).
- `sql_too_expensive` — guard rejected on cost cap.
- `sql_timed_out` — execution exceeded 30 s.
- `retrieval_miss` — zero `VECTOR_SEARCH` hits.
- `sql_not_generated` — Structured Outputs violation despite strict mode (vendor regression).
- `execution_error` — BQ runtime error.

### Recovery

Every question grade is appended to `<run_id>.partial.jsonl` before the next question starts. A mid-run crash leaves the partial file for review; the final JSON + MD are written only on clean completion. Re-run against the same warehouse; determinism (temperature 0.0, BQ query cache on, sequential order) yields byte-identical per-question grades modulo `run_id` and timestamps.

### Agent-mode baseline capture

`eval --agent-mode` replays a labeled question fixture through the live agent loop instead of the retrieval harness — one fresh single-turn conversation per question, tracing on when configured:

```sh
# Full baseline capture (writes eval/reports/agent-traces-<run_id>.json).
uv run semantic-enrich eval --agent-mode --questions eval/questions-agent-traces.yaml

# Pin the output name for a committed baseline.
uv run semantic-enrich eval --agent-mode --questions eval/questions-agent-traces.yaml \
  --output eval/reports/agent-traces-baseline-v1.json

# Self-test without OpenAI/BQ (canned client, no network). --questions is
# required here too — the default fixture is the retrieval harness's.
uv run semantic-enrich eval --agent-mode --dry-run --limit 3 \
  --questions eval/questions-agent-traces.yaml --output /tmp/agent-dry.json
```

The fixture (`eval/questions-agent-traces.yaml`) is derived from real agent traffic: every entry carries the verbatim user question, `expected.triage` (`in_scope | off_scope | meta | clarify`) and `expected.outcome` (`answered | no_data | deflected | clarify`) labels, optional `packages_any_of` / `must_caveat`, and an `observed_v1` record of what the loop actually did in the source traces. The report records terminal state, final message, tool-call count, dollars, and per-call token series per question — no automatic grading; it is the denominator for before/after comparisons of loop changes. This is a manual, occasional capture (14 questions, well under $1) — do not schedule repeated runs.

## Agent span tracing

With a Braintrust key configured, the agent loop emits a full span tree per conversation instead of disconnected per-call roots:

```text
session:<conversation_id>            one root per conversation (created once,
└── agent.run_turn                   exported, closed; turns attach by export string)
    ├── openai.chat / embeddings     wrap_openai auto-instrumentation
    └── tool.<name>                  per tool call, digest-only payloads
```

- **Turn spans** (`core/agent_tracing.run_turn_traced`) wrap the loop generator; they carry the question, `prompt_hash`, `snapshot_hash`, `loop_impl`, `system_prompt_tokens`, and log final message, terminal state (`done | error | timeout | abandoned`), tool-call count, dollars, and per-call `tokens_in/out` arrays (the series that shows history bloat across loop iterations).
- **Session spans** are stateless parents: `SessionSpanMap` keeps a per-process LRU (`WHENRICH_AGENT_TRACE_SESSION_MAP_ENTRIES`, TTL `WHENRICH_AGENT_TRACE_SESSION_TTL_SECONDS`) of `conversation_id → span export string`. An instance restart mid-conversation starts a second session root — accepted; the map is best-effort by design.
- **Tool spans** parent to the turn span by export string (robust across pool threads) and log digests only: full SQL + rationale for `run_sql`, first-3-rows samples, candidate lists reduced to ids + distance. Never full row payloads.
- **Prompt gauge**: `system_prompt_gauge` is logged at startup with the tiktoken count of the rendered system prompt; a unit test bounds it under 3.0K tokens so silent prompt growth fails CI.
- **Kill switch**: `WHENRICH_AGENT_TRACE_SESSIONS=false` (or simply no Braintrust key) turns every wrapper into a passthrough; the loop's event stream is identical either way.

## Self-enforcing tool contract

Every deterministic rule the system prompt used to spell out is enforced inside the tools (`core/agent_tools.py` + `core/retrieval.py`); the prompt keeps one-liners. Tool names and existing schema fields are frozen — everything below is additive or server-side. Normalization lives in `core/sql_normalize.py` and is shared by every LLM-SQL door: the agent's `run_sql` and the offline eval runner both call `normalize_sql` before the guard, so eval scores grade the SQL production actually runs.

**`run_sql` hardening** (order: normalize tables → auto-quote JSONPaths → pairing check → guard → execute → NULL-ratio advisory):

- *Table normalization*: bare `raw.rows`, backticked `` `raw.rows` ``, and placeholder-project forms (`<project>`, `PROJECT_ID`, …) after `FROM`/`JOIN` are rewritten to `` `<project>.raw.rows` `` before guarding. The scan masks string literals, `--`/`#`/`/* */` comments, and quote characters inside backtick identifiers first, so a `raw.rows` inside a quoted string or comment is never touched and an apostrophe in a comment can't desync the scan. The guard still runs after and remains the authority on allowed tables.
- *JSONPath auto-quoting*: for `JSON_VALUE` / `JSON_QUERY` / `JSON_EXTRACT` / `JSON_EXTRACT_SCALAR` and their `_ARRAY` variants, when the second arg is a string-literal path (single- or double-quoted), bare segments not matching `[A-Za-z_][A-Za-z0-9_]*` get double-quoted (`$.2020-21_Exp` → `$."2020-21_Exp"`); already-quoted segments and anything unparseable (array subscripts / bracketed keys, escapes, non-literal args) pass through untouched — the prompt tells the model bracketed keys are its own job. The rewritten SQL is what runs and what `sql_guarded`/`sql_executed` events show; the tool result's `normalizations` field (`json_paths_quoted`, `tables_rewritten`) teaches the model the corrected form, and the `agent_autoquote_applied` structlog counter measures how often it was needed.
- *Pairing check scoping*: set-operation queries are pairing-checked per arm, so the sanctioned cross-doc pattern — one SELECT per doc combined with `UNION ALL`, each arm referencing only its own doc's columns — passes instead of being cross-product-rejected. Violations within an arm still reject.
- *NULL-ratio advisory*: columns whose NULL share of the returned rows is ≥ `WHENRICH_AGENT_NULL_RATIO_THRESHOLD` (default 0.8) get a `null_ratio_warning` in the tool result and on `sql_executed`. `status` stays `"ok"` — it's a hint that the referenced key probably doesn't match the row bodies. Zero-row results, `document_id`, and ungrouped-aggregate output columns (including BigQuery's `f0_` auto-names; windowed aggregates stay in scope) are excluded. An ungrouped aggregate that comes back all-NULL instead gets an `aggregate_null_note` — NULL there is ambiguous between zero matching rows and a semantically wrong column, so the note asks the model to disambiguate with a `COUNT(*)` before answering "no data".

**`list_documents` enrichment**:

- Per-column `column_samples` ride alongside the unchanged `columns` list — one batched, cluster-pruned query per call (`WHENRICH_AGENT_SAMPLE_VALUES_ROWS` rows/doc, values truncated to `WHENRICH_AGENT_SAMPLE_VALUE_MAX_CHARS`, keys capped at `WHENRICH_AGENT_SAMPLE_VALUES_MAX_COLUMNS`), sharing `sample_rows`' budget posture. Sampling failure degrades to no samples, never a tool error.
- Docs whose `__col_N` share of columns exceeds `WHENRICH_AGENT_GENERATED_HEADER_RATIO` (default 0.5) carry `"quality": "low_generated_headers"` and sort after all clean docs — demoted, not dropped.
- Optional `required_columns` filter returns only docs containing every listed column (with `filtered_out` diagnostics); if nothing satisfies it, the full listing comes back with `required_columns_unsatisfiable: true` instead of an empty result.

**`search_datasets`** exposes `similarity` (`round(1 - distance, 4)`) per candidate and `top_similarity` on the envelope.

**`describe_corpus`** is a zero-arg read-only tool for corpus meta-questions ("how many rows do you have?"): package/document counts from the small metadata tables, `rows_total` from table metadata (`get_table`, free — `raw.rows` is never scanned), freshness from `MAX(loaded_at)`. Cached in-process for `WHENRICH_AGENT_SNAPSHOT_REFRESH_SECONDS`; still counts against the tool-call budget.

## How the library API is used

```py
gen_model = load_generation_model()
for pkg in packages:
    summary = generate_json(prompt(pkg), DatasetSchema, model=gen_model)
    ...

del gen_model
torch.cuda.empty_cache()

emb_model = load_embedding_model()
for batch in batched(texts, 128):
    vecs = embed_batch(batch, model=emb_model)
    ...
```

The `del` + `empty_cache()` between passes frees the generation-model VRAM so the embedder has room to load. The CLI's `datasets-generate` and `datasets-embed` are separate processes — process exit handles the VRAM release implicitly.

## Tests

Unit tests stub the model boundaries — `outlines.from_transformers`, the wrapped model's `__call__`, and `sentence_transformers.SentenceTransformer.encode` — so no real model loads happen in CI:

```sh
cd services/semantic-enrich
pytest                        # or `uv run pytest`
ruff check src tests
mypy src
lint-imports                  # layered-architecture contract
```

Integration tests use a `FakeBqClient` and deterministic stand-ins for `generate_json` / `embed_batch`, so the full extract/generate/embed/load flow exercises end-to-end without a GPU or a BQ project.

Real model loads happen on the GPU box via `semantic-enrich smoke-test --write-lock` (4.3) and the live smoke ladder above (4.4).
