# services/semantic-enrich

A Python package + Typer CLI for two responsibilities:

1. **Library (4.3).** Two functions the pipeline imports directly:
   - `core.generate.generate_json` — guided JSON generation backed by `outlines` + `transformers`. Returns a parsed dict that conforms to the supplied pydantic schema or JSON Schema.
   - `core.embed.embed_batch` — batch encoding via `sentence-transformers`. Returns L2-normalised 1024-dim vectors so downstream cosine similarity reduces to dot product.

2. **Datasets pipeline (4.4).** Four CLI subcommands that materialise one row per CKAN package into `semantic.datasets`, split across two machines:
   - `datasets-extract` (laptop) — reads `raw.documents` + `raw.rows`, writes `stage/<run_id>/inputs/*.jsonl`.
   - `datasets-generate` (GPU box) — reads inputs, calls `generate_json`, writes `stage/<run_id>/datasets/*.jsonl` with `embedding=null`.
   - `datasets-embed` (GPU box) — reads datasets, calls `embed_batch`, atomically rewrites each JSONL with `embedding` populated.
   - `datasets-load` (laptop) — coalesces the JSONL, `bq load`s into a session-scoped staging table, MERGEs into `semantic.datasets`.

The on-disk `stage/<run_id>/` dir is the only contract between the two machines — `rsync` (or `scp -r`) moves it. The GPU box never speaks to `googleapis.com`.

## Models

| Pass       | HF repo                       | Dtype      | Output                     |
| ---------- | ----------------------------- | ---------- | -------------------------- |
| Generation | `Qwen/Qwen2.5-14B-Instruct`   | `bfloat16` | constrained-JSON dict      |
| Embedding  | `Qwen/Qwen3-Embedding-0.6B`   | `float16`  | 1024-dim L2-normed vector  |

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
| `WHENRICH_EMBEDDING_DIM`           | `1024`                        | Validated per-vector at the embed boundary               |
| `WHENRICH_EMBEDDING_BATCH_SIZE`    | `64`                          | Embedder batch size                                      |
| `WHENRICH_SAMPLE_ROWS_PER_PACKAGE` | `10`                          | Sample-row count fed to the prompt                       |
| `WHENRICH_SAMPLE_COLUMN_CAP`       | `40`                          | Column-name list cap (prompt-bounding)                   |
| `WHENRICH_FLUSH_EVERY_N_PACKAGES`  | `500`                         | Stage flush cadence                                      |
| `WHENRICH_STAGING_DIR`             | `services/semantic-enrich/stage` | Where the per-`<run_id>` JSONLs live                  |
| `WHENRICH_RUN_ID`                  | new UUID per process          | Override with `--run-id` to resume                       |

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
