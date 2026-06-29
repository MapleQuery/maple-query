# services/semantic-enrich

One CLI subcommand: `semantic-enrich smoke-test`. It exercises the two model-execution functions the enrichment pipeline calls into:

- `core.generate.generate_json` — guided JSON generation backed by `outlines` + `transformers`. Returns a parsed dict that conforms to the supplied pydantic schema or JSON Schema.
- `core.embed.embed_batch` — batch encoding via `sentence-transformers`. Returns L2-normalised 1024-dim vectors so downstream cosine similarity reduces to dot product.

There is no HTTP server. There is no swap procedure. The runtime is a Python package; the downstream enrichment pipeline imports `load_generation_model` / `generate_json` for the text-gen pass, drops the model, then imports `load_embedding_model` / `embed_batch` for the embedding pass.

## Models

| Pass       | HF repo                       | Dtype      | Output                     |
| ---------- | ----------------------------- | ---------- | -------------------------- |
| Generation | `Qwen/Qwen2.5-14B-Instruct`   | `bfloat16` | constrained-JSON dict      |
| Embedding  | `Qwen/Qwen3-Embedding-0.6B`   | `float16`  | 1024-dim L2-normed vector  |

If the GPU box can't hold 14B-bf16 weights, swap the generation model to `Qwen/Qwen2.5-7B-Instruct` via `WHENRICH_GENERATION_MODEL` — the `generate_json` signature is identical.

## Configuration

| Env var                     | Default                       | Purpose                  |
| --------------------------- | ----------------------------- | ------------------------ |
| `WHENRICH_GENERATION_MODEL` | `Qwen/Qwen2.5-14B-Instruct`   | HF repo for generation   |
| `WHENRICH_EMBEDDING_MODEL`  | `Qwen/Qwen3-Embedding-0.6B`   | HF repo for embedding    |
| `WHENRICH_DEVICE`           | `cuda`                        | torch device string      |
| `WHENRICH_HF_CACHE_DIR`     | (HF default)                  | Optional cache override  |

`WHENRICH_DEV=1` swaps the structlog JSON renderer for the console renderer so local runs are readable.

## First-time setup on the GPU box

```sh
cd services/semantic-enrich
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e '.[dev]'

# Pre-pull both models to the local HF cache (~30 GB).
hf download Qwen/Qwen2.5-14B-Instruct
hf download Qwen/Qwen3-Embedding-0.6B

# Pin to the intended card (see GPU pinning below).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# Smoke-test. Loads both models sequentially, asserts shape on both
# halves, writes MODELS.lock.
uv run semantic-enrich smoke-test --write-lock
```

Exit codes:
- `0` — both passes succeeded.
- `2` — a precondition failed (model load error, schema violation, dimension or L2-norm drift).
- `1` — unexpected internal error.

That's the whole runbook. No `nohup`, no PID files, no swap procedure, no ports to watch, no two venvs.

## GPU pinning

The reference GPU box has two cards (A6000 + Blackwell). Different CUDA capability levels mean torch's default device selection can pick the wrong one. `CUDA_DEVICE_ORDER=PCI_BUS_ID` + `CUDA_VISIBLE_DEVICES=0` (or `1`) constrains to the intended card.

## Memory budget

Sequential, not co-resident:

- Generation peak: ~28 GB (14B-bf16 weights) + ~10 GB (KV cache + CUDA workspace + activations) ≈ 38 GB.
- Embedding peak: ~2 GB (0.6B-fp16 + small KV).

Fits comfortably on a 48 GB A6000. Fits on a 24 GB card with `WHENRICH_GENERATION_MODEL=Qwen/Qwen2.5-7B-Instruct` as the generation fallback.

## How the pipeline uses the API

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

`del` + `empty_cache()` is the swap procedure. It is one line.

## conda as a fallback

If `uv` isn't installed and can't be (no apt access), conda envs work the same way — the package is plain `pip`-installable:

```sh
conda create -n semantic-enrich python=3.12 -y
conda activate semantic-enrich
pip install -e '.[dev]'
```

`pyproject.toml` is the source of truth either way.

## Tests

Unit tests only, mocking `outlines.models.transformers`, `outlines.generate.json`, and `sentence_transformers.SentenceTransformer.encode` against the call boundaries. No real model loads in CI:

```sh
cd services/semantic-enrich
uv run pytest
uv run ruff check .
uv run mypy src
```

Real model loads happen as part of `smoke-test --write-lock` on the operator's GPU box.
