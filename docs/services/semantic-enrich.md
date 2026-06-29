# services/semantic-enrich

A Python package the downstream enrichment pipeline imports. Two functions and a smoke test:

- `core.generate.generate_json` — guided JSON generation backed by `outlines` + `transformers`. Returns a parsed dict that conforms to the supplied pydantic schema or JSON Schema.
- `core.embed.embed_batch` — batch encoding via `sentence-transformers`. Returns L2-normalised 1024-dim vectors so downstream cosine similarity reduces to dot product.

The pipeline imports `load_generation_model` + `generate_json` for the text-gen pass, drops the model, then imports `load_embedding_model` + `embed_batch` for the embedding pass — sequential because the 14B-bf16 weights and the embedder don't co-reside on a 48 GB card without crowding the generation KV cache.

## Models

| Pass       | HF repo                       | Dtype      | Output                     |
| ---------- | ----------------------------- | ---------- | -------------------------- |
| Generation | `Qwen/Qwen2.5-14B-Instruct`   | `bfloat16` | constrained-JSON dict      |
| Embedding  | `Qwen/Qwen3-Embedding-0.6B`   | `float16`  | 1024-dim L2-normed vector  |

If a card can't hold 14B-bf16, swap to `Qwen/Qwen2.5-7B-Instruct` via `WHENRICH_GENERATION_MODEL` — the `generate_json` signature is identical.

## Configuration

| Env var                     | Default                       | Purpose                  |
| --------------------------- | ----------------------------- | ------------------------ |
| `WHENRICH_GENERATION_MODEL` | `Qwen/Qwen2.5-14B-Instruct`   | HF repo for generation   |
| `WHENRICH_EMBEDDING_MODEL`  | `Qwen/Qwen3-Embedding-0.6B`   | HF repo for embedding    |
| `WHENRICH_DEVICE`           | `cuda`                        | torch device string      |
| `WHENRICH_HF_CACHE_DIR`     | (HF default)                  | Optional cache override  |

`WHENRICH_DEV=1` swaps the structlog JSON renderer for the console renderer so local runs are readable.

## First-time setup on the GPU box

With `uv`:

```sh
cd services/semantic-enrich
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e '.[dev]'
```

With `conda`:

```sh
cd services/semantic-enrich
conda create -n semantic-enrich python=3.12 -y
conda activate semantic-enrich
pip install -e '.[dev]'
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

Smoke-test, locking the resolved package versions and HF commit SHAs into `MODELS.lock`:

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

The `del` + `empty_cache()` between passes frees the generation-model VRAM so the embedder has room to load.

## Tests

Unit tests stub the model boundaries — `outlines.from_transformers`, the wrapped model's `__call__`, and `sentence_transformers.SentenceTransformer.encode` — so no real model loads happen in CI:

```sh
cd services/semantic-enrich
pytest                        # or `uv run pytest`
ruff check .
mypy src
```

Real model loads happen as part of `semantic-enrich smoke-test --write-lock` on the GPU box.
