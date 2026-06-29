# services/semantic-enrich

Semantic-enrich is the pipeline that turns the raw catalog (`raw.documents`, `raw.rows`, `raw.column_index`) into the semantic layer (`semantic.datasets`, `semantic.columns`) — per-package summaries plus per-column embeddings.

This directory currently ships only the **vLLM model-serving stack** the pipeline depends on. The text-gen + embedding passes themselves land in a later change.

```
services/semantic-enrich/
└── vllm/
    ├── scripts/                  # bash launchers, swap procedure, killswitch
    │   ├── serve_generation.sh
    │   ├── serve_embedding.sh
    │   ├── swap_to_generation.sh
    │   ├── swap_to_embedding.sh
    │   ├── kill_vllm.sh
    │   ├── wait_for_ready.sh
    │   └── wait_for_vram_clear.sh
    ├── tests/                    # pytest unit tests for validate_round_trip
    ├── validate_round_trip.py    # round-trip validation gate
    ├── MODELS.lock               # pinned vllm version + HF commit SHAs (operator-populated)
    ├── pyproject.toml            # validate-script deps only (NOT vLLM)
    └── .gitignore                # logs/, *.pid, venvs
```

## What the stack serves

Two vLLM servers, **one resident at a time** on the GPU:

| Server      | Port  | Bind        | Model                          | Dtype    | max_model_len |
| --          | --    | --          | --                             | --       | --            |
| generation  | 8001  | 127.0.0.1   | `Qwen/Qwen2.5-14B-Instruct`    | bfloat16 | 8192          |
| embedding   | 8002  | 127.0.0.1   | `Qwen/Qwen3-Embedding-0.6B`    | float16  | 8192          |

Both are exposed as OpenAI-compatible HTTP (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`). The generation server runs with `--guided-decoding-backend outlines` so the text-gen pass gets schema-conforming JSON without parse-repair logic.

The swap procedure (`swap_to_*.sh`) tears down the resident server, polls `nvidia-smi` until VRAM clears, starts the other server in the background, and waits for `/v1/models` to return 200. The single-resident-at-a-time rule is enforced by the GPU memory budget — bf16-14B + KV cache + CUDA workspace alone consumes most of a 48 GB card.

Two distinct ports are used (instead of reusing one) so the consumer reads two stable env vars at process start and never needs to re-resolve URLs across a swap.

## The validation gate

`validate_round_trip.py` is the entry point of every backfill. It exits non-zero if:

- `/v1/models` is unreachable or returns the wrong `data[0].id` (model name drift after a misconfigured restart).
- The generation server's guided-JSON response isn't parseable, doesn't match the synthetic test schema, or comes back with whitespace-only required fields.
- The embedding vector isn't 1024-dim, isn't L2-normalised (within 0.01), or is all zeros.

Latency above the per-surface threshold (60s generation, 5s embedding) is a warning, not a failure — first-request CUDA-graph JIT can be slow.

Exit codes:

- `0` — success.
- `2` — precondition failure (the gate refuses to proceed). The expected mode when the gate fires.
- `1` — unexpected internal error.

The gate uses a **synthetic** schema (`{package_id, summary}`) rather than the real `semantic.datasets` schema. Coupling the gate to a production schema that doesn't exist yet would expand its failure surface; structural correctness is the only concern at this layer.

## Configuration surface

The downstream semantic-enrich pipeline consumes the stack via four env vars, prefixed `WHENRICH_` to match the repo's pattern:

| Env var                          | Default                    | Meaning |
| --                               | --                         | --      |
| `WHENRICH_GENERATION_BASE_URL`   | `http://127.0.0.1:8001`    | Append `/v1` for the OpenAI SDK. |
| `WHENRICH_GENERATION_MODEL`      | `qwen2.5-14b-instruct`     | `model=` field on chat-completion requests. |
| `WHENRICH_EMBEDDING_BASE_URL`    | `http://127.0.0.1:8002`    |         |
| `WHENRICH_EMBEDDING_MODEL`       | `qwen3-embedding-0.6b`     | `model=` field on embedding requests. |

The launchers also accept per-flag overrides via `WHENRICH_GENERATION_DTYPE`, `WHENRICH_GENERATION_MAX_MODEL_LEN`, `WHENRICH_GENERATION_GPU_MEM_UTIL`, `WHENRICH_GUIDED_DECODING_BACKEND`, and the embedding equivalents. Defaults are pinned to the bf16 / fp16 settings above; the env overrides are the escape hatch for the AWQ-fallback path on a 24 GB workstation card.

## Two venvs by design

- **`venv/`** — vLLM server venv. CUDA-12.4 wheels, flash-attn, ~10 GB. Operator builds this once on the GPU box.
- **`validate-venv/`** — validate-script venv. ~30 MB, pure HTTP/JSON. No CUDA dependency, fast cold start, runs anywhere.

The split is deliberate. The gate's only job is to talk HTTP; dragging CUDA into its dependency closure would couple validation tooling to GPU-host-only execution and slow operator iteration.

Both directories are gitignored.

## Operator runbook

### First-time setup (on the GPU box)

```sh
cd services/semantic-enrich/vllm

# 1. vLLM server venv. CUDA 12.4 wheels are the assumption; adjust
#    the extra-index-url per your toolkit.
uv venv --python 3.12 venv
source venv/bin/activate
uv pip install 'vllm>=0.6.0,<0.7' \
  --extra-index-url https://download.pytorch.org/whl/cu124

# 2. Pre-pull both models so the first launch doesn't spend 5–15
#    minutes downloading weights.
huggingface-cli download Qwen/Qwen2.5-14B-Instruct
huggingface-cli download Qwen/Qwen3-Embedding-0.6B

# 3. Capture resolved versions in MODELS.lock.
python -c "
import vllm, huggingface_hub
api = huggingface_hub.HfApi()
print('vllm:', vllm.__version__)
print('gen:',  api.model_info('Qwen/Qwen2.5-14B-Instruct').sha)
print('emb:',  api.model_info('Qwen/Qwen3-Embedding-0.6B').sha)
" > MODELS.lock
deactivate

# 4. Validation venv (separate).
uv venv --python 3.12 validate-venv
source validate-venv/bin/activate
uv pip install -e '.[dev]'
deactivate
```

### Daily startup (generation pass)

```sh
cd services/semantic-enrich/vllm

# Background-start generation.
nohup ./scripts/serve_generation.sh > /dev/null 2>&1 &

# Wait for /v1/models.
./scripts/wait_for_ready.sh 8001 180

# Validate.
source validate-venv/bin/activate
python validate_round_trip.py --target generation
```

### Mid-run swap to embedding

```sh
./scripts/swap_to_embedding.sh
python validate_round_trip.py --target embedding
```

### Teardown

```sh
./scripts/kill_vllm.sh all
```

### tmux (convenience, not load-bearing)

```sh
tmux new -s vllm
# pane 0:
./scripts/serve_generation.sh
# Ctrl-b d to detach. Reattach: tmux attach -t vllm
```

The launchers always log to `services/semantic-enrich/vllm/logs/<server>-<run_id>.log`, so tmux is only useful when you want to watch live.

## Failure modes

- **OOM on load.** Lower `--gpu-memory-utilization` (env: `WHENRICH_GENERATION_GPU_MEM_UTIL`), then `--max-model-len`, then fall back to `Qwen/Qwen2.5-14B-Instruct-AWQ` (set `WHENRICH_GENERATION_MODEL_REPO`, `WHENRICH_GENERATION_DTYPE=float16`, etc.).
- **Wedged process.** `./scripts/kill_vllm.sh all && ./scripts/wait_for_vram_clear.sh 512 60` and inspect the crash log under `logs/`.
- **Embedding dimension drift.** The gate exits `2` with `embedding_dim_mismatch`. Do not weaken the gate — the `semantic.*.embedding` columns are 1024-dim by contract. Either revert the model swap or land a schema PR moving all rows to the new dimension before re-running.
- **Cold model download stalls `wait_for_ready`.** Pre-pull weights via `huggingface-cli download` (runbook step 2), or raise the timeout once: `./scripts/wait_for_ready.sh 8001 900`.

## Layering and tests

The validate script is a single module — no `core/clients/providers` layering needed. Tests live in `tests/` and use hand-rolled fakes (no HTTP, no openai client). Coverage target is 100% on `validate_round_trip.py`; `pytest --cov` is configured to fail under that line.

Bash scripts pass `shellcheck` and `bash -n`. Integration testing is the runbook walkthrough on the GPU box, recorded in `MODELS.lock` and the launcher logs.

## Decisions

- **Local-machine first.** No Docker, no Cloud Run, no systemd. The deliverable is scripts + a runbook the operator runs on a workstation or a GPU box they ssh into. Productionisation is a later milestone once the agent layer needs sustained uptime.
- **Single GPU, single resident model.** Co-residency is infeasible on 24 GB and tight on 48 GB. The swap is the answer.
- **Two distinct ports.** Stable client config across the swap is worth a free port number.
- **127.0.0.1, never 0.0.0.0.** vLLM exposes model weights and a GPU. Cross-machine access goes through SSH port-forwarding, not an open bind.
- **Validation gate runs every backfill.** Three seconds of gate prevents hours of polluted JSONL staging on a mis-launched server.
- **Synthetic prompt for the gate.** The gate's job is structural correctness, not semantic quality. Coupling it to production prompts (which don't exist yet) would expand its failure surface for no gain.
