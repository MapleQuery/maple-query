"""Round-trip validation gate for the semantic-enrich vLLM stack.

Single CLI surface:

    python validate_round_trip.py --target generation
    python validate_round_trip.py --target embedding
    python validate_round_trip.py --target both     # default

Exit codes:

    0  full success
    2  precondition failure (server unreachable, wrong model name,
       guided-decoding misconfigured, dimension/norm drift, etc.)
    1  unexpected internal error

The gate is the entry point of every semantic-enrich backfill. It
catches a broken server before the 3.7K-package pass stages a
single JSONL row.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np
import openai
import pydantic
import structlog

EXIT_OK = 0
EXIT_INTERNAL_ERROR = 1
EXIT_PRECONDITION = 2

GENERATION_DEFAULT_BASE_URL = "http://127.0.0.1:8001"
EMBEDDING_DEFAULT_BASE_URL = "http://127.0.0.1:8002"
GENERATION_MODEL_NAME = "qwen2.5-14b-instruct"
EMBEDDING_MODEL_NAME = "qwen3-embedding-0.6b"

EMBEDDING_DIM = 1024
EMBEDDING_NORM_TOLERANCE = 0.01
GENERATION_SLOW_THRESHOLD_MS = 60_000
EMBEDDING_SLOW_THRESHOLD_MS = 5_000

SYSTEM_PROMPT = (
    "You are a strict JSON emitter. Respond with one JSON object "
    "conforming exactly to the schema you have been given. No prose."
)
USER_PROMPT = (
    "Return a JSON object with a fake package_id (any UUID-like "
    "string) and a one-sentence summary of the following dataset: "
    "'Quarterly CPI by province, Statistics Canada, 2010-2024.'"
)

VALIDATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "package_id": {"type": "string", "minLength": 1},
        "summary":    {"type": "string", "minLength": 1, "maxLength": 500},
    },
    "required": ["package_id", "summary"],
    "additionalProperties": False,
}


class ValidationResponse(pydantic.BaseModel):
    """Pydantic mirror of VALIDATION_SCHEMA for response validation."""

    model_config = pydantic.ConfigDict(extra="forbid")

    package_id: str = pydantic.Field(min_length=1)
    summary: str = pydantic.Field(min_length=1, max_length=500)


@dataclass(frozen=True)
class GateConfig:
    generation_base_url: str
    embedding_base_url: str
    generation_model: str
    embedding_model: str


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
        cache_logger_on_first_use=True,
    )


def _config_from_env() -> GateConfig:
    return GateConfig(
        generation_base_url=os.environ.get(
            "WHENRICH_GENERATION_BASE_URL", GENERATION_DEFAULT_BASE_URL
        ).rstrip("/"),
        embedding_base_url=os.environ.get(
            "WHENRICH_EMBEDDING_BASE_URL", EMBEDDING_DEFAULT_BASE_URL
        ).rstrip("/"),
        generation_model=os.environ.get(
            "WHENRICH_GENERATION_MODEL", GENERATION_MODEL_NAME
        ),
        embedding_model=os.environ.get(
            "WHENRICH_EMBEDDING_MODEL", EMBEDDING_MODEL_NAME
        ),
    )


def _models_endpoint_id(base_url: str, http_client: httpx.Client) -> str:
    response = http_client.get(f"{base_url}/v1/models", timeout=10.0)
    response.raise_for_status()
    body = response.json()
    data = body.get("data") or []
    if not data:
        raise RuntimeError(f"{base_url}/v1/models returned empty data")
    return str(data[0]["id"])


def validate_generation(
    config: GateConfig,
    *,
    openai_client: openai.OpenAI | None = None,
    http_client: httpx.Client | None = None,
    log: structlog.stdlib.BoundLogger | None = None,
) -> int:
    """Round-trip the generation server. Returns an exit code."""
    log = log or structlog.get_logger()
    http_client = http_client or httpx.Client()
    openai_client = openai_client or openai.OpenAI(
        base_url=f"{config.generation_base_url}/v1",
        api_key="EMPTY",
    )

    # Step 1: /v1/models — assert the right model is served.
    try:
        actual_id = _models_endpoint_id(config.generation_base_url, http_client)
    except Exception as exc:
        log.error("generation_models_endpoint_unreachable",
                  base_url=config.generation_base_url, error=str(exc))
        return EXIT_PRECONDITION
    if actual_id != config.generation_model:
        log.error("model_name_mismatch", surface="generation",
                  expected=config.generation_model, actual=actual_id)
        return EXIT_PRECONDITION

    # Steps 2-3: issue the guided-JSON prompt; parse the response.
    t0 = time.monotonic()
    try:
        response = openai_client.chat.completions.create(
            model=config.generation_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_PROMPT},
            ],
            temperature=0.0,
            max_tokens=400,
            extra_body={"guided_json": VALIDATION_SCHEMA},
        )
    except Exception as exc:
        log.error("generation_request_failed", error=str(exc))
        return EXIT_PRECONDITION
    duration_ms = int((time.monotonic() - t0) * 1000)

    content = response.choices[0].message.content or ""
    try:
        obj = json.loads(content)
    except json.JSONDecodeError as exc:
        log.error("guided_json_unparseable", error=str(exc),
                  raw=content[:500])
        return EXIT_PRECONDITION

    # Step 4: pydantic schema validation.
    try:
        parsed = ValidationResponse.model_validate(obj)
    except pydantic.ValidationError as exc:
        log.error("guided_json_schema_violation", error=str(exc), obj=obj)
        return EXIT_PRECONDITION

    # Step 5: non-empty after strip.
    if not parsed.package_id.strip() or not parsed.summary.strip():
        log.error("guided_json_empty_after_strip",
                  package_id=parsed.package_id, summary=parsed.summary)
        return EXIT_PRECONDITION

    # Step 6: latency warning (not a hard failure).
    if duration_ms > GENERATION_SLOW_THRESHOLD_MS:
        log.warning("slow_round_trip", surface="generation",
                    duration_ms=duration_ms,
                    threshold_ms=GENERATION_SLOW_THRESHOLD_MS)
    log.info("generation_validation_ok", duration_ms=duration_ms)
    return EXIT_OK


def validate_embedding(
    config: GateConfig,
    *,
    openai_client: openai.OpenAI | None = None,
    http_client: httpx.Client | None = None,
    log: structlog.stdlib.BoundLogger | None = None,
) -> int:
    """Round-trip the embedding server. Returns an exit code."""
    log = log or structlog.get_logger()
    http_client = http_client or httpx.Client()
    openai_client = openai_client or openai.OpenAI(
        base_url=f"{config.embedding_base_url}/v1",
        api_key="EMPTY",
    )

    # Step 1: /v1/models name check.
    try:
        actual_id = _models_endpoint_id(config.embedding_base_url, http_client)
    except Exception as exc:
        log.error("embedding_models_endpoint_unreachable",
                  base_url=config.embedding_base_url, error=str(exc))
        return EXIT_PRECONDITION
    if actual_id != config.embedding_model:
        log.error("model_name_mismatch", surface="embedding",
                  expected=config.embedding_model, actual=actual_id)
        return EXIT_PRECONDITION

    # Step 2: one embedding request.
    t0 = time.monotonic()
    try:
        response = openai_client.embeddings.create(
            model=config.embedding_model,
            input="The federal government published quarterly CPI by province.",
        )
    except Exception as exc:
        log.error("embedding_request_failed", error=str(exc))
        return EXIT_PRECONDITION
    duration_ms = int((time.monotonic() - t0) * 1000)

    vector = np.asarray(response.data[0].embedding, dtype=np.float64)

    # Step 3: dimension. The semantic.*.embedding columns commit to
    # 1024-dim ARRAY<FLOAT64>; a silent drift here would corrupt a
    # 153K-row backfill at bq load time.
    if vector.shape != (EMBEDDING_DIM,):
        log.error("embedding_dim_mismatch",
                  expected=EMBEDDING_DIM, actual=int(vector.shape[0]))
        return EXIT_PRECONDITION

    # Step 4: degenerate-vector guard. An all-zeros response means
    # the model didn't actually run inference (e.g. a test/mock path
    # slipped in). Checked before the norm assertion so the failure
    # event names the real cause instead of "norm 0.0 != 1.0".
    if np.allclose(vector, 0.0):
        log.error("embedding_all_zero_vector")
        return EXIT_PRECONDITION

    # Step 5: L2 norm. Qwen3 embeddings are normalised by the model;
    # a norm far from 1.0 means a model swap broke that invariant.
    norm = float(np.linalg.norm(vector))
    if abs(norm - 1.0) > EMBEDDING_NORM_TOLERANCE:
        log.error("embedding_norm_drift", norm=norm,
                  tolerance=EMBEDDING_NORM_TOLERANCE)
        return EXIT_PRECONDITION

    # Step 6: latency warning.
    if duration_ms > EMBEDDING_SLOW_THRESHOLD_MS:
        log.warning("slow_round_trip", surface="embedding",
                    duration_ms=duration_ms,
                    threshold_ms=EMBEDDING_SLOW_THRESHOLD_MS)
    log.info("embedding_validation_ok", duration_ms=duration_ms, norm=norm)
    return EXIT_OK


def run(target: str) -> int:
    """Top-level gate entrypoint. Returns the process exit code."""
    config = _config_from_env()
    log = structlog.get_logger()

    if target == "generation":
        return validate_generation(config, log=log)
    if target == "embedding":
        return validate_embedding(config, log=log)
    if target == "both":
        gen_code = validate_generation(config, log=log)
        if gen_code != EXIT_OK:
            return gen_code
        emb_code = validate_embedding(config, log=log)
        if emb_code != EXIT_OK:
            return emb_code
        log.info("validation_gate_passed")
        return EXIT_OK

    log.error("unknown_target", target=target)
    return EXIT_PRECONDITION


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=("generation", "embedding", "both"),
        default="both",
        help="Which server(s) to validate (default: both).",
    )
    args = parser.parse_args(argv)

    _configure_logging()
    log = structlog.get_logger()
    try:
        return run(args.target)
    except SystemExit:
        raise
    except Exception as exc:
        log.error("validation_gate_internal_error", error=str(exc),
                  error_type=type(exc).__name__)
        return EXIT_INTERNAL_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
