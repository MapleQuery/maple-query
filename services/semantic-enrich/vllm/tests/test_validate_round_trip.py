"""Unit tests for validate_round_trip.

Coverage target: 100% on `validate_round_trip.py` — the script is
the gate, and any uncovered branch is a hole in the gate.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pytest

import validate_round_trip as vrt
from tests.conftest import (
    FakeHttpClient,
    FakeOpenAI,
    unit_vector,
    valid_chat_payload,
)

# --- Generation path ---------------------------------------------------------

def test_generation_happy_path(gate_config, caplog):
    caplog.set_level(logging.INFO)
    fake_oa = FakeOpenAI(chat_content=valid_chat_payload())
    fake_http = FakeHttpClient(model_id="qwen2.5-14b-instruct")

    code = vrt.validate_generation(gate_config, openai_client=fake_oa,
                                   http_client=fake_http)
    assert code == vrt.EXIT_OK


def test_model_name_mismatch_generation(gate_config):
    """/v1/models returns the wrong model -> exit 2."""
    fake_oa = FakeOpenAI(chat_content=valid_chat_payload())
    fake_http = FakeHttpClient(model_id="qwen2.5-7b-instruct")

    code = vrt.validate_generation(gate_config, openai_client=fake_oa,
                                   http_client=fake_http)
    assert code == vrt.EXIT_PRECONDITION


def test_models_endpoint_unreachable_generation(gate_config):
    fake_oa = FakeOpenAI(chat_content=valid_chat_payload())
    fake_http = FakeHttpClient(raise_on_get=RuntimeError("connection refused"))

    code = vrt.validate_generation(gate_config, openai_client=fake_oa,
                                   http_client=fake_http)
    assert code == vrt.EXIT_PRECONDITION


def test_chat_request_failure(gate_config):
    fake_oa = FakeOpenAI(chat_content=RuntimeError("server 500"))
    fake_http = FakeHttpClient()

    code = vrt.validate_generation(gate_config, openai_client=fake_oa,
                                   http_client=fake_http)
    assert code == vrt.EXIT_PRECONDITION


def test_guided_json_unparseable(gate_config):
    """Completion returns raw text 'not JSON' -> exit 2."""
    fake_oa = FakeOpenAI(chat_content="not JSON")
    fake_http = FakeHttpClient()

    code = vrt.validate_generation(gate_config, openai_client=fake_oa,
                                   http_client=fake_http)
    assert code == vrt.EXIT_PRECONDITION


def test_guided_json_schema_violation(gate_config):
    """Missing 'summary' field -> exit 2."""
    payload = json.dumps({"package_id": "x"})
    fake_oa = FakeOpenAI(chat_content=payload)
    fake_http = FakeHttpClient()

    code = vrt.validate_generation(gate_config, openai_client=fake_oa,
                                   http_client=fake_http)
    assert code == vrt.EXIT_PRECONDITION


def test_guided_json_empty_strings_pass_schema_but_fail_strip(gate_config):
    """Whitespace-only strings would slip past minLength if the schema
    only counted characters; the strip-check catches them."""
    # pydantic enforces minLength=1; whitespace counts as a character,
    # so the schema accepts. The strip-check is the second line of
    # defense for this case.
    payload = json.dumps({"package_id": "ok-id", "summary": "   "})
    fake_oa = FakeOpenAI(chat_content=payload)
    fake_http = FakeHttpClient()

    code = vrt.validate_generation(gate_config, openai_client=fake_oa,
                                   http_client=fake_http)
    assert code == vrt.EXIT_PRECONDITION


def test_slow_round_trip_warns_but_passes(gate_config, monkeypatch, caplog):
    """A 70-second response -> exit 0, slow_round_trip warning emitted."""
    fake_oa = FakeOpenAI(chat_content=valid_chat_payload())
    fake_http = FakeHttpClient()

    # Patch time.monotonic to simulate a 70-second round trip without
    # actually sleeping.
    ticks = iter([1000.0, 1070.0])
    monkeypatch.setattr(vrt.time, "monotonic", lambda: next(ticks))

    caplog.set_level(logging.WARNING)
    code = vrt.validate_generation(gate_config, openai_client=fake_oa,
                                   http_client=fake_http)
    assert code == vrt.EXIT_OK


# --- Embedding path ---------------------------------------------------------

def test_embedding_happy_path(gate_config):
    fake_oa = FakeOpenAI(embedding_vector=unit_vector())
    fake_http = FakeHttpClient(model_id="qwen3-embedding-0.6b")

    code = vrt.validate_embedding(gate_config, openai_client=fake_oa,
                                  http_client=fake_http)
    assert code == vrt.EXIT_OK


def test_model_name_mismatch_embedding(gate_config):
    fake_oa = FakeOpenAI(embedding_vector=unit_vector())
    fake_http = FakeHttpClient(model_id="qwen3-embedding-1.5b")

    code = vrt.validate_embedding(gate_config, openai_client=fake_oa,
                                  http_client=fake_http)
    assert code == vrt.EXIT_PRECONDITION


def test_embedding_models_endpoint_unreachable(gate_config):
    fake_oa = FakeOpenAI(embedding_vector=unit_vector())
    fake_http = FakeHttpClient(raise_on_get=RuntimeError("connection refused"))

    code = vrt.validate_embedding(gate_config, openai_client=fake_oa,
                                  http_client=fake_http)
    assert code == vrt.EXIT_PRECONDITION


def test_embedding_request_failure(gate_config):
    fake_oa = FakeOpenAI(embedding_vector=RuntimeError("server 500"))
    fake_http = FakeHttpClient(model_id="qwen3-embedding-0.6b")

    code = vrt.validate_embedding(gate_config, openai_client=fake_oa,
                                  http_client=fake_http)
    assert code == vrt.EXIT_PRECONDITION


def test_dimension_check_rejects_wrong_dim(gate_config):
    """1025-dim vector -> exit 2."""
    wrong = list(np.zeros(1025))
    # Make it non-zero so this test exercises the dim check, not the
    # all-zeros guard.
    wrong[0] = 1.0
    fake_oa = FakeOpenAI(embedding_vector=wrong)
    fake_http = FakeHttpClient(model_id="qwen3-embedding-0.6b")

    code = vrt.validate_embedding(gate_config, openai_client=fake_oa,
                                  http_client=fake_http)
    assert code == vrt.EXIT_PRECONDITION


def test_dimension_check_accepts_1024(gate_config):
    """1024-dim vector is accepted."""
    fake_oa = FakeOpenAI(embedding_vector=unit_vector(1024))
    fake_http = FakeHttpClient(model_id="qwen3-embedding-0.6b")

    code = vrt.validate_embedding(gate_config, openai_client=fake_oa,
                                  http_client=fake_http)
    assert code == vrt.EXIT_OK


def test_norm_check_rejects_half_norm(gate_config):
    """Vector of norm 0.5 -> exit 2."""
    v = np.asarray(unit_vector(1024)) * 0.5
    fake_oa = FakeOpenAI(embedding_vector=v.tolist())
    fake_http = FakeHttpClient(model_id="qwen3-embedding-0.6b")

    code = vrt.validate_embedding(gate_config, openai_client=fake_oa,
                                  http_client=fake_http)
    assert code == vrt.EXIT_PRECONDITION


def test_norm_check_accepts_near_one(gate_config):
    """Norm = 0.9995 is within tolerance."""
    v = np.asarray(unit_vector(1024)) * 0.9995
    fake_oa = FakeOpenAI(embedding_vector=v.tolist())
    fake_http = FakeHttpClient(model_id="qwen3-embedding-0.6b")

    code = vrt.validate_embedding(gate_config, openai_client=fake_oa,
                                  http_client=fake_http)
    assert code == vrt.EXIT_OK


def test_zero_vector_check(gate_config):
    """All-zeros vector -> exit 2."""
    fake_oa = FakeOpenAI(embedding_vector=list(np.zeros(1024)))
    fake_http = FakeHttpClient(model_id="qwen3-embedding-0.6b")

    code = vrt.validate_embedding(gate_config, openai_client=fake_oa,
                                  http_client=fake_http)
    assert code == vrt.EXIT_PRECONDITION


def test_embedding_slow_round_trip_warns(gate_config, monkeypatch):
    """A 6-second embedding call passes but emits slow_round_trip."""
    fake_oa = FakeOpenAI(embedding_vector=unit_vector(1024))
    fake_http = FakeHttpClient(model_id="qwen3-embedding-0.6b")

    ticks = iter([100.0, 106.0])
    monkeypatch.setattr(vrt.time, "monotonic", lambda: next(ticks))

    code = vrt.validate_embedding(gate_config, openai_client=fake_oa,
                                  http_client=fake_http)
    assert code == vrt.EXIT_OK


# --- run() dispatch ---------------------------------------------------------

def test_run_unknown_target():
    code = vrt.run("nonsense")
    assert code == vrt.EXIT_PRECONDITION


def test_run_dispatches_generation(monkeypatch):
    called = {"gen": 0, "emb": 0}
    monkeypatch.setattr(vrt, "validate_generation",
                        lambda *_a, **_kw: called.update(gen=called["gen"] + 1) or vrt.EXIT_OK)
    monkeypatch.setattr(vrt, "validate_embedding",
                        lambda *_a, **_kw: called.update(emb=called["emb"] + 1) or vrt.EXIT_OK)
    assert vrt.run("generation") == vrt.EXIT_OK
    assert called == {"gen": 1, "emb": 0}


def test_run_dispatches_embedding(monkeypatch):
    called = {"gen": 0, "emb": 0}
    monkeypatch.setattr(vrt, "validate_generation",
                        lambda *_a, **_kw: called.update(gen=called["gen"] + 1) or vrt.EXIT_OK)
    monkeypatch.setattr(vrt, "validate_embedding",
                        lambda *_a, **_kw: called.update(emb=called["emb"] + 1) or vrt.EXIT_OK)
    assert vrt.run("embedding") == vrt.EXIT_OK
    assert called == {"gen": 0, "emb": 1}


def test_run_both_stops_on_generation_failure(monkeypatch):
    monkeypatch.setattr(vrt, "validate_generation",
                        lambda *_a, **_kw: vrt.EXIT_PRECONDITION)
    sentinel = {"emb_called": False}

    def _emb(*_a, **_kw):
        sentinel["emb_called"] = True
        return vrt.EXIT_OK
    monkeypatch.setattr(vrt, "validate_embedding", _emb)

    assert vrt.run("both") == vrt.EXIT_PRECONDITION
    assert sentinel["emb_called"] is False


def test_run_both_passes_when_both_pass(monkeypatch):
    monkeypatch.setattr(vrt, "validate_generation", lambda *_a, **_kw: vrt.EXIT_OK)
    monkeypatch.setattr(vrt, "validate_embedding", lambda *_a, **_kw: vrt.EXIT_OK)
    assert vrt.run("both") == vrt.EXIT_OK


def test_run_both_surfaces_embedding_failure(monkeypatch):
    monkeypatch.setattr(vrt, "validate_generation", lambda *_a, **_kw: vrt.EXIT_OK)
    monkeypatch.setattr(vrt, "validate_embedding",
                        lambda *_a, **_kw: vrt.EXIT_PRECONDITION)
    assert vrt.run("both") == vrt.EXIT_PRECONDITION


# --- main() CLI -------------------------------------------------------------

def test_main_passes_through_run(monkeypatch):
    monkeypatch.setattr(vrt, "run", lambda _t: vrt.EXIT_OK)
    assert vrt.main(["--target", "generation"]) == vrt.EXIT_OK


def test_main_handles_internal_error(monkeypatch):
    def _boom(_target):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(vrt, "run", _boom)
    assert vrt.main(["--target", "both"]) == vrt.EXIT_INTERNAL_ERROR


def test_main_propagates_system_exit(monkeypatch):
    def _exit(_target):
        raise SystemExit(7)
    monkeypatch.setattr(vrt, "run", _exit)
    with pytest.raises(SystemExit) as excinfo:
        vrt.main(["--target", "both"])
    assert excinfo.value.code == 7


# --- Config from env --------------------------------------------------------

def test_config_from_env_defaults(monkeypatch):
    for k in (
        "WHENRICH_GENERATION_BASE_URL", "WHENRICH_EMBEDDING_BASE_URL",
        "WHENRICH_GENERATION_MODEL", "WHENRICH_EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(k, raising=False)
    cfg = vrt._config_from_env()
    assert cfg.generation_base_url == "http://127.0.0.1:8001"
    assert cfg.embedding_base_url == "http://127.0.0.1:8002"
    assert cfg.generation_model == "qwen2.5-14b-instruct"
    assert cfg.embedding_model == "qwen3-embedding-0.6b"


def test_config_from_env_overrides(monkeypatch):
    monkeypatch.setenv("WHENRICH_GENERATION_BASE_URL", "http://gpu-host:8001/")
    monkeypatch.setenv("WHENRICH_EMBEDDING_BASE_URL", "http://gpu-host:8002/")
    monkeypatch.setenv("WHENRICH_GENERATION_MODEL", "custom-gen")
    monkeypatch.setenv("WHENRICH_EMBEDDING_MODEL", "custom-emb")
    cfg = vrt._config_from_env()
    assert cfg.generation_base_url == "http://gpu-host:8001"  # trailing slash stripped
    assert cfg.embedding_base_url == "http://gpu-host:8002"
    assert cfg.generation_model == "custom-gen"
    assert cfg.embedding_model == "custom-emb"


def test_models_endpoint_id_empty_data_raises(gate_config):
    class EmptyHttp:
        def get(self, _url, timeout=10.0):
            from tests.conftest import _FakeResponse
            return _FakeResponse(status=200, body={"data": []})

    with pytest.raises(RuntimeError, match="empty data"):
        vrt._models_endpoint_id(gate_config.generation_base_url, EmptyHttp())


def test_configure_logging_smoke():
    """Just exercise the configure_logging call path."""
    vrt._configure_logging()
