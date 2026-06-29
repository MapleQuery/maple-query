"""Deploy-time configuration loaded from environment variables.

Prefix `WHENRICH_` matches the WHLOAD_ / WHINGEST_ pattern in sibling
services. Per-run intent (the smoke-test flags) is CLI-only.
"""
from __future__ import annotations

from pathlib import Path

from dotenv import find_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WHENRICH_",
        env_file=find_dotenv(usecwd=True) or None,
        extra="ignore",
    )

    generation_model: str = "Qwen/Qwen2.5-14B-Instruct"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    device: str = "cuda"

    # Optional HF cache override. Falls through to the HF default
    # (`$HF_HOME` or `~/.cache/huggingface`) when unset.
    hf_cache_dir: Path | None = None
