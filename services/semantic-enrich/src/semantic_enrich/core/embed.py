"""Sentence-transformers embeddings.

Returns L2-normalised 1024-dim vectors so downstream cosine similarity
reduces to dot product — the `semantic.*.embedding` columns commit to
that contract.
"""
from __future__ import annotations

from semantic_enrich.types import EmbeddingModel


def load_embedding_model(
    repo: str = "Qwen/Qwen3-Embedding-0.6B",
    *,
    device: str = "cuda",
    cache_dir: str | None = None,
) -> EmbeddingModel:
    """Load once; caller owns lifetime."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(repo, device=device, cache_folder=cache_dir)


def embed_batch(
    texts: list[str],
    *,
    model: EmbeddingModel,
    batch_size: int = 128,
) -> list[list[float]]:
    """Batch-encode. Returns L2-normalised 1024-dim vectors.

    `normalize_embeddings=True` is hard-coded — the
    `semantic.*.embedding` columns commit to L2-normalised vectors so
    cosine similarity reduces to dot product. Making this configurable
    would invite silent drift between writers and readers.
    """
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return [list(map(float, row)) for row in vectors]
