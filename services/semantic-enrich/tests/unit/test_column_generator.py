"""Unit tests for `core.column_generator`.

Covers chunking properties (§7.2), per-chunk and per-package
invariant validators (§8.1-§8.2), and the safety-belt cap (§7.4).
"""
from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from semantic_enrich.core import column_generator
from semantic_enrich.types import (
    ColumnChunkInvariantError,
    ColumnInputs,
    ColumnOutput,
    ColumnPackageInvariantError,
)


def _inputs(column_names: tuple[str, ...]) -> ColumnInputs:
    return ColumnInputs(
        package_id="pkg-1",
        package_title="t",
        package_description="d",
        package_subjects=(),
        package_summary=None,
        representative_document_id="doc",
        column_names=column_names,
        sample_values=dict.fromkeys(column_names, ()),
        dropped_columns=(),
        overflow_column_count=0,
        extracted_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _output(name: str) -> ColumnOutput:
    return ColumnOutput(
        column_name=name,
        semantic_type="text",
        description="x" * 40,
        sample_values=[],
    )


# ── chunk_columns properties ──


@pytest.mark.parametrize(
    "n,chunk_size,expected",
    [
        (0, 100, 0),
        (1, 100, 1),
        (100, 100, 1),
        (101, 100, 2),
        (250, 100, 3),
        (1383, 100, 14),
        (50, 50, 1),
        (50, 25, 2),
    ],
)
def test_chunk_count_equals_ceil(n: int, chunk_size: int, expected: int) -> None:
    inputs = _inputs(tuple(f"c{i}" for i in range(n)))
    chunks = list(
        column_generator.chunk_columns(inputs=inputs, chunk_size=chunk_size)
    )
    assert len(chunks) == expected
    assert expected == (math.ceil(n / chunk_size) if n else 0)


def test_chunk_concat_round_trip() -> None:
    inputs = _inputs(tuple(f"c{i}" for i in range(255)))
    chunks = list(column_generator.chunk_columns(inputs=inputs, chunk_size=100))
    concat = [n for c in chunks for n in c.column_names]
    assert concat == list(inputs.column_names)


def test_chunk_carries_sample_values_per_column() -> None:
    inputs = ColumnInputs(
        package_id="pkg",
        package_title=None,
        package_description=None,
        package_subjects=(),
        package_summary=None,
        representative_document_id="doc",
        column_names=("a", "b"),
        sample_values={"a": ("1", "2"), "b": ("3",)},
        dropped_columns=(),
        overflow_column_count=0,
        extracted_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    chunks = list(column_generator.chunk_columns(inputs=inputs, chunk_size=100))
    assert len(chunks) == 1
    assert chunks[0].sample_values == {"a": ["1", "2"], "b": ["3"]}


def test_chunk_last_chunk_may_be_short() -> None:
    inputs = _inputs(tuple(f"c{i}" for i in range(7)))
    chunks = list(column_generator.chunk_columns(inputs=inputs, chunk_size=3))
    assert [len(c.column_names) for c in chunks] == [3, 3, 1]


# ── validate_chunk_output ──


def _chunk(names: list[str]) -> column_generator._ColumnChunk:
    return column_generator._ColumnChunk(
        package_id="pkg",
        chunk_index=0,
        chunk_count=1,
        column_names=names,
        sample_values={n: [] for n in names},
    )


def test_validate_chunk_accepts_clean_response() -> None:
    chunk = _chunk(["a", "b"])
    response = [
        {"column_name": "a", "semantic_type": "text",
         "description": "x" * 40, "sample_values": []},
        {"column_name": "b", "semantic_type": "text",
         "description": "y" * 40, "sample_values": []},
    ]
    outputs = column_generator.validate_chunk_output(chunk, response)
    assert [o.column_name for o in outputs] == ["a", "b"]


def test_validate_chunk_rejects_length_mismatch() -> None:
    chunk = _chunk(["a", "b"])
    response = [
        {"column_name": "a", "description": "x" * 40},
    ]
    with pytest.raises(ColumnChunkInvariantError, match="returned 1 entries"):
        column_generator.validate_chunk_output(chunk, response)


def test_validate_chunk_rejects_name_mismatch() -> None:
    chunk = _chunk(["a", "b"])
    response = [
        {"column_name": "a", "description": "x" * 40},
        {"column_name": "c", "description": "y" * 40},
    ]
    with pytest.raises(ColumnChunkInvariantError, match="position 1"):
        column_generator.validate_chunk_output(chunk, response)


def test_validate_chunk_rejects_short_description() -> None:
    chunk = _chunk(["a"])
    response = [{"column_name": "a", "description": "short"}]
    with pytest.raises(ColumnChunkInvariantError, match="pydantic"):
        column_generator.validate_chunk_output(chunk, response)


# ── validate_package_output ──


def test_validate_package_accepts_clean_concat() -> None:
    inputs = _inputs(("a", "b", "c"))
    outputs = [_output("a"), _output("b"), _output("c")]
    column_generator.validate_package_output(
        inputs=inputs, package_outputs=outputs
    )


def test_validate_package_rejects_mid_sequence_divergence() -> None:
    inputs = _inputs(("a", "b", "c"))
    outputs = [_output("a"), _output("X"), _output("c")]
    with pytest.raises(ColumnPackageInvariantError, match="position 1"):
        column_generator.validate_package_output(
            inputs=inputs, package_outputs=outputs
        )


def test_validate_package_rejects_length_mismatch() -> None:
    inputs = _inputs(("a", "b", "c"))
    outputs = [_output("a"), _output("b")]
    with pytest.raises(ColumnPackageInvariantError, match="have 2 entries"):
        column_generator.validate_package_output(
            inputs=inputs, package_outputs=outputs
        )
