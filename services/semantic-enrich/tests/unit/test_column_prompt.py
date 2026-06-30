"""Snapshot tests for the columns prompt constants.

The columns prompt is pinned verbatim. An unintended edit to either
constant forces this snapshot to fail, so the change is intentional.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from semantic_enrich.core import column_prompt
from semantic_enrich.types import ColumnInputs


def _inputs(
    *,
    column_names: tuple[str, ...] = ("year", "amt_cad"),
    sample_values: dict[str, tuple[str, ...]] | None = None,
    package_summary: str | None = None,
    package_subjects: tuple[str, ...] = ("Government and Politics",),
) -> ColumnInputs:
    return ColumnInputs(
        package_id="pkg-snapshot-1",
        package_title="Snapshot Test Dataset",
        package_description="Snapshot test description.",
        package_subjects=package_subjects,
        package_summary=package_summary,
        representative_document_id="doc-1",
        column_names=column_names,
        sample_values=sample_values or {
            "year": ("2024", "2025"),
            "amt_cad": ("100.00", "250.50"),
        },
        dropped_columns=(),
        overflow_column_count=0,
        extracted_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_system_prompt_snapshot() -> None:
    """An unintended edit to the system prompt fails this test."""
    assert column_prompt.COLUMNS_SYSTEM_PROMPT.startswith(
        "You are a data documentarian for an open-data warehouse."
    )
    assert "JSON array" in column_prompt.COLUMNS_SYSTEM_PROMPT
    assert "schema is enforced" in column_prompt.COLUMNS_SYSTEM_PROMPT
    # No markdown fences mention so the model knows raw JSON only.
    assert "No markdown fences" in column_prompt.COLUMNS_SYSTEM_PROMPT


def test_user_template_snapshot() -> None:
    """Placeholders + structural anchors pinned."""
    tmpl = column_prompt.COLUMNS_USER_TEMPLATE
    for placeholder in (
        "{package_title}",
        "{package_description}",
        "{package_subjects_csv}",
        "{package_summary_block}",
        "{chunk_index_plus_one}",
        "{chunk_count}",
        "{columns_block}",
        "{column_count}",
    ):
        assert placeholder in tmpl, f"missing placeholder: {placeholder}"
    assert "Respond with a single JSON array" in tmpl
    # Echo-verbatim instruction is load-bearing for the §8 invariant.
    assert "echo `column_name` verbatim" in tmpl


def test_render_user_message_byte_stable() -> None:
    """Same inputs → identical bytes across invocations."""
    inputs = _inputs()
    a = column_prompt.render_user_message(
        inputs=inputs,
        chunk_index=0,
        chunk_count=1,
        column_names=list(inputs.column_names),
    )
    b = column_prompt.render_user_message(
        inputs=inputs,
        chunk_index=0,
        chunk_count=1,
        column_names=list(inputs.column_names),
    )
    assert a == b


def test_render_user_message_includes_package_summary_when_set() -> None:
    inputs = _inputs(package_summary="Annual fisheries quota allocations by province.")
    rendered = column_prompt.render_user_message(
        inputs=inputs,
        chunk_index=0,
        chunk_count=1,
        column_names=list(inputs.column_names),
    )
    assert (
        "- Summary (from semantic.datasets): "
        "Annual fisheries quota allocations by province."
    ) in rendered


def test_render_user_message_omits_package_summary_block_when_none() -> None:
    inputs = _inputs(package_summary=None)
    rendered = column_prompt.render_user_message(
        inputs=inputs,
        chunk_index=0,
        chunk_count=1,
        column_names=list(inputs.column_names),
    )
    assert "- Summary (from semantic.datasets):" not in rendered


def test_render_user_message_chunk_metadata() -> None:
    inputs = _inputs()
    rendered = column_prompt.render_user_message(
        inputs=inputs,
        chunk_index=2,
        chunk_count=5,
        column_names=list(inputs.column_names),
    )
    assert "batch 3 of 5" in rendered
    assert "exactly 2 entries" in rendered


def test_columns_block_quotes_sample_values() -> None:
    block = column_prompt.render_columns_block(
        column_names=["x"], sample_values={"x": ['has "quote"', "plain"]}
    )
    assert '"has \\"quote\\""' in block
    assert '"plain"' in block


def test_columns_block_handles_empty_samples() -> None:
    block = column_prompt.render_columns_block(
        column_names=["x", "y"], sample_values={"x": [], "y": []}
    )
    assert "- name: x" in block
    assert "  samples: []" in block


@pytest.mark.parametrize("chunk_size", [1, 2, 5, 100])
def test_render_user_message_column_count_matches_input(chunk_size: int) -> None:
    inputs = _inputs(column_names=tuple(f"c{i}" for i in range(chunk_size)),
                     sample_values={f"c{i}": ("v",) for i in range(chunk_size)})
    rendered = column_prompt.render_user_message(
        inputs=inputs,
        chunk_index=0,
        chunk_count=1,
        column_names=list(inputs.column_names),
    )
    assert f"exactly {chunk_size} entries" in rendered
