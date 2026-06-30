"""Prompt template + schema constant."""
from __future__ import annotations

import pydantic
import pytest

from semantic_enrich.core.dataset_prompt import (
    SYSTEM_PROMPT,
    estimate_tokens,
    render_user_message,
)
from semantic_enrich.core.schemas import DATASET_CARD_GUIDED_JSON
from semantic_enrich.types import DatasetCard, PackageInputs, PackageResource


def _pkg() -> PackageInputs:
    return PackageInputs(
        package_id="pkg-001",
        resources=(
            PackageResource(
                document_id="doc-1",
                title="Quarterly Employment",
                description="Federal head-count by department.",
                subjects=("employment",),
                organization_code="StatCan",
                file_format="csv",
                resource_last_modified=None,
                row_count=500,
            ),
        ),
        column_names=("department", "fiscal_year", "headcount"),
        column_names_truncated_to=None,
        representative_document_id="doc-1",
        sample_rows=(
            {"department": "Foo", "fiscal_year": "2024", "headcount": "120"},
        ),
    )


def test_render_user_message_includes_package_id() -> None:
    out = render_user_message(_pkg())
    assert "package_id: pkg-001" in out


def test_render_user_message_lists_resource_titles() -> None:
    out = render_user_message(_pkg())
    assert "Quarterly Employment" in out


def test_render_user_message_truncation_note() -> None:
    pkg = _pkg().model_copy(
        update={
            "column_names": ("a", "b"),
            "column_names_truncated_to": 87,
        }
    )
    out = render_user_message(pkg)
    assert "truncated from 87" in out


def test_render_user_message_no_truncation_note_when_unset() -> None:
    out = render_user_message(_pkg())
    assert "truncated from" not in out


def test_render_user_message_no_columns() -> None:
    pkg = _pkg().model_copy(update={"column_names": ()})
    out = render_user_message(pkg)
    assert "(none)" in out


def test_render_user_message_no_sample_rows() -> None:
    pkg = _pkg().model_copy(update={"sample_rows": ()})
    out = render_user_message(pkg)
    assert "(no rows)" in out


def test_system_prompt_is_english_only_instruction() -> None:
    # Pinned wording — diff if you change it.
    assert "emit English only" in SYSTEM_PROMPT


def test_guided_json_shape() -> None:
    s = DATASET_CARD_GUIDED_JSON
    assert s["type"] == "object"
    assert s["additionalProperties"] is False
    assert set(s["required"]) == {"package_id", "summary"}
    assert s["properties"]["summary"]["minLength"] == 50
    assert s["properties"]["summary"]["maxLength"] == 1200
    assert s["properties"]["measures"]["maxItems"] == 20


def test_dataset_card_roundtrip() -> None:
    raw = {
        "package_id": "pkg-001",
        "summary": "A" * 60,
        "grain": "row",
        "measures": ["x"],
        "dimensions": ["y"],
        "date_range_start": None,
        "date_range_end": None,
    }
    card = DatasetCard.model_validate(raw)
    assert card.package_id == "pkg-001"
    assert card.grain == "row"


def test_dataset_card_empty_grain_becomes_none() -> None:
    raw = {
        "package_id": "pkg-001",
        "summary": "A" * 60,
        "grain": "",
    }
    assert DatasetCard.model_validate(raw).grain is None


def test_dataset_card_short_summary_rejected() -> None:
    with pytest.raises(pydantic.ValidationError):
        DatasetCard.model_validate(
            {"package_id": "pkg-001", "summary": "too short"}
        )


def test_dataset_card_too_many_measures_rejected() -> None:
    with pytest.raises(pydantic.ValidationError):
        DatasetCard.model_validate(
            {
                "package_id": "pkg-001",
                "summary": "A" * 60,
                "measures": [f"m{i}" for i in range(21)],
            }
        )


def test_estimate_tokens_monotonic() -> None:
    assert estimate_tokens("hi") <= estimate_tokens("hello world foo bar baz")
