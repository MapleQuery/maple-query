"""§16.1 unit tests for the SQL generator prompt path.

Focus on the template contract, not the vendor call:
- StrictUndefined raises on a missing variable.
- Rendered prompt contains the question, candidate ids, and LIMIT.
- Prompt hash is stable across identical Settings.
"""
from __future__ import annotations

from pathlib import Path

import jinja2
import pytest

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.retrieval import (
    ColumnCandidate,
    DocumentCandidate,
    PackageCandidate,
)
from semantic_enrich.core.sql_generator import (
    load_prompt_template,
    prompt_template_hash,
    render_prompt,
)

_SERVICE_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATE = _SERVICE_ROOT / "eval" / "prompts" / "sql_generation.j2"


def _settings() -> Settings:
    return Settings(gcp_project_id="proj")


def test_render_contains_question_and_limit() -> None:
    tmpl = load_prompt_template(_TEMPLATE)
    prompt = render_prompt(
        template=tmpl,
        question="How much was spent on housing?",
        packages=[
            PackageCandidate(
                package_id="pkg-1",
                summary="Housing expenditure by fiscal year.",
                grain="row",
                measures=("total",),
                dimensions=("year",),
                date_range_start=None,
                date_range_end=None,
                distance=0.1,
            )
        ],
        columns=[
            ColumnCandidate(
                package_id="pkg-1",
                column_name="TOT_EXP",
                semantic_type="currency",
                description="Total expenditure",
                sample_values=("100", "200"),
                distance=0.15,
            )
        ],
        documents=[],
        settings=_settings(),
    )
    assert "How much was spent on housing?" in prompt
    assert "pkg-1" in prompt
    assert "TOT_EXP" in prompt
    assert "LIMIT 100" in prompt


def test_render_contains_documents() -> None:
    """The resolved document_ids must appear in the prompt so the model
    can inline them as literals — that's the whole point of the
    third-stage retrieval."""
    tmpl = load_prompt_template(_TEMPLATE)
    prompt = render_prompt(
        template=tmpl,
        question="How much was spent on housing?",
        packages=[],
        columns=[],
        documents=[
            DocumentCandidate(
                document_id="doc-abc",
                package_id="pkg-1",
                title="Housing expenditure 2023",
                row_count=1234,
                resource_last_modified=None,
                columns=("Amount", "Organization"),
            ),
            DocumentCandidate(
                document_id="doc-def",
                package_id="pkg-1",
                title=None,
                row_count=None,
                resource_last_modified=None,
                columns=(),
            ),
        ],
        settings=_settings(),
    )
    assert "doc-abc" in prompt
    assert "doc-def" in prompt
    # Prompt must instruct the model to inline literals rather than emit
    # a subquery IN — that's the whole point of the third-stage retrieval.
    assert "LITERAL `document_id` IN-list" in prompt
    # Missing row_count / title fall back to the "unspecified" sentinel
    # rather than rendering Python `None`.
    assert "unspecified" in prompt
    # Per-doc columns rendered so the model can pair the right doc
    # with the right column.
    assert '"Amount"' in prompt
    assert '"Organization"' in prompt
    # Direct-access guidance: row is a native JSON object; the legacy
    # PARSE_JSON(STRING(r.row)) unwrap throws and must not be taught.
    assert "JSON_VALUE(r.row, '$.<column_name>')" in prompt
    assert "JSON_VALUE(PARSE_JSON" not in prompt


def test_render_missing_var_raises(tmp_path: Path) -> None:
    """StrictUndefined: a template that references a missing var must
    raise at render time rather than emit literal `None`."""
    bad_template_path = tmp_path / "bad.j2"
    bad_template_path.write_text("{{ nonexistent_var }}", encoding="utf-8")
    tmpl = load_prompt_template(bad_template_path)
    with pytest.raises(jinja2.UndefinedError):
        tmpl.render()


def test_prompt_hash_stable() -> None:
    tmpl = load_prompt_template(_TEMPLATE)
    s = _settings()
    h1 = prompt_template_hash(tmpl, s)
    h2 = prompt_template_hash(tmpl, s)
    assert h1 == h2
    assert len(h1) == 64
