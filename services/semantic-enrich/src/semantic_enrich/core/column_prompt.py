"""Columns prompt template + per-chunk renderer.

Pinned verbatim — diffs read as ordinary PR diffs. Plain `str.format`;
no templating engine.

A snapshot test pins both constants — any edit forces an intentional
snapshot re-approval.
"""
from __future__ import annotations

from typing import Final

from semantic_enrich.types import ColumnInputs

COLUMNS_SYSTEM_PROMPT: Final[str] = """\
You are a data documentarian for an open-data warehouse. The Canadian
federal government has published a dataset; you are describing one
batch of its columns.

The columns may be in English, French, or a mixture. Read whatever
language you see; emit English only. Preserve publisher-specific
identifiers and short codes verbatim (e.g. "NAICS", "FSA",
"AMT_CAD").

Respond with one JSON array matching the provided schema. No prose
outside the JSON. No markdown fences. The schema is enforced."""


COLUMNS_USER_TEMPLATE: Final[str] = """\
Dataset metadata:
- Title: {package_title}
- Description: {package_description}
- Subjects (controlled taxonomy): {package_subjects_csv}
{package_summary_block}

This is batch {chunk_index_plus_one} of {chunk_count} for this dataset.
The columns in this batch are listed below, each with up to 10 distinct
sample values drawn from one representative resource.

Columns:
{columns_block}

Produce a JSON array with exactly {column_count} entries, one per
column above, in the same order. Each entry must:
- echo `column_name` verbatim (no normalisation, no translation, no
  case change),
- assign a short `semantic_type` (e.g. "currency_cad", "fiscal_year",
  "province_code", "kg", "headcount", "identifier", "text",
  "iso_date"),
- write a 1–3 sentence English `description` (20–600 characters) that
  states what the column measures, its unit if any, and any
  publisher-specific quirk visible in the sample values,
- include up to 10 `sample_values` drawn verbatim from the provided
  samples; you may omit duplicates or values that look like sentinels.

Respond with a single JSON array. No prose, no markdown."""


def render_columns_block(
    column_names: list[str], sample_values: dict[str, list[str] | tuple[str, ...]]
) -> str:
    """Render the `Columns:` block (one entry per column).

    Format:
        - name: <col>
          samples: ["v1", "v2", ...]
    """
    lines: list[str] = []
    for name in column_names:
        samples = list(sample_values.get(name, ()))
        # JSON-array-ish literal so the model sees the structure
        # plainly without needing to parse YAML/quote rules.
        quoted = ", ".join(_quote_sample(v) for v in samples)
        lines.append(f"- name: {name}")
        lines.append(f"  samples: [{quoted}]")
    return "\n".join(lines)


def _quote_sample(v: str) -> str:
    """Quote a sample value for the columns_block.

    Keeps it simple: replace embedded double-quotes and wrap in `"`.
    This is a prompt-side display, not a parsed payload — the model
    is told to echo verbatim what it sees.
    """
    return '"' + v.replace('"', '\\"') + '"'


def render_user_message(
    *,
    inputs: ColumnInputs,
    chunk_index: int,
    chunk_count: int,
    column_names: list[str],
) -> str:
    """Render one chunk's user message.

    Pure function — same inputs always produce the same string. The
    snapshot test in `tests/unit/test_column_prompt.py` pins this
    against a fixture so an unintended template edit is caught.
    """
    package_summary_block = ""
    if inputs.package_summary is not None and inputs.package_summary.strip():
        package_summary_block = (
            f"- Summary (from semantic.datasets): {inputs.package_summary}"
        )

    subjects_csv = (
        ", ".join(inputs.package_subjects) if inputs.package_subjects else "(none)"
    )
    columns_block = render_columns_block(column_names, dict(inputs.sample_values))

    return COLUMNS_USER_TEMPLATE.format(
        package_title=inputs.package_title or "(no title)",
        package_description=inputs.package_description or "(no description)",
        package_subjects_csv=subjects_csv,
        package_summary_block=package_summary_block,
        chunk_index_plus_one=chunk_index + 1,
        chunk_count=chunk_count,
        columns_block=columns_block,
        column_count=len(column_names),
    )


def estimate_tokens(text: str) -> int:
    """Rough token-count estimate for dry-run logging.

    Same 1-token-per-4-chars heuristic as `dataset_prompt.estimate_tokens`.
    """
    return max(1, len(text) // 4)
