"""DatasetCard prompt template + per-package renderer.

Pinned verbatim — diffs to this module read as ordinary PR diffs.
Plain `str.format`; no templating engine.

The system message is a module-level constant. The user message is
rendered per `PackageInputs` by `render_user_message`.
"""
from __future__ import annotations

import json
from typing import Final

from semantic_enrich.types import PackageInputs

SYSTEM_PROMPT: Final[str] = """\
You are a data catalog assistant. You read a CKAN dataset's
metadata, column names, and a handful of sample rows, then produce a
concise English description that helps a downstream search system
match user questions to the right dataset.

The dataset may be in English, French, or a mixture. Read whatever
language you see; emit English only. Never translate publisher names
(e.g. "Statistique Canada") — preserve them verbatim.

Respond with one JSON object matching the provided schema. No prose
outside the JSON. No markdown fences. The schema is enforced."""


_USER_TEMPLATE: Final[str] = """\
Dataset metadata
================
package_id: {package_id}

Representative resource: {rep_title}
Organization: {rep_organization}
Resource count in package: {resource_count}
Resource subjects: {subjects_csv}

Per-resource titles (most recent first, up to 5):
{resource_titles_block}

Description (from representative resource):
{rep_description}

Column names ({column_count} total{column_truncation_note}):
{column_names_block}

Sample rows from representative resource (document_id={rep_document_id}, total rows={rep_row_count}):
{sample_rows_block}

Instructions
============
Produce a JSON object with these fields:

- package_id: copy {package_id} verbatim.
- summary: 2 to 4 English sentences (50 to 1200 chars). State what
  one row represents, the time/geographic coverage if discernible,
  and any caveats (e.g. "snapshot", "annual update"). Avoid
  generic phrasing like "this dataset contains data about X."
- grain: a short phrase naming the per-row entity, if discernible
  (e.g. "program-year", "province-fiscal-year", "vessel-trip").
  Empty string if not discernible.
- measures: up to 20 short tokens naming numeric/quantity concepts
  the dataset measures (e.g. "expenditure_cad", "headcount",
  "tonnes_landed"). Use semantic names, not column names.
- dimensions: up to 20 short tokens naming dimensional concepts
  (e.g. "fiscal_year", "province", "industry_naics").
- date_range_start / date_range_end: ISO-format YYYY-MM-DD bounds if
  discernible from the sample rows or column names; null otherwise.

Respond with one JSON object. No prose."""


def render_user_message(pkg: PackageInputs) -> str:
    """Render the per-package user message.

    Pure function — same inputs always produce the same string, so two
    extracts of the same package against the same `raw.*` snapshot
    feed the model byte-identical prompts."""
    rep = next(
        (r for r in pkg.resources if r.document_id == pkg.representative_document_id),
        pkg.resources[0],
    )

    # Subject union across all resources, sorted, comma-separated.
    subject_union = sorted({s for r in pkg.resources for s in r.subjects})
    subjects_csv = ", ".join(subject_union) if subject_union else "(none)"

    # Per-resource titles — already recency-sorted by the candidate
    # query's ARRAY_AGG ORDER BY resource_last_modified DESC NULLS LAST.
    titles = []
    for r in pkg.resources[:5]:
        title = r.title or "(no title)"
        titles.append(f"- {title} [{r.file_format}, document_id={r.document_id}]")
    resource_titles_block = "\n".join(titles)

    column_truncation_note = (
        f", truncated from {pkg.column_names_truncated_to}"
        if pkg.column_names_truncated_to is not None
        else ""
    )
    column_names_block = ", ".join(pkg.column_names) if pkg.column_names else "(none)"

    if pkg.sample_rows:
        sample_rows_block = "\n".join(
            json.dumps(row, ensure_ascii=False) for row in pkg.sample_rows
        )
    else:
        sample_rows_block = "(no rows)"

    return _USER_TEMPLATE.format(
        package_id=pkg.package_id,
        rep_title=rep.title or "(no title)",
        rep_organization=rep.organization_code,
        resource_count=len(pkg.resources),
        subjects_csv=subjects_csv,
        resource_titles_block=resource_titles_block,
        rep_description=rep.description or "(no description)",
        column_count=len(pkg.column_names),
        column_truncation_note=column_truncation_note,
        column_names_block=column_names_block,
        rep_document_id=rep.document_id,
        rep_row_count=rep.row_count if rep.row_count is not None else "unknown",
        sample_rows_block=sample_rows_block,
    )


def estimate_tokens(text: str) -> int:
    """Rough token-count estimate for dry-run logging.

    Not loadbearing — the Qwen tokenizer would give a precise count
    but importing it costs ~3 GB of weights on first call. The 1
    token ≈ 4 chars heuristic is within ~25% for English-heavy text,
    which is fine for an order-of-magnitude check.
    """
    return max(1, len(text) // 4)
