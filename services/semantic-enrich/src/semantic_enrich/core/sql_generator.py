"""Single-shot SQL generation via OpenAI Structured Outputs.

One rendered Jinja2 prompt, one OpenAI call, one schema-conforming
dict. No retry, no multi-turn — that boundary is what separates the
4.6 harness from the M4 agent.

The prompt template at `services/semantic-enrich/eval/prompts/
sql_generation.j2` is the load-bearing artefact of the harness; the
operator iterates it after reading reports.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jinja2

from semantic_enrich.clients.openai import (
    OpenAIClient,
    StructuredGenerationResult,
)
from semantic_enrich.config.settings import Settings
from semantic_enrich.core.retrieval import ColumnCandidate, PackageCandidate

SQL_GEN_SCHEMA_NAME = "sql_result"

# OpenAI Structured Outputs (`strict: true`) requires
# `additionalProperties: false` and every property to appear in
# `required`. Post-parse checks (empty SQL, absurd length) live in the
# guard rather than the schema, since Structured Outputs does not
# support minLength / maxLength under strict mode.
SQL_GEN_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["sql", "rationale", "answer_summary"],
    "properties": {
        "sql": {"type": "string"},
        "rationale": {"type": "string"},
        "answer_summary": {"type": "string"},
    },
}


@dataclass(frozen=True)
class SqlGenerationResult:
    """Runner-facing output of one generate call.

    `sql` is the model's raw statement (pre-guard); the guard may
    auto-wrap it with a LIMIT and hand back a rewritten `sql_final`.
    """

    sql: str
    rationale: str
    answer_summary: str
    tokens_in: int
    tokens_out: int
    latency_ms: int


class SqlGenerationError(RuntimeError):
    """Structured Outputs returned a shape that violates the schema
    despite `strict: true`. Treated as a vendor regression by the
    runner (`structured_output_violation` event, `sql_not_generated`
    grade)."""


def load_prompt_template(path: Path) -> jinja2.Template:
    """Load the SQL-gen Jinja2 template.

    `autoescape=False` — output is a prompt, not HTML.
    `StrictUndefined` — a missing variable raises rather than rendering
    a literal `None`, so a schema drift between the runner and the
    template shows up as a load-time crash.
    """
    if not path.exists():
        raise RuntimeError(f"eval prompt template missing: {path}")
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(path.parent)),
        autoescape=False,
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
    )
    return env.get_template(path.name)


def render_prompt(
    *,
    template: jinja2.Template,
    question: str,
    packages: list[PackageCandidate],
    columns: list[ColumnCandidate],
    settings: Settings,
) -> str:
    return template.render(
        question=question,
        packages=packages,
        columns=columns,
        row_limit=settings.eval_row_limit,
        allowed_datasets=settings.eval_allowed_datasets,
    )


def prompt_template_hash(template: jinja2.Template, settings: Settings) -> str:
    """SHA-256 of the rendered prompt for a canonical fixture question.

    Anchors prompt iteration across runs: identical hash means an
    identical prompt, so any per-question delta is attributable to
    something other than the template."""
    canonical = render_prompt(
        template=template,
        question="canonical prompt hash question",
        packages=[],
        columns=[],
        settings=settings,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def generate_sql(
    *,
    openai_client: OpenAIClient,
    template: jinja2.Template,
    question: str,
    packages: list[PackageCandidate],
    columns: list[ColumnCandidate],
    settings: Settings,
) -> tuple[SqlGenerationResult, str]:
    """Emit one SELECT statement for `question`. Returns
    `(result, rendered_prompt)`; the runner logs the prompt once per
    question for post-hoc auditing without a re-run."""
    prompt = render_prompt(
        template=template,
        question=question,
        packages=packages,
        columns=columns,
        settings=settings,
    )
    started = time.monotonic()
    raw = openai_client.generate_structured(
        prompt=prompt,
        schema=SQL_GEN_OUTPUT_SCHEMA,
        schema_name=SQL_GEN_SCHEMA_NAME,
        model=settings.openai_generation_model,
        temperature=settings.openai_generation_temperature,
        max_tokens=settings.openai_generation_max_tokens,
    )
    latency_ms = int((time.monotonic() - started) * 1000)
    return _to_result(raw, latency_ms=latency_ms), prompt


def _to_result(
    raw: StructuredGenerationResult, *, latency_ms: int
) -> SqlGenerationResult:
    parsed = raw.parsed
    for key in ("sql", "rationale", "answer_summary"):
        value = parsed.get(key)
        if not isinstance(value, str):
            raise SqlGenerationError(
                f"structured_output_violation: field {key!r} missing or "
                f"non-string in {parsed!r}"
            )
    return SqlGenerationResult(
        sql=parsed["sql"],
        rationale=parsed["rationale"],
        answer_summary=parsed["answer_summary"],
        tokens_in=raw.tokens_in,
        tokens_out=raw.tokens_out,
        latency_ms=latency_ms,
    )
