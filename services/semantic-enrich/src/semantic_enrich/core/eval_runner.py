"""Per-question orchestrator for `semantic-enrich eval`.

Layout, per PRD §9:
  1. Assert preconditions (§10). Abort exit 3 on failure.
  2. Load prompt template.
  3. Load + validate the question fixture.
  4. For each question:
       a. Emit `question_start`.
       b. Embed → package search → column search.
       c. If retrieval empty: grade `retrieval_miss`, continue.
       d. Generate SQL. On structured-output violation: grade
          `sql_not_generated`, continue.
       e. Guard. On reject: grade per reason.
       f. If --no-execute: grade skipped-execution.
       g. Execute; grade per outcome.
  5. Write reports.

Sequential — 20 x ~1.5s per gpt-4o call is well under 90 s total.
Parallelism would complicate ordering + log interleaving and gain
nothing at this size.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from semantic_enrich.clients.bq import BqClient
from semantic_enrich.clients.openai import OpenAIClient
from semantic_enrich.config.settings import Settings
from semantic_enrich.core.eval_question_set import (
    EvalQuestion,
    load_question_set,
)
from semantic_enrich.core.eval_report import (
    EvalRunSummary,
    GradeInputs,
    RunAggregator,
    finalise_grade,
    write_partial_grade,
    write_reports,
)
from semantic_enrich.core.retrieval import (
    embed_question,
    retrieve_columns,
    retrieve_documents,
    retrieve_packages,
)
from semantic_enrich.core.sql_executor import execute as execute_sql
from semantic_enrich.core.sql_generator import (
    SqlGenerationError,
    generate_sql,
    load_prompt_template,
    prompt_template_hash,
)
from semantic_enrich.core.sql_guard import guard
from semantic_enrich.providers.logging import get_logger

# Cap the per-question row sample carried into the report. Full rows
# land in the JSON payload verbatim; the sample is what the Markdown
# renderer shows.
_ROWS_SAMPLE_CAP = 20


@dataclass(frozen=True)
class EvalRequest:
    """CLI intent for one eval run. Everything else is env / Settings."""

    run_id: str
    dry_run: bool
    no_execute: bool
    limit: int | None
    question_ids: tuple[str, ...] | None
    max_bytes_billed_override: int | None
    output_override: Path | None


class PreconditionError(RuntimeError):
    """A precondition (§10) did not hold. Exit code 3 at the CLI."""


def run_eval(
    *,
    request: EvalRequest,
    settings: Settings,
    bq: BqClient,
    openai_client: OpenAIClient,
    logger: structlog.BoundLogger | None = None,
) -> EvalRunSummary:
    log = logger or get_logger("semantic_enrich.eval")
    started = datetime.now(UTC)

    if request.max_bytes_billed_override is not None:
        settings = settings.model_copy(
            update={"eval_max_bytes_billed": request.max_bytes_billed_override}
        )

    template = load_prompt_template(settings.eval_prompt_template)
    template_hash = prompt_template_hash(template, settings)

    if request.dry_run:
        return _run_dry(
            request=request,
            settings=settings,
            template_hash=template_hash,
            started=started,
            log=log,
        )

    _assert_preconditions(
        settings=settings,
        bq=bq,
        openai_client=openai_client,
        log=log,
    )

    questions = load_question_set(settings.eval_questions_path)
    questions = _filter(questions, request=request)

    log.info(
        "eval_run_start",
        run_id=request.run_id,
        dry_run=False,
        no_execute=request.no_execute,
        questions_count=len(questions),
        k_packages=settings.eval_k_packages,
        k_columns=settings.eval_k_columns,
        max_bytes_billed=settings.eval_max_bytes_billed,
        generation_model=settings.openai_generation_model,
        embedding_model=settings.openai_embedding_model,
    )

    aggregator = RunAggregator(
        run_id=request.run_id,
        started_at=started,
        k_packages=settings.eval_k_packages,
        k_columns=settings.eval_k_columns,
        no_execute=request.no_execute,
        fixture_path=str(settings.eval_questions_path),
        prompt_template_path=str(settings.eval_prompt_template),
        prompt_template_hash=template_hash,
        generation_model=settings.openai_generation_model,
        embedding_model=settings.openai_embedding_model,
    )

    reports_dir = settings.eval_reports_dir
    partial_path = reports_dir / f"{request.run_id}.partial.jsonl"

    for question in questions:
        grade_inputs = _process_question(
            question=question,
            template=template,
            settings=settings,
            bq=bq,
            openai_client=openai_client,
            no_execute=request.no_execute,
            log=log,
            aggregator=aggregator,
        )
        grade = finalise_grade(grade_inputs)
        aggregator.add(grade)
        write_partial_grade(partial_path=partial_path, grade=grade)
        log.info(
            "question_finish",
            question_id=grade.question_id,
            terminal_state=grade.terminal_state,
            recall_packages=grade.retrieval_recall_packages_at_5,
            recall_columns=grade.retrieval_recall_columns_at_15,
        )

    finished = datetime.now(UTC)
    summary = aggregator.finalise(finished_at=finished)
    json_path, md_path = write_reports(
        summary=summary,
        grades=aggregator.grades,
        reports_dir=reports_dir,
    )
    # Clean-completion signal: remove the partial log so the operator
    # doesn't confuse a completed run for a crashed one.
    if partial_path.exists():
        partial_path.unlink()

    duration_ms = int((finished - started).total_seconds() * 1000)
    log.info(
        "eval_run_finish",
        run_id=request.run_id,
        summary=summary.__dict__,
        duration_ms=duration_ms,
        report_json_path=str(json_path),
        report_md_path=str(md_path),
    )
    return summary


# ── Preconditions (§10) ──


def _assert_preconditions(
    *,
    settings: Settings,
    bq: BqClient,
    openai_client: OpenAIClient,
    log: structlog.BoundLogger,
) -> None:
    project_id = settings.gcp_project_id
    if not project_id:
        raise PreconditionError(
            "WHENRICH_GCP_PROJECT_ID (or GCP_PROJECT_ID) is required for eval."
        )
    if settings.openai_api_key is None:
        raise PreconditionError(
            "WHENRICH_OPENAI_API_KEY (or OPENAI_API_KEY) is required for eval."
        )

    ping = openai_client.embed(["ping"])
    if len(ping) != 1 or len(ping[0]) != settings.openai_embedding_dim:
        raise PreconditionError(
            f"openai embed preflight returned unexpected shape "
            f"(vectors={len(ping)}, "
            f"dim={len(ping[0]) if ping else 'n/a'}, "
            f"expected_dim={settings.openai_embedding_dim})."
        )

    # Canary structured-output call — trivial schema, one field. Verifies
    # the model + strict-json path is reachable before we start billing
    # per-question requests.
    canary = openai_client.generate_structured(
        prompt="Return {\"ok\": \"yes\"} exactly.",
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["ok"],
            "properties": {"ok": {"type": "string"}},
        },
        schema_name="canary",
        model=settings.openai_generation_model,
        temperature=0.0,
        max_tokens=32,
    )
    if not isinstance(canary.parsed, dict) or "ok" not in canary.parsed:
        raise PreconditionError(
            f"openai generate_structured canary returned unexpected shape: "
            f"{canary.parsed!r}"
        )

    datasets_table = (
        f"{project_id}.{settings.bq_dataset_semantic}."
        f"{settings.bq_datasets_table}"
    )
    columns_table = (
        f"{project_id}.{settings.bq_dataset_semantic}."
        f"{settings.bq_columns_table}"
    )

    datasets_count = _count_rows(bq, datasets_table)
    columns_count = _count_rows(bq, columns_table)
    if datasets_count == 0:
        raise PreconditionError(
            f"`{datasets_table}` is empty; run `semantic-enrich datasets-load`."
        )
    if columns_count == 0:
        raise PreconditionError(
            f"`{columns_table}` is empty; run `semantic-enrich columns-load`."
        )

    dim_sql = (
        f"SELECT ARRAY_LENGTH(embedding) AS dim FROM `{datasets_table}` "
        "WHERE embedding IS NOT NULL LIMIT 1"
    )
    dim_rows = list(bq.query_rows(dim_sql))
    if not dim_rows:
        raise PreconditionError(
            f"`{datasets_table}` has no populated embeddings; run "
            "`semantic-enrich datasets-reembed`."
        )
    dim = int(dim_rows[0]["dim"])
    if dim != settings.openai_embedding_dim:
        raise PreconditionError(
            f"`{datasets_table}` embedding dim={dim} != "
            f"expected {settings.openai_embedding_dim}. "
            "Run `semantic-enrich datasets-reembed` and "
            "`semantic-enrich columns-reembed`."
        )

    log.info(
        "preconditions_passed",
        datasets_row_count=datasets_count,
        columns_row_count=columns_count,
        embedding_dim=dim,
    )


def _count_rows(bq: BqClient, table: str) -> int:
    rows = list(bq.query_rows(f"SELECT COUNT(*) AS n FROM `{table}`"))
    return int(rows[0]["n"]) if rows else 0


# ── Per-question pipeline ──


def _process_question(
    *,
    question: EvalQuestion,
    template: Any,
    settings: Settings,
    bq: BqClient,
    openai_client: OpenAIClient,
    no_execute: bool,
    log: structlog.BoundLogger,
    aggregator: RunAggregator,
) -> GradeInputs:
    log.info(
        "question_start",
        question_id=question.id,
        domain=question.domain,
        must_return_rows=question.must_return_rows,
    )
    inputs = GradeInputs(
        question_id=question.id,
        question_text=question.question,
        domain=question.domain,
        expected_packages=question.expected_packages,
        expected_columns=question.expected_columns,
        must_return_rows=question.must_return_rows,
        no_execute=no_execute,
    )

    embed_started = time.monotonic()
    question_vec = embed_question(
        openai_client=openai_client, question=question.question, settings=settings
    )
    log.info(
        "question_embedding_done",
        question_id=question.id,
        embedding_dim=len(question_vec),
        latency_ms=int((time.monotonic() - embed_started) * 1000),
    )

    packages, packages_ms = retrieve_packages(
        bq=bq, question_vec=question_vec, settings=settings
    )
    if not packages:
        inputs.retrieval_miss = True
        log.info(
            "retrieval_miss", question_id=question.id, stage="packages"
        )
        return inputs
    inputs.top_packages = tuple(p.package_id for p in packages)
    log.info(
        "retrieval_packages_done",
        question_id=question.id,
        top_k_packages=len(packages),
        min_distance=min(p.distance for p in packages),
        max_distance=max(p.distance for p in packages),
        latency_ms=packages_ms,
    )

    scoped_ids = [p.package_id for p in packages]
    columns, columns_ms = retrieve_columns(
        bq=bq,
        question_vec=question_vec,
        scoped_packages=scoped_ids,
        settings=settings,
    )
    if not columns:
        inputs.retrieval_miss = True
        log.info(
            "retrieval_miss", question_id=question.id, stage="columns"
        )
        return inputs
    inputs.top_columns = tuple(
        (c.package_id, c.column_name) for c in columns
    )
    log.info(
        "retrieval_columns_done",
        question_id=question.id,
        top_k_columns=len(columns),
        latency_ms=columns_ms,
    )

    documents, documents_ms = retrieve_documents(
        bq=bq, package_ids=scoped_ids, settings=settings
    )
    if not documents:
        inputs.retrieval_miss = True
        log.info(
            "retrieval_miss", question_id=question.id, stage="documents"
        )
        return inputs
    log.info(
        "retrieval_documents_done",
        question_id=question.id,
        top_k_documents=len(documents),
        document_ids=[d.document_id for d in documents],
        latency_ms=documents_ms,
    )

    try:
        sql_result, prompt = generate_sql(
            openai_client=openai_client,
            template=template,
            question=question.question,
            packages=packages,
            columns=columns,
            documents=documents,
            settings=settings,
        )
    except SqlGenerationError as exc:
        log.error(
            "structured_output_violation",
            question_id=question.id,
            error=str(exc),
        )
        inputs.structured_output_violation = True
        return inputs

    inputs.sql_generated = True
    inputs.sql_text = sql_result.sql
    inputs.rationale = sql_result.rationale
    inputs.answer_summary = sql_result.answer_summary
    aggregator.add_tokens(sql_result.tokens_in, sql_result.tokens_out)
    log.info(
        "sql_gen_prompt",
        question_id=question.id,
        prompt=prompt,
        prompt_char_len=len(prompt),
    )
    log.info(
        "sql_gen_done",
        question_id=question.id,
        sql=sql_result.sql,
        rationale=sql_result.rationale,
        tokens_in=sql_result.tokens_in,
        tokens_out=sql_result.tokens_out,
        latency_ms=sql_result.latency_ms,
    )

    guard_result = guard(sql=sql_result.sql, bq=bq, settings=settings)
    inputs.sql_final_text = guard_result.sql_final
    inputs.dry_run_bytes = guard_result.dry_run_bytes
    inputs.sql_valid = guard_result.accepted
    inputs.guard_reject_reason = guard_result.reason
    if not guard_result.accepted:
        log.info(
            "sql_guard_rejected",
            question_id=question.id,
            reason=guard_result.reason,
            sql_final=guard_result.sql_final,
        )
        return inputs
    log.info(
        "sql_guard_accepted",
        question_id=question.id,
        sql_final=guard_result.sql_final,
        dry_run_bytes=guard_result.dry_run_bytes,
        limit_wrapped=guard_result.limit_wrapped,
    )

    if no_execute:
        return inputs

    execution = execute_sql(
        sql=guard_result.sql_final, bq=bq, settings=settings
    )
    inputs.rows_returned = execution.row_count
    inputs.bytes_billed = execution.bytes_billed
    inputs.slot_ms = execution.slot_ms
    inputs.elapsed_ms = execution.elapsed_ms
    inputs.sql_timed_out = execution.timed_out
    inputs.execution_error = execution.error
    inputs.rows_sample = tuple(execution.rows[:_ROWS_SAMPLE_CAP])

    if execution.error and not execution.timed_out:
        log.info(
            "sql_execute_failed",
            question_id=question.id,
            error=execution.error,
            elapsed_ms=execution.elapsed_ms,
        )
    else:
        log.info(
            "sql_execute_done",
            question_id=question.id,
            row_count=execution.row_count,
            bytes_billed=execution.bytes_billed,
            slot_ms=execution.slot_ms,
            elapsed_ms=execution.elapsed_ms,
        )
    return inputs


# ── Dry-run harness self-test ──


def _run_dry(
    *,
    request: EvalRequest,
    settings: Settings,
    template_hash: str,
    started: datetime,
    log: structlog.BoundLogger,
) -> EvalRunSummary:
    """`--dry-run` harness self-test.

    Loads the fixture, renders the prompt hash, writes an empty report.
    No OpenAI, no BQ calls. Verifies the fixture parses, the prompt
    template renders against StrictUndefined, and the report writer
    lays out its output paths.
    """
    questions = load_question_set(settings.eval_questions_path)
    questions = _filter(questions, request=request)
    log.info(
        "eval_run_start",
        run_id=request.run_id,
        dry_run=True,
        no_execute=request.no_execute,
        questions_count=len(questions),
        k_packages=settings.eval_k_packages,
        k_columns=settings.eval_k_columns,
        max_bytes_billed=settings.eval_max_bytes_billed,
        generation_model=settings.openai_generation_model,
        embedding_model=settings.openai_embedding_model,
    )

    # Dry-run flips `no_execute=True` on the aggregate so the report
    # doesn't claim `answered_count = 0` when the run never touched a
    # question end-to-end. Keeps the summary shape honest.
    aggregator = RunAggregator(
        run_id=request.run_id,
        started_at=started,
        k_packages=settings.eval_k_packages,
        k_columns=settings.eval_k_columns,
        no_execute=True,
        fixture_path=str(settings.eval_questions_path),
        prompt_template_path=str(settings.eval_prompt_template),
        prompt_template_hash=template_hash,
        generation_model=settings.openai_generation_model,
        embedding_model=settings.openai_embedding_model,
    )

    # Dry-run intentionally does not call generate_grade so terminal_state
    # exposes only the retrieval-miss branch; every fixture question
    # grades `retrieval_miss` in dry-run, which is honest — nothing was
    # actually retrieved.
    for question in questions:
        inputs = GradeInputs(
            question_id=question.id,
            question_text=question.question,
            domain=question.domain,
            expected_packages=question.expected_packages,
            expected_columns=question.expected_columns,
            must_return_rows=question.must_return_rows,
            no_execute=request.no_execute,
            retrieval_miss=True,
        )
        aggregator.add(finalise_grade(inputs))

    finished = datetime.now(UTC)
    summary = aggregator.finalise(finished_at=finished)
    json_path, md_path = write_reports(
        summary=summary,
        grades=aggregator.grades,
        reports_dir=settings.eval_reports_dir,
    )
    duration_ms = int((finished - started).total_seconds() * 1000)
    log.info(
        "eval_run_finish",
        run_id=request.run_id,
        summary=summary.__dict__,
        duration_ms=duration_ms,
        report_json_path=str(json_path),
        report_md_path=str(md_path),
    )
    return summary


def _filter(
    questions: list[EvalQuestion], *, request: EvalRequest
) -> list[EvalQuestion]:
    """Apply --question-ids then --limit, in that order. --limit wins
    if both are set (§14)."""
    filtered = questions
    if request.question_ids:
        wanted = set(request.question_ids)
        filtered = [q for q in filtered if q.id in wanted]
    if request.limit is not None:
        filtered = filtered[: request.limit]
    if not filtered:
        raise RuntimeError(
            "eval: no questions matched --question-ids / --limit filters"
        )
    return filtered
