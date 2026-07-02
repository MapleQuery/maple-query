"""§16.2 integration test — wire Fake* clients into the eval runner.

Covers the mixed-outcome path (`answered` + `sql_invalid` +
`retrieval_miss`) so the aggregate counts and failure taxonomy are
tested end-to-end.
"""
from __future__ import annotations

import math
from pathlib import Path

import yaml

from semantic_enrich.clients.bq import BoundedQueryResult
from semantic_enrich.config.settings import Settings
from semantic_enrich.core.eval_runner import EvalRequest, run_eval
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient


def _write_fixture(tmp_path: Path) -> Path:
    payload = [
        {
            "id": "q01",
            "question": "How much did we spend on housing?",
            "domain": "housing",
            "expected_packages": [],
            "expected_columns": ["TOT_EXP"],
            "must_return_rows": True,
        },
        {
            "id": "q02",
            "question": "hallucinated table time",
            "domain": "taxes",
            "expected_packages": [],
            "expected_columns": [],
            "must_return_rows": True,
        },
        {
            "id": "q03",
            "question": "retrieval never finds anything",
            "domain": "environment",
            "expected_packages": [],
            "expected_columns": [],
            "must_return_rows": True,
        },
    ]
    path = tmp_path / "questions.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _copy_template(tmp_path: Path) -> Path:
    src = Path(__file__).resolve().parents[2] / "eval" / "prompts" / "sql_generation.j2"
    dst = tmp_path / "sql_generation.j2"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dst


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
        eval_questions_path=_write_fixture(tmp_path),
        eval_prompt_template=_copy_template(tmp_path),
        eval_reports_dir=tmp_path / "reports",
    )


def test_mixed_outcome_run(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    bq = FakeBqClient()
    # Preconditions.
    bq.register_query("COUNT(*) AS n", [{"n": 10}])
    bq.register_query("COUNT(*) AS n", [{"n": 100}])
    bq.register_query("ARRAY_LENGTH(embedding)", [{"dim": 1536}])
    # q01: VECTOR_SEARCH returns one package + one column, then the
    # documents lookup returns one loaded doc.
    bq.register_query("VECTOR_SEARCH", [
        {"package_id": "pkg-1", "summary": "housing", "grain": None,
         "measures": [], "dimensions": [], "distance": 0.1}
    ])
    bq.register_query("VECTOR_SEARCH", [
        {"package_id": "pkg-1", "column_name": "TOT_EXP",
         "semantic_type": "currency", "description": "spend",
         "sample_values": [], "distance": 0.15}
    ])
    bq.register_query("load_status = 'loaded'", [
        {"document_id": "doc-1", "package_id": "pkg-1",
         "title": "Housing 2023", "row_count": 42,
         "resource_last_modified": None},
    ])
    bq.register_query("JSON_KEYS(PARSE_JSON(STRING(row)))", [
        {"document_id": "doc-1", "columns": ["Amount", "Organization"]},
    ])
    # q02: same retrieval shape as q01.
    bq.register_query("VECTOR_SEARCH", [
        {"package_id": "pkg-2", "summary": "taxes", "grain": None,
         "measures": [], "dimensions": [], "distance": 0.2}
    ])
    bq.register_query("VECTOR_SEARCH", [
        {"package_id": "pkg-2", "column_name": "AMT",
         "semantic_type": None, "description": "tax",
         "sample_values": [], "distance": 0.25}
    ])
    bq.register_query("load_status = 'loaded'", [
        {"document_id": "doc-2", "package_id": "pkg-2",
         "title": None, "row_count": None,
         "resource_last_modified": None},
    ])
    bq.register_query("JSON_KEYS(PARSE_JSON(STRING(row)))", [
        {"document_id": "doc-2", "columns": []},
    ])
    # q03: retrieval miss — both VECTOR_SEARCH calls return [].
    bq.register_query("VECTOR_SEARCH", [])
    bq.register_query("VECTOR_SEARCH", [])

    # Executor result for q01 → answered. Key off the literal doc_id the
    # model inlines (not the package_id — the emitted SQL no longer
    # references packages at all).
    bq.register_bounded_query(
        "doc-1",
        BoundedQueryResult(
            rows=[{"n": 1}],
            total_bytes_billed=1024,
            slot_ms=10,
            elapsed_ms=50,
            timed_out=False,
            error=None,
        ),
    )

    q1_response = {
        "sql": (
            "SELECT 1 AS n FROM `proj.raw.rows` "
            "WHERE document_id IN ('doc-1') LIMIT 10"
        ),
        "rationale": "pkg-1 matches housing",
        "answer_summary": "one row",
    }
    q2_response = {
        # Hallucinated table — guard rejects on dataset whitelist.
        "sql": "SELECT 1 FROM `proj.other.foo` LIMIT 10",
        "rationale": "wrong dataset",
        "answer_summary": "n/a",
    }

    client = FakeOpenAIClient(
        vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536,
        structured_responses=[q1_response, q2_response],
    )

    request = EvalRequest(
        run_id="mixed",
        dry_run=False,
        no_execute=False,
        limit=None,
        question_ids=None,
        max_bytes_billed_override=None,
        output_override=None,
    )
    summary = run_eval(
        request=request, settings=settings, bq=bq, openai_client=client
    )

    assert summary.questions_total == 3
    assert summary.sql_generated_count == 2
    assert summary.sql_valid_count == 1
    assert summary.answered_count == 1
    assert summary.failures_by_reason.get("retrieval_miss") == 1
    assert summary.failures_by_reason.get("sql_invalid") == 1

    json_path = settings.eval_reports_dir / "mixed.json"
    md_path = settings.eval_reports_dir / "mixed.md"
    assert json_path.exists()
    assert md_path.exists()
