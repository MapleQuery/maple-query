"""§16.3 E2E dry-run — `semantic-enrich eval --dry-run` against the
real 20-question fixture.

Zero OpenAI calls, zero BQ calls, both report files written, exit 0.
"""
from __future__ import annotations

import json
from pathlib import Path

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.eval_runner import EvalRequest, run_eval
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient

_SERVICE_ROOT = Path(__file__).resolve().parents[2]


def test_dry_run_reads_real_fixture(tmp_path: Path) -> None:
    settings = Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
        eval_questions_path=_SERVICE_ROOT / "eval" / "questions.yaml",
        eval_prompt_template=_SERVICE_ROOT / "eval" / "prompts" / "sql_generation.j2",
        eval_reports_dir=tmp_path / "reports",
    )
    bq = FakeBqClient()
    client = FakeOpenAIClient()

    request = EvalRequest(
        run_id="dry",
        dry_run=True,
        no_execute=False,
        limit=None,
        question_ids=None,
        max_bytes_billed_override=None,
        output_override=None,
    )
    summary = run_eval(
        request=request, settings=settings, bq=bq, openai_client=client
    )

    assert summary.questions_total == 20
    assert summary.retrieval_misses == 20
    assert bq.calls == []
    assert client.calls == []
    assert client.structured_calls == []

    json_path = settings.eval_reports_dir / "dry.json"
    md_path = settings.eval_reports_dir / "dry.md"
    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["summary"]["run_id"] == "dry"
    assert len(payload["grades"]) == 20
