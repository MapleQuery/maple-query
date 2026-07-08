"""`run_agent_eval` end-to-end against a scripted loop (no network).

Uses the committed fixture and a canned OpenAI client so the real
`run_turn` executes: history validation, cache, cost accounting, and
the report writer all run for real."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from semantic_enrich.clients.openai import ChatCompletionResult
from semantic_enrich.config.settings import Settings
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_eval import (
    AgentEvalRequest,
    run_agent_eval,
)
from semantic_enrich.core.agent_loop import LoopDeps

FIXTURE = (
    Path(__file__).resolve().parents[2] / "eval" / "questions-agent-traces.yaml"
)


class _CannedOpenAI:
    """Always answers immediately with no tool calls."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 1536 for _ in texts]

    def generate_structured(self, **kwargs: Any) -> Any:
        raise RuntimeError("not used")

    def chat_with_tools(self, **kwargs: Any) -> ChatCompletionResult:
        return ChatCompletionResult(
            content="canned answer",
            tool_calls=[],
            tokens_in=25,
            tokens_out=5,
            finish_reason="stop",
        )


def test_run_agent_eval_writes_baseline_report(tmp_path: Path) -> None:
    settings = Settings().model_copy(
        update={
            "eval_questions_path": FIXTURE,
            "eval_reports_dir": tmp_path,
        }
    )
    deps = LoopDeps(
        bq=object(),
        openai_client=_CannedOpenAI(),
        settings=settings,
        system_prompt="test prompt",
        prompt_hash="ph-test",
        cache=ResponseCache(
            max_entries=4, max_value_bytes=100_000, ttl_seconds=60
        ),
        snapshot_hash_provider=lambda: "snap-test",
        system_prompt_tokens=7,
    )
    request = AgentEvalRequest(
        run_id="testrun-123", limit=3, output_override=None
    )

    report = run_agent_eval(request=request, settings=settings, deps=deps)

    assert report["questions_count"] == 3
    assert report["loop_impl"] == "v1"
    assert report["prompt_hash"] == "ph-test"
    assert report["totals"]["terminal_counts"] == {"done": 3}
    for row in report["questions"]:
        assert row["run"]["terminal"] == "done"
        assert row["run"]["final_message"] == "canned answer"
        assert row["run"]["tokens_in_per_call"] == [25]
        assert row["expected"]["triage"]

    written = tmp_path / "agent-traces-testrun-123.json"
    assert written.exists()
    on_disk = json.loads(written.read_text(encoding="utf-8"))
    assert on_disk["run_id"] == "testrun-123"
    assert len(on_disk["questions"]) == 3
