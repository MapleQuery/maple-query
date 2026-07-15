"""Category-handler output: deterministic deflection templates, the
suggestion-clause guards, the describe_corpus-backed meta path, and
the fixed identity line."""
from __future__ import annotations

from typing import Any

import pytest

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_tools
from semantic_enrich.core.agent.phases import PipelineDeps, TurnContext
from semantic_enrich.core.agent.triage import (
    IDENTITY_LINE,
    QueryTriage,
    off_scope_message,
)
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_request import ChatRequest
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient


@pytest.fixture(autouse=True)
def _fresh_corpus_cache() -> None:
    agent_tools.reset_corpus_stats_cache()


# ── off_scope template ──


@pytest.mark.parametrize(
    ("sub_reason", "fragment"),
    [
        ("provincial", "Provincial and municipal"),
        ("news", "News and current events"),
        ("opinion", "Opinion and ranking"),
        ("non_canada", "other countries"),
        ("personal", "personal or private records"),
        ("jailbreak", "outside what it can help with"),
        ("other", "outside what the data can answer"),
    ],
)
def test_off_scope_renders_the_fixed_clause_per_sub_reason(
    sub_reason: str, fragment: str
) -> None:
    message = off_scope_message(
        sub_reason=sub_reason, deflection_hint=None
    )
    assert message.startswith(
        "MapleQuery answers questions from Canadian **federal** open data"
    )
    assert fragment in message


def test_unknown_sub_reason_falls_back_to_other() -> None:
    message = off_scope_message(
        sub_reason="something-new", deflection_hint=None
    )
    assert "outside what the data can answer" in message


def test_valid_deflection_hint_renders_as_a_suggestion() -> None:
    message = off_scope_message(
        sub_reason="provincial",
        deflection_hint="federal infrastructure spending by province",
    )
    assert (
        "You could ask instead: federal infrastructure spending"
        in message
    )


def test_oversized_hint_is_dropped() -> None:
    message = off_scope_message(
        sub_reason="news", deflection_hint="x" * 161
    )
    assert "You could ask instead" not in message


def test_url_bearing_hints_are_dropped() -> None:
    for hint in (
        "see https://example.com for more",
        "try www.canada.ca instead",
    ):
        message = off_scope_message(sub_reason="news", deflection_hint=hint)
        assert "You could ask instead" not in message


def test_jailbreak_gets_the_bare_template_even_with_a_hint() -> None:
    message = off_scope_message(
        sub_reason="jailbreak",
        deflection_hint="a perfectly valid-looking suggestion",
    )
    assert "You could ask instead" not in message
    assert "suggestion" not in message


# ── meta handler ──


def _settings(**overrides: Any) -> Settings:
    return Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
        agent_triage_mode="act",
        **overrides,
    )


def _meta_classification() -> dict[str, Any]:
    return {
        "category": "meta",
        "confidence": 0.95,
        "reason": "about the system",
        "off_scope_reason": None,
        "deflection_hint": None,
        "clarify_question": None,
    }


def _stats_bq() -> FakeBqClient:
    bq = FakeBqClient()
    bq.table_num_rows_by_ref["proj.raw.rows"] = 205_000_000
    bq.register_query(
        "AS packages",
        [
            {
                "packages": 210,
                "documents_loaded": 950,
                "latest_load_at": "2026-07-01T00:00:00",
            }
        ],
    )
    return bq


def _classify(
    *,
    question: str,
    bq: FakeBqClient,
    openai: FakeOpenAIClient,
) -> tuple[str | None, TurnContext]:
    settings = _settings()
    deps = PipelineDeps(
        bq=bq,  # type: ignore[arg-type]
        openai_client=openai,
        settings=settings,
        system_prompt="test system prompt",
        prompt_hash="hash-test",
        cache=ResponseCache(
            max_entries=10, max_value_bytes=1_000_000, ttl_seconds=60
        ),
        snapshot_hash_provider=lambda: "snap-0",
    )
    ctx = TurnContext.begin(
        request=ChatRequest(
            conversation_id="c1", history=[], question=question
        ),
        deps=deps,
    )
    outcome = QueryTriage.from_settings(settings).classify(ctx)
    return outcome.short_circuit, ctx


def test_meta_calls_describe_corpus_and_not_search_datasets() -> None:
    bq = _stats_bq()
    openai = FakeOpenAIClient(
        structured_responses=[_meta_classification()]
    )
    answer, _ctx = _classify(
        question="how many rows of data do you have access to?",
        bq=bq,
        openai=openai,
    )

    assert answer is not None
    assert "205,000,000" in answer
    # describe_corpus surfaces only: table metadata + the stats COUNTs.
    assert bq.table_num_rows_calls == ["proj.raw.rows"]
    assert all("AS packages" in c["sql"] for c in bq.calls)
    # No retrieval: no embeddings, no bounded (run_sql) queries, and no
    # second model call for a stat-mapped question.
    assert openai.calls == []
    assert bq.bounded_calls == []
    assert len(openai.structured_calls) == 1


def test_dataset_count_question_uses_the_stats_template() -> None:
    answer, _ctx = _classify(
        question="how many datasets do you have?",
        bq=_stats_bq(),
        openai=FakeOpenAIClient(
            structured_responses=[_meta_classification()]
        ),
    )
    assert answer is not None
    assert "210" in answer


def test_identity_question_hits_the_fixed_line_without_bq() -> None:
    bq = _stats_bq()
    answer, _ctx = _classify(
        question="what model are you using",
        bq=bq,
        openai=FakeOpenAIClient(
            structured_responses=[_meta_classification()]
        ),
    )
    assert answer == IDENTITY_LINE
    assert bq.calls == []
    assert bq.table_num_rows_calls == []


def test_non_stat_meta_question_phrases_stats_with_one_mini_call() -> None:
    openai = FakeOpenAIClient(
        structured_responses=[
            _meta_classification(),
            {"answer": "MapleQuery searches open.canada.ca datasets."},
        ]
    )
    answer, _ctx = _classify(
        question="what can you help me with?",
        bq=_stats_bq(),
        openai=openai,
    )
    assert answer == "MapleQuery searches open.canada.ca datasets."
    assert len(openai.structured_calls) == 2
    phrasing = openai.structured_calls[1]
    assert phrasing["schema_name"] == "triage_meta_answer"
    # The phrasing prompt carries our stats, not retrieval output.
    assert "205000000" in phrasing["prompt"].replace(",", "")


def test_meta_survives_a_warehouse_error() -> None:
    class BrokenBq(FakeBqClient):
        def table_num_rows(self, table_ref: str) -> int:
            raise RuntimeError("bq down")

    answer, _ctx = _classify(
        question="how many rows do you have?",
        bq=BrokenBq(),
        openai=FakeOpenAIClient(
            structured_responses=[_meta_classification()]
        ),
    )
    # Still a short-circuit answer — just numberless.
    assert answer == IDENTITY_LINE
