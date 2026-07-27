"""`explored` must survive the record round-trip.

`records.build` coerces any unknown outcome with
`outcome if outcome in OUTCOMES else "error"`, and `validate` drops a
record whose outcome it doesn't recognise. Both are silent. So adding
the tag to `_outcome` without adding it to `OUTCOMES` would record every
exploration as a failure and then discard it on the way back in, with
nothing raised anywhere. This file is the mechanical guard on that pair
of edits staying together.
"""
from __future__ import annotations

from typing import Any

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events
from semantic_enrich.core.agent import memory, records
from semantic_enrich.core.agent.phases import (
    PipelineDeps,
    ResearchResult,
    TurnContext,
)
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_request import ChatRequest
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient

PKG = "0f3765d1-3375-4423-8fd6-6da7f382fa1a"


def _ctx() -> TurnContext:
    settings = Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
    )
    deps = PipelineDeps(
        bq=FakeBqClient(),
        openai_client=FakeOpenAIClient(),
        settings=settings,
        system_prompt="p",
        prompt_hash="h",
        cache=ResponseCache(
            max_entries=4, max_value_bytes=100_000, ttl_seconds=60
        ),
        snapshot_hash_provider=lambda: "snap-0",
    )
    return TurnContext.begin(
        request=ChatRequest(
            conversation_id="c1",
            history=[],
            question="summarize this dataset",
            scope_package_ids=(PKG,),
        ),
        deps=deps,
    )


def _result() -> ResearchResult:
    return ResearchResult(
        candidate_answer="This dataset covers spending authorities.",
        terminal_reason="final_answer",
        packages_cited=[PKG],
    )


def test_explored_is_a_registered_outcome() -> None:
    assert "explored" in records.OUTCOMES


def test_build_does_not_coerce_explored_to_error() -> None:
    record = records.build(
        _ctx(),
        message="This dataset covers spending authorities.",
        result=_result(),
        outcome="explored",
    )
    assert record["outcome"] == "explored"


def test_explored_record_round_trips_through_validate() -> None:
    record = records.build(
        _ctx(),
        message="This dataset covers spending authorities.",
        result=_result(),
        outcome="explored",
    )
    assert records.validate(record) is not None
    kept = records.sanitize_incoming([record], max_records=10)
    assert len(kept) == 1
    assert kept[0]["outcome"] == "explored"


def test_an_unregistered_tag_still_coerces() -> None:
    """The coercion this file guards against is real, not hypothetical."""
    record = records.build(
        _ctx(), message="x", result=_result(), outcome="browsed"
    )
    assert record["outcome"] == "error"


def test_explored_turns_are_not_replay_cached() -> None:
    """Deliberate, per the PRD's known limitations: the replay gate is
    strictly `answered`, so an exploration is *omitted* from the cache
    rather than corrupting it. Re-running the same exploration pays full
    price. Pinned here so widening the gate is a decision someone makes,
    not a side effect of adding an outcome."""
    cache = memory.ReplayCacheV2(max_entries=4, ttl_seconds=60)
    done: list[Any] = [
        agent_events.Done(
            turn_id="t1", total_tool_calls=1, total_dollars=0.0, elapsed_ms=1
        )
    ]
    assert (
        cache.put(
            "k", events=done, outcome="explored", snapshot_hash="snap-0"
        )
        == "not_answered"
    )
    assert (
        cache.put(
            "k", events=done, outcome="answered", snapshot_hash="snap-0"
        )
        == "stored"
    )
