"""`run_turn_traced` — turn-span lifecycle around the loop generator.

The driver is tested against a stubbed `run_turn` so the assertions
target span behaviour only: open/close on drain, close on abandonment,
no-op passthrough when tracing is unconfigured, and the metadata /
output payloads."""
from __future__ import annotations

from typing import Any

from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_events, agent_tracing
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_loop import ChatRequest, LoopDeps
from tests.unit.conftest import FakeBraintrustModule


def _make_deps(**overrides: Any) -> tuple[LoopDeps, list[int]]:
    provider_calls: list[int] = []

    def provider() -> str:
        provider_calls.append(1)
        return "snap-hash"

    deps = LoopDeps(
        bq=object(),
        openai_client=object(),
        settings=Settings(),
        system_prompt="prompt text",
        prompt_hash="ph-123",
        cache=ResponseCache(
            max_entries=4, max_value_bytes=100_000, ttl_seconds=60
        ),
        snapshot_hash_provider=provider,
        system_prompt_tokens=42,
        **overrides,
    )
    return deps, provider_calls


def _request() -> ChatRequest:
    return ChatRequest(
        conversation_id="conv-1", history=[], question="how many rows?"
    )


def _happy_events() -> list[agent_events.AgentEvent]:
    return [
        agent_events.TurnStart(
            conversation_id="conv-1", turn_id="t1", cached=False
        ),
        agent_events.CostUpdate(
            tokens_in_total=100, tokens_out_total=10, dollars_spent=0.001
        ),
        agent_events.CostUpdate(
            tokens_in_total=250, tokens_out_total=30, dollars_spent=0.002
        ),
        agent_events.MessageDelta(delta="final "),
        agent_events.MessageDelta(delta="answer"),
        agent_events.Done(
            turn_id="t1",
            total_tool_calls=2,
            total_dollars=0.002,
            elapsed_ms=12,
        ),
    ]


def _stub_run_turn(events: list[agent_events.AgentEvent], seen: dict[str, Any]):
    def stub(*, request: ChatRequest, deps: LoopDeps):
        seen["request"] = request
        seen["deps"] = deps
        yield from events

    return stub


def test_passthrough_when_tracing_unconfigured(
    monkeypatch, tracing_disabled
) -> None:
    deps, provider_calls = _make_deps()
    events = _happy_events()
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        agent_tracing, "run_turn", _stub_run_turn(events, seen)
    )

    out = list(agent_tracing.run_turn_traced(request=_request(), deps=deps))

    # Identical objects, untouched deps, no snapshot pre-fetch: the
    # traced driver must be invisible when tracing is off.
    assert out == events
    assert all(a is b for a, b in zip(out, events, strict=True))
    assert seen["deps"] is deps
    assert provider_calls == []


def test_turn_span_opens_and_closes_around_drained_generator(
    monkeypatch, fake_braintrust: FakeBraintrustModule
) -> None:
    deps, _ = _make_deps()
    events = _happy_events()
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        agent_tracing, "run_turn", _stub_run_turn(events, seen)
    )

    out = list(
        agent_tracing.run_turn_traced(
            request=_request(), deps=deps, session_parent="export:session:1"
        )
    )

    assert out == events
    assert len(fake_braintrust.spans) == 1
    span = fake_braintrust.spans[0]
    assert span.name == "agent.run_turn"
    assert span.ended is True
    assert span.kwargs["parent"] == "export:session:1"
    assert span.kwargs["input"] == {
        "question": "how many rows?",
        "conversation_id": "conv-1",
    }
    # The loop receives a per-turn deps copy carrying the exported turn
    # span for explicit tool-span parenting.
    assert seen["deps"] is not deps
    assert str(seen["deps"].trace_parent).startswith("export:agent.run_turn")

    final = span.logs[-1]
    assert final["output"] == {
        "final_message": "final answer",
        "tool_call_count": 2,
        "dollars_spent": 0.002,
        "terminal": "done",
    }
    assert final["metadata"]["turn_id"] == "t1"
    assert final["metadata"]["cached"] is False
    assert final["metadata"]["tokens_in_per_call"] == [100, 150]
    assert final["metadata"]["tokens_out_per_call"] == [10, 20]


def test_turn_span_metadata_carries_hashes_and_loop_impl(
    monkeypatch, fake_braintrust: FakeBraintrustModule
) -> None:
    deps, provider_calls = _make_deps()
    monkeypatch.setattr(
        agent_tracing, "run_turn", _stub_run_turn(_happy_events(), {})
    )

    list(agent_tracing.run_turn_traced(request=_request(), deps=deps))

    metadata = fake_braintrust.spans[0].kwargs["metadata"]
    assert metadata == {
        "prompt_hash": "ph-123",
        "snapshot_hash": "snap-hash",
        "loop_impl": "v1",
        "system_prompt_tokens": 42,
    }
    assert provider_calls == [1]


def test_snapshot_provider_shared_with_loop_is_memoized(
    monkeypatch, fake_braintrust: FakeBraintrustModule
) -> None:
    deps, provider_calls = _make_deps()

    def stub(*, request: ChatRequest, deps: LoopDeps):
        # The loop's own cache-key lookup re-calls the provider; the
        # memo must serve it without a second BQ round-trip.
        assert deps.snapshot_hash_provider() == "snap-hash"
        assert deps.snapshot_hash_provider() == "snap-hash"
        yield from _happy_events()

    monkeypatch.setattr(agent_tracing, "run_turn", stub)
    list(agent_tracing.run_turn_traced(request=_request(), deps=deps))
    assert provider_calls == [1]


def test_turn_span_closes_when_iterator_abandoned(
    monkeypatch, fake_braintrust: FakeBraintrustModule
) -> None:
    deps, _ = _make_deps()
    monkeypatch.setattr(
        agent_tracing, "run_turn", _stub_run_turn(_happy_events(), {})
    )

    gen = agent_tracing.run_turn_traced(request=_request(), deps=deps)
    next(gen)
    next(gen)
    gen.close()  # client disconnect

    span = fake_braintrust.spans[0]
    assert span.ended is True
    assert span.logs[-1]["output"]["terminal"] == "abandoned"


def test_error_and_timeout_terminals(
    monkeypatch, fake_braintrust: FakeBraintrustModule
) -> None:
    for reason, expected_terminal in (
        ("turn_timeout", "timeout"),
        ("openai_error", "error"),
    ):
        events: list[agent_events.AgentEvent] = [
            agent_events.TurnStart(
                conversation_id="conv-1", turn_id="t9", cached=False
            ),
            agent_events.ErrorEvent(
                message="boom", retryable=False, reason=reason
            ),
        ]
        deps, _ = _make_deps()
        monkeypatch.setattr(
            agent_tracing, "run_turn", _stub_run_turn(events, {})
        )
        list(agent_tracing.run_turn_traced(request=_request(), deps=deps))
        span = fake_braintrust.spans[-1]
        assert span.logs[-1]["output"]["terminal"] == expected_terminal


def test_span_recurrented_on_each_resumption(
    monkeypatch, fake_braintrust: FakeBraintrustModule
) -> None:
    deps, _ = _make_deps()
    events = _happy_events()
    monkeypatch.setattr(
        agent_tracing, "run_turn", _stub_run_turn(events, {})
    )

    list(agent_tracing.run_turn_traced(request=_request(), deps=deps))

    # One set/unset pair per resumption (len(events) yields + the final
    # StopIteration probe) keeps wrap_openai auto-spans attached even
    # when the caller iterates from a threadpool.
    span = fake_braintrust.spans[0]
    assert span.set_current_calls == len(events) + 1
    assert span.set_current_calls == span.unset_current_calls
