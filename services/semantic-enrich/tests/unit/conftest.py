"""Unit-test fixtures for the span-wrapping tests.

`fake_braintrust` swaps a recording stand-in for the `braintrust`
module and force-enables the tracing gate, so span topology is
assertable without a Braintrust account, API key, or network.
"""
from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from semantic_enrich.providers import braintrust_tracing


class FakeSpan:
    """Recording span: captures constructor kwargs, `log` calls, and
    lifecycle transitions."""

    def __init__(self, name: str | None, kwargs: dict[str, Any]) -> None:
        self.name = name
        self.kwargs = kwargs
        self.logs: list[dict[str, Any]] = []
        self.ended = False
        self.set_current_calls = 0
        self.unset_current_calls = 0

    def log(self, **event: Any) -> None:
        self.logs.append(event)

    def export(self) -> str:
        return f"export:{self.name}:{id(self)}"

    def set_current(self) -> None:
        self.set_current_calls += 1

    def unset_current(self) -> None:
        self.unset_current_calls += 1

    def __enter__(self) -> FakeSpan:
        return self

    def __exit__(self, *exc: object) -> bool:
        self.ended = True
        return False


class FakeBraintrustModule(ModuleType):
    """Just enough of the braintrust module surface for the wrappers:
    `start_span` recording every created span."""

    def __init__(self) -> None:
        super().__init__("braintrust")
        self.spans: list[FakeSpan] = []

    def start_span(
        self,
        *,
        name: str | None = None,
        parent: str | None = None,
        **event: Any,
    ) -> FakeSpan:
        span = FakeSpan(name, {"parent": parent, **event})
        self.spans.append(span)
        return span


@pytest.fixture
def fake_braintrust(
    monkeypatch: pytest.MonkeyPatch,
) -> FakeBraintrustModule:
    fake = FakeBraintrustModule()
    monkeypatch.setitem(sys.modules, "braintrust", fake)
    monkeypatch.setattr(braintrust_tracing, "_configured", True)
    monkeypatch.setattr(braintrust_tracing, "_enabled", True)
    return fake


@pytest.fixture
def tracing_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(braintrust_tracing, "_configured", True)
    monkeypatch.setattr(braintrust_tracing, "_enabled", False)
