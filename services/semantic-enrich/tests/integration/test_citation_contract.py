"""Citation contract: dataset references in the final answer are
markdown links to `/datasets/<package_id>` with the title and id taken
verbatim from the turn's tool results.

Regression for the dead UUID-autolink behaviour: the web app used to
linkify raw UUIDs in code spans; once titles replaced UUIDs that
heuristic went dead and answers lost their dataset links. The link is
now part of the answer contract — model-emitted and FE-independent —
so a scripted turn's answer is checked the same way a live one would
be.
"""
from __future__ import annotations

import math
import re
from typing import Any

from semantic_enrich.config.settings import Settings
from semantic_enrich.core.agent.phases import PipelineDeps
from semantic_enrich.core.agent.pipeline import run_turn_collected
from semantic_enrich.core.agent_cache import ResponseCache
from semantic_enrich.core.agent_request import ChatRequest
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient

_LINK = re.compile(r"\[([^\]]+)\]\(/datasets/([^)]+)\)")

_CANDIDATES = [
    {
        "package_id": "b2c3d4e5-1111-2222-3333-444455556666",
        "title": "Housing Starts",
        "summary": "housing",
        "grain": None,
        "measures": [],
        "dimensions": [],
        "distance": 0.1,
    },
    {
        "package_id": "f6e5d4c3-9999-8888-7777-666655554444",
        "title": "Rental Market Survey",
        "summary": "rentals",
        "grain": None,
        "measures": [],
        "dimensions": [],
        "distance": 0.2,
    },
]


def _outcome_for_answer(answer: str) -> Any:
    settings = Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
        agent_cache_replay_delay_ms=0,
    )
    bq = FakeBqClient()
    bq.register_query("VECTOR_SEARCH", list(_CANDIDATES))
    openai = FakeOpenAIClient(
        vector_factory=lambda _t: [1.0 / math.sqrt(1536)] * 1536,
        chat_script=[
            {
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "search_datasets",
                        "arguments": {"query": "housing"},
                    }
                ]
            },
            {"content": answer},
        ],
    )
    deps = PipelineDeps(
        bq=bq,
        openai_client=openai,
        settings=settings,
        system_prompt="p",
        prompt_hash="h",
        cache=ResponseCache(
            max_entries=10, max_value_bytes=1_000_000, ttl_seconds=60
        ),
        snapshot_hash_provider=lambda: "snap-0",
    )
    return run_turn_collected(
        request=ChatRequest(
            conversation_id="c1", history=[], question="housing?"
        ),
        deps=deps,
    )


def _assert_citations_well_formed(message: str) -> None:
    """Every candidate whose title the answer cites appears as a
    `[<title>](/datasets/<package_id>)` link with the exact id from
    the tool results, and no raw UUID leaks outside a link."""
    links = dict(_LINK.findall(message))
    for candidate in _CANDIDATES:
        title, pid = str(candidate["title"]), str(candidate["package_id"])
        if title not in message:
            continue
        assert links.get(title) == pid, (
            f"cited dataset {title!r} lacks a verbatim /datasets link"
        )
    outside_links = _LINK.sub("", message)
    for candidate in _CANDIDATES:
        assert str(candidate["package_id"]) not in outside_links, (
            "raw package UUID leaked outside a markdown link"
        )


def test_cited_datasets_carry_verbatim_links() -> None:
    outcome = _outcome_for_answer(
        "Housing starts rose 4% "
        "([Housing Starts](/datasets/b2c3d4e5-1111-2222-3333-444455556666)); "
        "rents were flat "
        "([Rental Market Survey](/datasets/f6e5d4c3-9999-8888-7777-666655554444))."
    )
    assert outcome.events[-1].event_type == "done"
    _assert_citations_well_formed(outcome.final_message)


def test_helper_rejects_bare_title_citation() -> None:
    import pytest

    with pytest.raises(AssertionError, match="lacks a verbatim"):
        _assert_citations_well_formed(
            "Housing Starts rose 4% this quarter."
        )


def test_helper_rejects_raw_uuid_citation() -> None:
    import pytest

    with pytest.raises(AssertionError, match="raw package UUID"):
        _assert_citations_well_formed(
            "See b2c3d4e5-1111-2222-3333-444455556666 for details."
        )
