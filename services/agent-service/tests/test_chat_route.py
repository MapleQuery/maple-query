"""`POST /chat` — SSE round-trip with a scripted OpenAI response.

Asserts the response is `text/event-stream`, the frames follow the
`event: <type>\\ndata: <json>` shape, and the stream terminates in a
`done` frame."""
from __future__ import annotations

import json

from fastapi.testclient import TestClient
from semantic_enrich.clients.openai import ChatCompletionResult

from tests.conftest import FIXED_TOKEN, FakeOpenAIClient


def _parse_frames(text: str) -> list[tuple[str, dict[str, object]]]:
    frames: list[tuple[str, dict[str, object]]] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block or block.startswith(":"):
            continue
        event_line: str | None = None
        data_line: str | None = None
        for line in block.splitlines():
            if line.startswith("event:"):
                event_line = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_line = line[len("data:") :].strip()
        if event_line is None or data_line is None:
            continue
        frames.append((event_line, json.loads(data_line)))
    return frames


def test_chat_streams_done(
    client: TestClient, fake_openai: FakeOpenAIClient
) -> None:
    fake_openai.chat_responses = [
        ChatCompletionResult(
            content="the answer is 42",
            tool_calls=[],
            tokens_in=20,
            tokens_out=5,
            finish_reason="stop",
        )
    ]
    with client.stream(
        "POST",
        "/chat",
        json={"conversation_id": "conv-1", "question": "meaning of life"},
        headers={"Authorization": f"Bearer {FIXED_TOKEN}"},
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers.get("cache-control") == "no-cache"
        assert r.headers.get("x-accel-buffering") == "no"
        body = r.read().decode("utf-8")

    frames = _parse_frames(body)
    types = [t for t, _ in frames]
    assert types[0] == "turn_start"
    assert "cost_update" in types
    assert "message_delta" in types
    assert types[-1] == "done"

    delta_frame = next(payload for t, payload in frames if t == "message_delta")
    assert delta_frame["delta"] == "the answer is 42"


def test_chat_rejects_invalid_body(client: TestClient) -> None:
    r = client.post(
        "/chat",
        json={"conversation_id": "", "question": ""},
        headers={"Authorization": f"Bearer {FIXED_TOKEN}"},
    )
    assert r.status_code == 422
