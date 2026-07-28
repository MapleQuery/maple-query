"""What `list_documents` returns once a document's header is recovered.

The documents here are real: their rows come from the same warehouse
fixture the detector was built against, so a change that breaks recovery
on an actual federal spreadsheet fails here rather than passing against a
hand-written shape that happens to match the code.

The load-bearing assertion is the *negative* one — `columns` never
changes. SQL addresses the stored keys, and the moment the payload starts
renaming them in place every consumer has to know which of the two forms
it is holding.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from semantic_enrich.clients.bq import BoundedQueryResult
from semantic_enrich.config.settings import Settings
from semantic_enrich.core import agent_tools
from tests.integration.conftest import FakeBqClient
from tests.integration.openai_fakes import FakeOpenAIClient

_FIXTURE = (
    Path(__file__).parent.parent / "fixtures" / "header_recovery_documents.json"
)
_DOCUMENTS: list[dict[str, Any]] = json.loads(
    _FIXTURE.read_text(encoding="utf-8")
)

# Housing benefit table 1: a blank row, then
# Province/Territory | Number of Unique Applicants | Total Amount ($000).
RECOVERABLE = "43d968b125"
# Legal aid expenditures: a genuine three-tier header. Declines.
DECLINING = "57f2ef5417"


def _fixture(prefix: str) -> dict[str, Any]:
    matches = [d for d in _DOCUMENTS if d["document_id"].startswith(prefix)]
    assert len(matches) == 1
    return matches[0]


def _settings(**overrides: Any) -> Settings:
    return Settings(
        gcp_project_id="proj",
        openai_api_key="sk-test",  # type: ignore[arg-type]
    ).model_copy(update=overrides)


def _ctx(
    *, bq: FakeBqClient, recovery: bool
) -> agent_tools.ToolContext:
    return agent_tools.ToolContext(
        bq=bq,
        openai_client=FakeOpenAIClient(vector_factory=lambda _t: [0.0] * 1536),
        settings=_settings(agent_header_recovery=recovery),
        state=agent_tools.LoopState(
            conversation_id="c1", turn_id="t1", question="q"
        ),
        emit=lambda _e: None,
    )


def _bq_for(
    docs: list[tuple[str, list[dict[str, Any]]]],
) -> FakeBqClient:
    """Wire a fake warehouse holding `(doc_id, row bodies)` pairs."""
    bq = FakeBqClient()
    bq.register_query(
        "load_status = 'loaded'",
        [
            {
                "document_id": doc_id,
                "package_id": "pkg-1",
                "title": f"title-{doc_id}",
                "row_count": 100,
                "resource_last_modified": None,
            }
            for doc_id, _rows in docs
        ],
    )
    bq.register_bounded_query(
        "TO_JSON_STRING(row)",
        BoundedQueryResult(
            rows=[
                {
                    "document_id": doc_id,
                    "row_index": index,
                    "row_json": json.dumps(body),
                }
                for doc_id, rows in docs
                for index, body in enumerate(rows)
            ],
            total_bytes_billed=1024,
            slot_ms=1,
            elapsed_ms=3,
            timed_out=False,
            error=None,
        ),
    )
    return bq


def _list(bq: FakeBqClient, *, recovery: bool) -> dict[str, Any]:
    ctx = _ctx(bq=bq, recovery=recovery)
    ctx.state.known_package_ids.add("pkg-1")
    result = agent_tools.run_list_documents(
        ctx=ctx, args={"package_ids": ["pkg-1"]}
    )
    result["_state"] = ctx.state
    return result


def _only(result: dict[str, Any]) -> dict[str, Any]:
    docs = result["documents"]
    assert len(docs) == 1
    return dict(docs[0])


CLEAN_ROWS: list[dict[str, Any]] = [
    {"Province": "ON", "Amount": "5"},
    {"Province": "QC", "Amount": "7"},
    {"Province": "BC", "Amount": "9"},
]


def test_recovered_document_gains_names_and_keeps_its_columns() -> None:
    doc = _fixture(RECOVERABLE)
    bq = _bq_for([("doc-1", doc["rows"])])
    entry = _only(_list(bq, recovery=True))

    assert entry["column_names_recovered"] == {
        "__col_1": "Number of Unique Applicants",
        "__col_2": "Total Amount ($000)",
    }
    assert entry["header_row_index"] == 1
    assert entry["quality"] == "recovered_headers"
    # The keys SQL addresses are untouched.
    assert "__col_1" in entry["columns"]
    assert "__col_2" in entry["columns"]
    assert "Number of Unique Applicants" not in entry["columns"]


def test_recovered_document_is_not_demoted_in_the_listing() -> None:
    """A demoted doc sorts last and counts as unusable. A recovered one
    is readable with an asterisk, so it does neither."""
    doc = _fixture(RECOVERABLE)
    bq = _bq_for([("doc-1", doc["rows"])])
    result = _list(bq, recovery=True)
    assert "guidance" not in result


def test_declined_document_is_indistinguishable_from_today() -> None:
    doc = _fixture(DECLINING)
    bq = _bq_for([("doc-1", doc["rows"])])
    entry = _only(_list(bq, recovery=True))

    assert entry["quality"] == "low_generated_headers"
    assert "column_names_recovered" not in entry
    assert "header_row_index" not in entry


def test_clean_document_payload_is_identical_either_way() -> None:
    """The whole cost claim for the clean path, asserted rather than
    stated: a document that needs no recovery gets a byte-identical
    entry whether the feature is on or off."""
    off = _only(_list(_bq_for([("doc-1", CLEAN_ROWS)]), recovery=False))
    on = _only(_list(_bq_for([("doc-1", CLEAN_ROWS)]), recovery=True))
    assert off == on
    assert "quality" not in on
    assert "column_names_recovered" not in on


def test_kill_switch_reproduces_the_current_build() -> None:
    doc = _fixture(RECOVERABLE)
    entry = _only(_list(_bq_for([("doc-1", doc["rows"])]), recovery=False))
    assert entry["quality"] == "low_generated_headers"
    assert "column_names_recovered" not in entry
    assert "header_row_index" not in entry


def test_read_widens_only_when_recovery_is_on() -> None:
    """Recovery needs rows the sample window does not reach — and rows
    *below* the header, since that contrast is what distinguishes a
    header from a banner. With the feature off the bound is untouched."""
    settings = _settings()
    doc = _fixture(RECOVERABLE)

    off_bq = _bq_for([("doc-1", doc["rows"])])
    _list(off_bq, recovery=False)
    assert off_bq.bounded_params[0]["n"] == settings.agent_sample_values_rows

    on_bq = _bq_for([("doc-1", doc["rows"])])
    _list(on_bq, recovery=True)
    assert on_bq.bounded_params[0]["n"] == settings.agent_header_scan_rows


def test_widened_read_does_not_widen_the_sample_values() -> None:
    """Reading 8 rows must not put 8 rows of samples in the payload —
    the extra rows exist for the detector, not for the model."""
    rows = [{"Province": f"p{i}", "Amount": str(i)} for i in range(8)]
    entry = _only(_list(_bq_for([("doc-1", rows)]), recovery=True))
    settings = _settings()
    assert (
        len(entry["column_samples"]["Province"])
        == settings.agent_sample_values_rows
    )


def test_recovery_survives_a_mixed_listing() -> None:
    """One recoverable, one declining, one clean, in one call."""
    bq = _bq_for(
        [
            ("doc-good", _fixture(RECOVERABLE)["rows"]),
            ("doc-bad", _fixture(DECLINING)["rows"]),
            ("doc-clean", CLEAN_ROWS),
        ]
    )
    documents = _list(bq, recovery=True)["documents"]
    by_id = {d["document_id"]: d for d in documents}
    assert by_id["doc-good"]["quality"] == "recovered_headers"
    assert by_id["doc-bad"]["quality"] == "low_generated_headers"
    assert "quality" not in by_id["doc-clean"]
    # Only the still-garbage doc is demoted to the back.
    assert [d["document_id"] for d in documents][-1] == "doc-bad"


def test_a_document_whose_rows_have_a_gap_is_not_recovered() -> None:
    """Row bodies are positional. If row 1 failed to parse, every index
    below it shifts, and `header_row_index` becomes an off-by-one that
    later excludes the wrong rows from an aggregate. Drop it instead."""
    doc = _fixture(RECOVERABLE)
    bq = FakeBqClient()
    bq.register_query(
        "load_status = 'loaded'",
        [
            {
                "document_id": "doc-1",
                "package_id": "pkg-1",
                "title": "gapped",
                "row_count": 100,
                "resource_last_modified": None,
            }
        ],
    )
    bq.register_bounded_query(
        "TO_JSON_STRING(row)",
        BoundedQueryResult(
            rows=[
                {
                    "document_id": "doc-1",
                    "row_index": index,
                    "row_json": json.dumps(body),
                }
                # Row 1 — the header row — never arrives.
                for index, body in enumerate(doc["rows"])
                if index != 1
            ],
            total_bytes_billed=1024,
            slot_ms=1,
            elapsed_ms=3,
            timed_out=False,
            error=None,
        ),
    )
    entry = _only(_list(bq, recovery=True))
    assert "column_names_recovered" not in entry
