"""Live end-to-end eval against the deployed agent.

Asks every question in a fixture (default eval/questions-live-sources.yaml)
through the real /chat stream, records what each turn did (route, sources
read, cost, time, answer), and, given a baseline report, flags the turns
that got worse. This is the check that caught routing sending federal
health spending to itemized grants before it shipped for good.

It spends real OpenAI money (about $0.08 a question), so it stops starting
new questions once the spend cap would be crossed.

    uv run python scripts/live_eval.py                       # all sets
    uv run python scripts/live_eval.py --set stay_great
    uv run python scripts/live_eval.py --baseline eval/reports/live-sources-baseline-2026-10-08.json
    uv run python scripts/live_eval.py --ids ukraine-aid,tariff-revenue --max-dollars 0.5

By default it goes through the web app's relay (no token needed). Pass
--base https://agent-service-...run.app with MAPLEQUERY_API_TOKEN set to
call Cloud Run directly.

Reports land in eval/reports/ (gitignored); commit a deliberate baseline
with `git add -f`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

SERVICE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURE = SERVICE_DIR / "eval" / "questions-live-sources.yaml"
REPORTS = SERVICE_DIR / "eval" / "reports"
DEFAULT_BASE = "https://maple-query.vercel.app/api/mq"
# Budgeted per question before it runs, so a cap is never overshot by more
# than the questions already in flight.
EST_DOLLARS_PER_QUESTION = 0.12

_DATA_EVENTS = {"source_data", "sql_executed"}


def ask(base: str, token: str, question: str, timeout: float) -> dict[str, Any]:
    body = json.dumps(
        {
            "conversation_id": str(uuid.uuid4()),
            "question": question,
            "history": [],
            # A repeat within the cache TTL would replay, measuring nothing.
            "bypass_cache": True,
        }
    ).encode()
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{base}/chat", data=body, headers=headers, method="POST")
    started = time.monotonic()
    answer, done, error = "", None, None
    route: dict[str, Any] = {}
    sources: list[dict[str, Any]] = []
    tool_errors: list[str] = []
    read_data = False
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            name = None
            for raw in resp:
                line = raw.decode().rstrip("\n")
                if line.startswith("event:"):
                    name = line[6:].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                data = json.loads(line[5:])
                if name == "message_delta":
                    answer += data.get("delta", "")
                elif name == "done":
                    done = data
                elif name == "error":
                    error = data.get("reason") or data.get("message")
                elif name == "triage_result":
                    route = {
                        "category": data.get("category"),
                        "source": data.get("source"),
                        "source_confidence": data.get("source_confidence"),
                    }
                elif name == "tool_error":
                    tool_errors.append(f"{data.get('tool')}: {str(data.get('message'))[:160]}")
                elif name == "source_data":
                    sources.append(
                        {
                            "source": data.get("source"),
                            "table": data.get("title"),
                            "rows": data.get("row_count"),
                        }
                    )
                    read_data = read_data or (data.get("row_count") or 0) > 0
                elif name == "sql_executed":
                    sources.append({"source": "warehouse", "table": "run_sql", "rows": data.get("row_count")})
                    read_data = read_data or (data.get("row_count") or 0) > 0
    except Exception as exc:  # a failed turn is a result, not a crash
        error = f"transport: {exc}"
    return {
        "secs": round(time.monotonic() - started, 1),
        "dollars": round((done or {}).get("total_dollars") or 0.0, 4),
        "tool_calls": (done or {}).get("total_tool_calls"),
        "route": route,
        "sources": sources,
        "read_data": read_data,
        "tool_errors": tool_errors,
        "error": error,
        "answer": answer,
    }


def warm_up(base: str, token: str, timeout: float) -> float:
    """Wake agent-service before timing anything. It scales to zero, and a
    cold start (~35 s) would read as a regression on whichever question
    ran first. An empty body is rejected by validation (422) before any
    model call, so this costs nothing."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{base}/chat", data=b"{}", headers=headers, method="POST")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
    except urllib.error.HTTPError:
        pass  # 422 is the expected answer; it means the service is up.
    except Exception as exc:  # the eval itself will surface a real outage
        print(f"warm-up failed: {exc}", file=sys.stderr)
    return round(time.monotonic() - started, 1)


def regressions(entry: dict[str, Any], before: dict[str, Any] | None) -> list[str]:
    """Why this turn looks worse than its baseline (or its contract)."""
    flags: list[str] = []
    if entry.get("error"):
        flags.append(f"error: {entry['error']}")
    expected = entry.get("expected_source")
    got = (entry.get("route") or {}).get("source")
    if expected and got and got != expected:
        flags.append(f"routed to {got}, expected {expected}")
    want_cat = entry.get("expected_category")
    got_cat = (entry.get("route") or {}).get("category")
    if want_cat and got_cat != want_cat:
        flags.append(f"triaged {got_cat}, expected {want_cat}")
    if want_cat and want_cat != "in_scope":
        return flags  # a refusal reads no data by design
    if before is None:
        if entry["set"] == "stay_great" and not entry.get("read_data"):
            flags.append("read no data")
        return flags
    if before.get("read_data") and not entry.get("read_data"):
        flags.append("read data before, none now")
    if before.get("secs") and entry["secs"] > 2 * before["secs"] + 10:
        flags.append(f"slower: {before['secs']}s -> {entry['secs']}s")
    if before.get("dollars") and entry["dollars"] > 1.5 * before["dollars"] + 0.03:
        flags.append(f"costlier: ${before['dollars']} -> ${entry['dollars']}")
    return flags


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    ap.add_argument("--set", dest="sets", action="append", help="stay_great | get_better | known_gap")
    ap.add_argument("--ids", help="comma-separated question ids")
    ap.add_argument("--base", default=os.environ.get("MAPLEQUERY_EVAL_BASE", DEFAULT_BASE))
    ap.add_argument("--max-dollars", type=float, default=2.0)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--baseline", type=Path, help="a previous report to diff against")
    ap.add_argument("--out", type=Path, help="report path (default eval/reports/live-<timestamp>.json)")
    ap.add_argument("--no-warmup", action="store_true", help="time the cold start too")
    args = ap.parse_args()

    questions: list[dict[str, Any]] = yaml.safe_load(args.fixture.read_text())
    # --set and --ids add up: "the stay_great set plus these new ones".
    if args.sets or args.ids:
        wanted = set(args.ids.split(",")) if args.ids else set()
        questions = [q for q in questions if q["set"] in (args.sets or []) or q["id"] in wanted]
    if not questions:
        print("no questions selected", file=sys.stderr)
        return 2

    baseline: dict[str, dict[str, Any]] = {}
    if args.baseline:
        baseline = {r["id"]: r for r in json.loads(args.baseline.read_text())["results"]}

    token = os.environ.get("MAPLEQUERY_API_TOKEN", "")
    lock = threading.Lock()
    committed = 0.0  # spent + reserved for questions in flight

    def run(q: dict[str, Any]) -> dict[str, Any] | None:
        nonlocal committed
        with lock:
            if committed + EST_DOLLARS_PER_QUESTION > args.max_dollars:
                return None
            committed += EST_DOLLARS_PER_QUESTION
        result = ask(args.base, token, q["question"], args.timeout)
        with lock:
            committed += result["dollars"] - EST_DOLLARS_PER_QUESTION
        keys = ("id", "set", "question", "expected_source", "expected_category")
        entry = {**{k: q.get(k) for k in keys}, **result}
        entry["flags"] = regressions(entry, baseline.get(q["id"]))
        mark = "!!" if entry["flags"] else "ok"
        print(f"  {mark} {q['id']} ({entry['secs']}s, ${entry['dollars']})", flush=True)
        return entry

    print(f"{len(questions)} questions via {args.base}, cap ${args.max_dollars:.2f}", flush=True)
    if not args.no_warmup:
        print(f"  warm-up {warm_up(args.base, token, args.timeout)}s", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        results = list(pool.map(run, questions))
    skipped = [q["id"] for q, r in zip(questions, results, strict=True) if r is None]
    done = [r for r in results if r is not None]

    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    out = args.out or REPORTS / f"live-{stamp}.json"
    out.write_text(
        json.dumps(
            {
                "run_at": stamp,
                "endpoint": args.base,
                "fixture": str(args.fixture.relative_to(SERVICE_DIR)),
                "baseline": str(args.baseline) if args.baseline else None,
                "total_dollars": round(sum(r["dollars"] for r in done), 4),
                "skipped_for_budget": skipped,
                "results": done,
            },
            indent=1,
        )
    )

    print()
    print("| id | set | route | data | secs | $ | flags |")
    print("|---|---|---|---|---|---|---|")
    for r in done:
        before = baseline.get(r["id"])
        secs = f"{before['secs']}→{r['secs']}" if before else str(r["secs"])
        cost = f"{before['dollars']}→{r['dollars']}" if before else str(r["dollars"])
        route = (r.get("route") or {}).get("source") or "-"
        print(
            f"| {r['id']} | {r['set']} | {route} | {'yes' if r['read_data'] else 'no'} "
            f"| {secs} | {cost} | {'; '.join(r['flags']) or ''} |"
        )
    print()
    print(f"spent ${sum(r['dollars'] for r in done):.2f}; report: {out}")
    if skipped:
        print(f"skipped for budget: {', '.join(skipped)}")
    flagged = [r for r in done if r["flags"] and r["set"] == "stay_great"]
    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
