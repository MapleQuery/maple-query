"""openparliament.ca API client: MPs, votes, ballots, bills, speeches.

The House of Commons' own record (who voted how, what was said, which
bills passed) is published by openparliament.ca as a JSON API built from
ourcommons.ca and LEGISinfo. It is a volunteer-run project, not a
government service, so this client is deliberately polite: it identifies
itself, caches the lists that change slowly (MPs, a session's bills and
votes) and never fans out more than a handful of detail requests a turn.

Two quirks shape the callers: the API ignores `name=` and `q=` filters on
politicians and bills, so those lists are fetched whole (a few hundred
rows) and matched locally; and votes, ballots and speeches *do* filter
server-side by politician, bill, session and date.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from typing import Any, Protocol, runtime_checkable

from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

API_BASE = "https://api.openparliament.ca"
SITE_BASE = "https://openparliament.ca"
USER_AGENT = "MapleQuery/1.0 (+https://maple-query.vercel.app; research agent over Canadian public data)"


class ParliamentError(RuntimeError):
    """The API refused, failed, or answered with something unexpected."""


class _TransientError(RuntimeError):
    """5xx / connection failure — worth one more attempt."""


@runtime_checkable
class ParliamentClient(Protocol):
    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """One API object (detail endpoint)."""

    def list_all(
        self, path: str, params: dict[str, Any] | None = None, *, max_rows: int = 2000
    ) -> list[dict[str, Any]]:
        """Every object of a list endpoint, following pagination."""

    def list_cached(
        self, path: str, params: dict[str, Any] | None = None, *, ttl_s: float = 6 * 3600
    ) -> list[dict[str, Any]]:
        """`list_all`, memoised for `ttl_s`."""


class RealParliamentClient:
    def __init__(self, *, base_url: str = API_BASE, timeout_s: float = 20.0, cache_size: int = 64) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout_s
        self._lock = threading.Lock()
        self._lists: OrderedDict[str, tuple[float, list[dict[str, Any]]]] = OrderedDict()
        self._cap = cache_size

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        body = self._request(path, params)
        if not isinstance(body, dict):
            raise ParliamentError(f"{path}: unexpected response shape")
        return body

    def list_all(
        self, path: str, params: dict[str, Any] | None = None, *, max_rows: int = 2000
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        query = {**(params or {}), "limit": min(500, max_rows)}
        next_path: str | None = path
        while next_path and len(out) < max_rows:
            body = self._request(next_path, query)
            objects = body.get("objects") if isinstance(body, dict) else None
            if not isinstance(objects, list):
                raise ParliamentError(f"{path}: unexpected list shape")
            out.extend(o for o in objects if isinstance(o, dict))
            next_path = (body.get("pagination") or {}).get("next_url")
            query = {}  # next_url carries its own query string
        return out[:max_rows]

    def list_cached(
        self, path: str, params: dict[str, Any] | None = None, *, ttl_s: float = 6 * 3600
    ) -> list[dict[str, Any]]:
        key = path + "?" + urllib.parse.urlencode(sorted((params or {}).items()))
        with self._lock:
            hit = self._lists.get(key)
            if hit is not None and time.monotonic() - hit[0] < ttl_s:
                self._lists.move_to_end(key)
                return hit[1]
        rows = self.list_all(path, params, max_rows=5000)
        with self._lock:
            self._lists[key] = (time.monotonic(), rows)
            while len(self._lists) > self._cap:
                self._lists.popitem(last=False)
        return rows

    # ── transport ──

    def _request(self, path: str, params: dict[str, Any] | None) -> Any:
        query = {"format": "json", **(params or {})}
        sep = "&" if "?" in path else "?"
        url = f"{self._base}{path}{sep}{urllib.parse.urlencode(query)}"
        try:
            for attempt in Retrying(
                stop=stop_after_attempt(2),
                wait=wait_exponential_jitter(initial=0.5, max=2.0),
                retry=retry_if_exception_type(_TransientError),
                reraise=True,
            ):
                with attempt:
                    return self._once(url)
        except _TransientError as exc:
            raise ParliamentError(str(exc)) from exc
        raise ParliamentError("unreachable")  # pragma: no cover

    def _once(self, url: str) -> Any:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code >= 500 or exc.code == 429:
                raise _TransientError(f"openparliament.ca {exc.code}") from exc
            raise ParliamentError(f"openparliament.ca {exc.code} for {url}") from exc
        except TimeoutError as exc:
            raise ParliamentError(f"openparliament.ca took over {self._timeout:.0f}s") from exc
        except (urllib.error.URLError, ConnectionError) as exc:
            raise _TransientError(f"openparliament.ca unreachable: {exc}") from exc
        try:
            return json.loads(raw)
        except ValueError as exc:
            raise ParliamentError("openparliament.ca returned non-JSON") from exc
