"""openparliament.ca API reads for the people build.

Volunteer-run: requests identify the job, list pages are fetched as
they come, and per-person detail fetches are paced
(`detail_requests_per_second`). Only people without stored detail, plus
sitting MPs, are fetched on a run, so a weekly run is a few hundred
requests after the first.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Protocol, runtime_checkable

from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter


class OpenParliamentError(RuntimeError):
    """The API refused or returned something unusable. Fails the run."""


class _TransientError(RuntimeError):
    pass


@runtime_checkable
class OpenParliamentClient(Protocol):
    def list_all(self, path: str) -> list[dict[str, Any]]:
        """Every object of a paged list endpoint."""

    def detail(self, path: str) -> dict[str, Any]:
        """One object; paced."""


class RealOpenParliamentClient:
    def __init__(self, *, base_url: str, user_agent: str, timeout_s: float, detail_rps: float) -> None:
        self._base = base_url.rstrip("/")
        self._ua = user_agent
        self._timeout = timeout_s
        self._min_gap = 1.0 / detail_rps if detail_rps > 0 else 0.0
        self._last = 0.0

    def list_all(self, path: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        sep = "&" if "?" in path else "?"
        next_path: str | None = f"{path}{sep}format=json&limit=500"
        while next_path:
            body = self._get(next_path)
            objects = body.get("objects")
            if not isinstance(objects, list):
                raise OpenParliamentError(f"{path}: unexpected list shape")
            out.extend(o for o in objects if isinstance(o, dict))
            next_path = (body.get("pagination") or {}).get("next_url")
        return out

    def detail(self, path: str) -> dict[str, Any]:
        wait = self._last + self._min_gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        sep = "&" if "?" in path else "?"
        return self._get(f"{path}{sep}format=json")

    def _get(self, path: str) -> dict[str, Any]:
        url = f"{self._base}{path}"
        try:
            for attempt in Retrying(
                stop=stop_after_attempt(4),
                wait=wait_exponential_jitter(initial=1.0, max=20.0),
                retry=retry_if_exception_type(_TransientError),
                reraise=True,
            ):
                with attempt:
                    return self._once(url)
        except _TransientError as exc:
            raise OpenParliamentError(str(exc)) from exc
        raise OpenParliamentError("unreachable")  # pragma: no cover

    def _once(self, url: str) -> dict[str, Any]:
        req = urllib.request.Request(url, headers={"User-Agent": self._ua, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 429 or exc.code >= 500:
                raise _TransientError(f"openparliament.ca {exc.code}") from exc
            raise OpenParliamentError(f"openparliament.ca {exc.code} for {url}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise _TransientError(f"openparliament.ca unreachable: {exc}") from exc
        try:
            body = json.loads(raw)
        except ValueError as exc:
            raise OpenParliamentError(f"non-JSON from {url}") from exc
        if not isinstance(body, dict):
            raise OpenParliamentError(f"unexpected shape from {url}")
        return body
