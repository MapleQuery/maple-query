"""open.canada.ca CKAN client: catalogue search and the DataStore.

The warehouse mirrors open.canada.ca CSVs, but not all of them: the
largest proactive-disclosure tables (grants and contributions at 2.3 GB,
contracts at 640 MB) never made it in, and re-ingesting a catalogue the
portal already serves is the cost the project is trying not to pay.
open.canada.ca loads those same files into CKAN's DataStore and answers
filtered reads over them in about a second. This client is that read.

Endpoints (public, unauthenticated, JSON):

- `package_search`   catalogue search
- `package_show`     one package's resources
- `datastore_search` filtered, paged reads of one DataStore resource

No SQL: open.canada.ca does not expose `datastore_search_sql`, so
aggregation happens on our side over the filtered rows.
"""
from __future__ import annotations

import json
import threading
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

CKAN_BASE = "https://open.canada.ca/data/api/3/action"


class OpenCanadaError(RuntimeError):
    """CKAN answered with a failure (`success: false`, a 4xx, non-JSON).
    The message is CKAN's own, which usually names the bad field."""


class _TransientError(RuntimeError):
    """5xx / connection failure — worth one more attempt."""


@runtime_checkable
class OpenCanadaClient(Protocol):
    def package_search(self, query: str, rows: int) -> list[dict[str, Any]]:
        """Matching packages, CKAN's own ranking."""

    def datastore_search(self, params: dict[str, Any]) -> dict[str, Any]:
        """One `datastore_search` result object."""

    def resource_fields(self, resource_id: str) -> tuple[list[dict[str, Any]], int]:
        """(fields, total rows) for a DataStore resource."""

    def resource_show(self, resource_id: str) -> dict[str, Any]:
        """The resource record: name, package_id, format, url."""


class RealOpenCanadaClient:
    def __init__(
        self,
        *,
        base_url: str = CKAN_BASE,
        timeout_s: float = 45.0,
        cache_size: int = 128,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout_s
        self._lock = threading.Lock()
        self._fields: OrderedDict[str, tuple[list[dict[str, Any]], int]] = OrderedDict()
        self._resources: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._cap = cache_size

    def package_search(self, query: str, rows: int) -> list[dict[str, Any]]:
        result = self._call("package_search", {"q": query, "rows": rows})
        packages = result.get("results")
        if not isinstance(packages, list):
            raise OpenCanadaError("package_search: unexpected response shape")
        return packages

    def datastore_search(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._call("datastore_search", params)

    def resource_fields(self, resource_id: str) -> tuple[list[dict[str, Any]], int]:
        with self._lock:
            hit = self._fields.get(resource_id)
            if hit is not None:
                self._fields.move_to_end(resource_id)
                return hit
        result = self._call("datastore_search", {"resource_id": resource_id, "limit": 0})
        fields = [f for f in result.get("fields") or [] if f.get("id") != "_id"]
        out = (fields, int(result.get("total") or 0))
        with self._lock:
            self._fields[resource_id] = out
            while len(self._fields) > self._cap:
                self._fields.popitem(last=False)
        return out

    def resource_show(self, resource_id: str) -> dict[str, Any]:
        with self._lock:
            hit = self._resources.get(resource_id)
            if hit is not None:
                return hit
        result = self._call("resource_show", {"id": resource_id})
        with self._lock:
            self._resources[resource_id] = result
            while len(self._resources) > self._cap:
                self._resources.popitem(last=False)
        return result

    # ── transport ──

    def _call(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        query = {
            k: json.dumps(v) if isinstance(v, dict | list) and k != "fields" else v
            for k, v in params.items()
        }
        if isinstance(query.get("fields"), list):
            query["fields"] = ",".join(query["fields"])
        url = f"{self._base}/{action}?{urllib.parse.urlencode(query)}"
        for attempt in Retrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential_jitter(initial=0.5, max=4.0),
            retry=retry_if_exception_type(_TransientError),
            reraise=True,
        ):
            with attempt:
                return self._once(url)
        raise OpenCanadaError("unreachable")  # pragma: no cover

    def _once(self, url: str) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(url, timeout=self._timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code >= 500:
                raise _TransientError(f"open.canada.ca {exc.code}") from exc
            body = _error_message(exc.read())
            raise OpenCanadaError(body or f"open.canada.ca {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise _TransientError(f"open.canada.ca unreachable: {exc}") from exc
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise OpenCanadaError("open.canada.ca returned non-JSON") from exc
        if not payload.get("success"):
            raise OpenCanadaError(_error_message(raw) or "open.canada.ca request failed")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise OpenCanadaError("open.canada.ca: unexpected response shape")
        return result


def _error_message(raw: bytes) -> str:
    try:
        err = json.loads(raw).get("error") or {}
    except (ValueError, AttributeError):
        return ""
    if isinstance(err, dict):
        msg = err.get("message") or ""
        extra = {k: v for k, v in err.items() if k not in ("message", "__type")}
        return f"{msg} {extra}".strip() if extra else str(msg)
    return str(err)
