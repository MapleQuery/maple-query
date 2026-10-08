"""Statistics Canada Web Data Service (WDS) client.

The warehouse mirrors open.canada.ca CSVs; it does not hold StatCan's
~8,000 statistical tables (CPI, GDP, population, trade, government
finance, housing starts …), and mirroring them would duplicate a
public API that already serves every table, current to the day. So the
agent reads them live from WDS at question time instead.

Endpoints used (all public, unauthenticated, JSON):

- `getAllCubesListLite`            the table catalogue (~5 MB)
- `getCubeMetadata`                one table's dimensions and members
- `getDataFromCubePidCoordAndLatestNPeriods`  series values
- `getCodeSets`                    unit / scalar / symbol code tables

Catalogue, code sets and metadata are cached in-process: they change at
most daily, and the service scales to zero, so a cold start pays one
catalogue fetch. Transport errors and 5xx retry; anything else raises
`StatCanError` so the tool turns it into a named error the model sees.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from typing import Any, Protocol, runtime_checkable

from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

WDS_BASE = "https://www150.statcan.gc.ca/t1/wds/rest"


class StatCanError(RuntimeError):
    """WDS answered with something other than data: a 4xx, a non-JSON
    body, or a per-item status other than SUCCESS."""


class _TransientError(RuntimeError):
    """5xx / connection failure — worth one more attempt."""


@runtime_checkable
class StatCanClient(Protocol):
    def list_cubes(self) -> list[dict[str, Any]]:
        """Every table in the WDS catalogue (archived ones included)."""

    def cube_metadata(self, product_id: int) -> dict[str, Any]:
        """`getCubeMetadata` object for one table."""

    def series_latest_n(
        self, product_id: int, coordinates: list[str], latest_n: int
    ) -> list[dict[str, Any]]:
        """One `getDataFromCubePidCoordAndLatestNPeriods` object per
        coordinate, in request order."""

    def code_sets(self) -> dict[str, Any]:
        """`getCodeSets` object."""


class RealStatCanClient:
    """urllib-backed WDS client with process-wide caches."""

    def __init__(
        self,
        *,
        base_url: str = WDS_BASE,
        timeout_s: float = 20.0,
        catalogue_ttl_s: float = 6 * 3600,
        metadata_cache_size: int = 256,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout_s
        self._catalogue_ttl = catalogue_ttl_s
        self._lock = threading.Lock()
        self._cubes: list[dict[str, Any]] | None = None
        self._cubes_at = 0.0
        self._codes: dict[str, Any] | None = None
        self._meta: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._meta_cap = metadata_cache_size

    # ── public surface ──

    def list_cubes(self) -> list[dict[str, Any]]:
        with self._lock:
            fresh = (
                self._cubes is not None
                and time.monotonic() - self._cubes_at < self._catalogue_ttl
            )
            if fresh:
                assert self._cubes is not None
                return self._cubes
        cubes = self._get("getAllCubesListLite")
        if not isinstance(cubes, list):
            raise StatCanError("getAllCubesListLite: expected a list")
        with self._lock:
            self._cubes = cubes
            self._cubes_at = time.monotonic()
        return cubes

    def cube_metadata(self, product_id: int) -> dict[str, Any]:
        with self._lock:
            hit = self._meta.get(product_id)
            if hit is not None:
                self._meta.move_to_end(product_id)
                return hit
        body = self._post("getCubeMetadata", [{"productId": product_id}])
        obj = _single_object(body, what=f"getCubeMetadata({product_id})")
        with self._lock:
            self._meta[product_id] = obj
            while len(self._meta) > self._meta_cap:
                self._meta.popitem(last=False)
        return obj

    def series_latest_n(
        self, product_id: int, coordinates: list[str], latest_n: int
    ) -> list[dict[str, Any]]:
        payload = [
            {"productId": product_id, "coordinate": c, "latestN": latest_n}
            for c in coordinates
        ]
        body = self._post("getDataFromCubePidCoordAndLatestNPeriods", payload)
        if not isinstance(body, list) or len(body) != len(coordinates):
            raise StatCanError("series request: unexpected response shape")
        out: list[dict[str, Any]] = []
        for coord, item in zip(coordinates, body, strict=True):
            if not isinstance(item, dict) or item.get("status") != "SUCCESS":
                # A coordinate WDS doesn't hold is a per-series miss,
                # not a failed request: report it in-band.
                out.append({"coordinate": coord, "missing": True})
                continue
            obj = item.get("object")
            out.append(obj if isinstance(obj, dict) else {"coordinate": coord, "missing": True})
        return out

    def code_sets(self) -> dict[str, Any]:
        with self._lock:
            if self._codes is not None:
                return self._codes
        body = self._get("getCodeSets")
        if not isinstance(body, dict) or not isinstance(body.get("object"), dict):
            raise StatCanError("getCodeSets: unexpected response shape")
        codes: dict[str, Any] = body["object"]
        with self._lock:
            self._codes = codes
        return codes

    # ── transport ──

    def _get(self, method: str) -> Any:
        return self._request(urllib.request.Request(f"{self._base}/{method}"))

    def _post(self, method: str, payload: Any) -> Any:
        req = urllib.request.Request(
            f"{self._base}/{method}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._request(req)

    def _request(self, req: urllib.request.Request) -> Any:
        for attempt in Retrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential_jitter(initial=0.5, max=4.0),
            retry=retry_if_exception_type(_TransientError),
            reraise=True,
        ):
            with attempt:
                return self._once(req)
        raise StatCanError("unreachable")  # pragma: no cover

    def _once(self, req: urllib.request.Request) -> Any:
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code >= 500:
                raise _TransientError(f"WDS {exc.code}") from exc
            raise StatCanError(f"WDS {exc.code} for {req.full_url}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise _TransientError(f"WDS unreachable: {exc}") from exc
        try:
            return json.loads(raw)
        except ValueError as exc:
            raise StatCanError(f"WDS returned non-JSON for {req.full_url}") from exc


def _single_object(body: Any, *, what: str) -> dict[str, Any]:
    if not isinstance(body, list) or not body or not isinstance(body[0], dict):
        raise StatCanError(f"{what}: unexpected response shape")
    item = body[0]
    if item.get("status") != "SUCCESS" or not isinstance(item.get("object"), dict):
        raise StatCanError(f"{what}: {item.get('object') or item.get('status')}")
    obj: dict[str, Any] = item["object"]
    return obj
