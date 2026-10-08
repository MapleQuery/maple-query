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

import gzip
import json
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from semantic_enrich.providers.logging import get_logger

WDS_BASE = "https://www150.statcan.gc.ca/t1/wds/rest"
_LOG = get_logger("semantic_enrich.clients.statcan")


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
        timeout_s: float = 15.0,
        catalogue_ttl_s: float = 6 * 3600,
        metadata_cache_size: int = 256,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout_s
        self._catalogue_ttl = catalogue_ttl_s
        self._lock = threading.Lock()
        self._cubes: list[dict[str, Any]] | None = None
        self._cubes_at = 0.0
        self._refreshing = False
        self._codes: dict[str, Any] | None = None
        self._meta: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._meta_cap = metadata_cache_size

    # ── public surface ──

    def list_cubes(self) -> list[dict[str, Any]]:
        """The catalogue, never blocking on the network.

        The live catalogue is a 5 MB download that took over 20 s from
        Cloud Run on a cold start, and every concurrent first turn
        started its own. So search reads a bundled snapshot (titles and
        date ranges, ~170 KB) immediately, and one background thread
        swaps in the live list when it arrives. A stale title costs
        nothing: data and metadata calls are always live."""
        with self._lock:
            if self._cubes is None:
                self._cubes = _bundled_catalogue()
                self._cubes_at = 0.0
            stale = time.monotonic() - self._cubes_at >= self._catalogue_ttl
            if (stale or self._cubes_at == 0.0) and not self._refreshing:
                self._refreshing = True
                threading.Thread(target=self._refresh_catalogue, daemon=True).start()
            return self._cubes

    def _refresh_catalogue(self) -> None:
        try:
            cubes = self._request(
                urllib.request.Request(f"{self._base}/getAllCubesListLite"),
                timeout=max(self._timeout, 90.0),
            )
            if isinstance(cubes, list) and cubes:
                with self._lock:
                    self._cubes = cubes
                    self._cubes_at = time.monotonic()
        except StatCanError as exc:
            _LOG.warning("statcan_catalogue_refresh_failed", error=str(exc))
        finally:
            with self._lock:
                self._refreshing = False

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
        if not isinstance(body, list):
            raise StatCanError("series request: unexpected response shape")
        # WDS does not answer in request order. Pairing by position put
        # Alberta's population against Newfoundland; pair by the
        # coordinate each object names instead.
        by_coord: dict[str, dict[str, Any]] = {}
        for item in body:
            if not isinstance(item, dict) or item.get("status") != "SUCCESS":
                continue
            obj = item.get("object")
            if isinstance(obj, dict) and obj.get("coordinate"):
                by_coord[_norm_coord(str(obj["coordinate"]))] = obj
        return [
            by_coord.get(_norm_coord(c)) or {"coordinate": c, "missing": True}
            for c in coordinates
        ]

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

    def _request(
        self, req: urllib.request.Request, *, timeout: float | None = None
    ) -> Any:
        try:
            for attempt in Retrying(
                stop=stop_after_attempt(2),
                wait=wait_exponential_jitter(initial=0.5, max=2.0),
                retry=retry_if_exception_type(_TransientError),
                reraise=True,
            ):
                with attempt:
                    return self._once(req, timeout or self._timeout)
        except _TransientError as exc:
            # Out of retries: a named, catchable failure rather than an
            # internal error the tool layer never sees coming.
            raise StatCanError(str(exc)) from exc
        raise StatCanError("unreachable")  # pragma: no cover

    def _once(self, req: urllib.request.Request, timeout: float) -> Any:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
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


def _bundled_catalogue() -> list[dict[str, Any]]:
    path = Path(__file__).resolve().parent.parent / "data" / "statcan_catalogue.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        cubes: list[dict[str, Any]] = json.load(fh)
    return cubes


def _norm_coord(coord: str) -> str:
    """'2.0.0' and '2.0.0.0.0.0.0.0.0.0' name the same series."""
    parts = [p.strip() for p in coord.split(".")]
    parts += ["0"] * (10 - len(parts))
    return ".".join(str(int(p or 0)) for p in parts[:10])


def _single_object(body: Any, *, what: str) -> dict[str, Any]:
    if not isinstance(body, list) or not body or not isinstance(body[0], dict):
        raise StatCanError(f"{what}: unexpected response shape")
    item = body[0]
    if item.get("status") != "SUCCESS" or not isinstance(item.get("object"), dict):
        raise StatCanError(f"{what}: {item.get('object') or item.get('status')}")
    obj: dict[str, Any] = item["object"]
    return obj
