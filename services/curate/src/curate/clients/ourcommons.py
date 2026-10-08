"""ourcommons.ca proactive disclosure: members' quarterly expenditures.

The members page links every published quarter
(`/ProactiveDisclosure/en/members/<year>/<quarter>?summaryId=…`); each
quarter's page carries its own CSV link and the period it covers. Plain
HTTP, no challenge, ~30 KB a quarter, paced at one request a second.
"""

from __future__ import annotations

import html
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol, runtime_checkable

_QUARTER_LINK = re.compile(r'href="(/ProactiveDisclosure/en/members/(\d{4})/([1-4])\?summaryId=[0-9a-f-]+)"')
_CSV_LINK = re.compile(r'href="(/ProactiveDisclosure/en/members/[0-9a-f-]+/csv)"')
_PERIOD = re.compile(r"([A-Z][a-z]+ \d{1,2}, \d{4}) to ([A-Z][a-z]+ \d{1,2}, \d{4})")


class OurCommonsError(RuntimeError):
    """The site answered with something the parser cannot read. Fails
    the run: a redesign would otherwise yield zero quarters, silently."""


@dataclass(frozen=True)
class QuarterRef:
    path: str
    report_year: int  # the year in the URL: fiscal year ending in it
    quarter: int


@dataclass(frozen=True)
class QuarterReport:
    ref: QuarterRef
    period_start: date
    period_end: date
    csv_url: str
    csv_text: str


@runtime_checkable
class OurCommonsClient(Protocol):
    def quarters(self) -> list[QuarterRef]:
        """Every quarter the members page links."""

    def report(self, ref: QuarterRef) -> QuarterReport:
        """A quarter's period and CSV."""


def parse_quarters(page: str) -> list[QuarterRef]:
    refs = {
        (int(y), int(q)): QuarterRef(html.unescape(path), int(y), int(q))
        for path, y, q in _QUARTER_LINK.findall(page)
    }
    return [refs[k] for k in sorted(refs)]


def parse_quarter_page(page: str) -> tuple[date, date, str]:
    csv_link = _CSV_LINK.search(page)
    period = _PERIOD.search(html.unescape(page))
    if not csv_link or not period:
        raise OurCommonsError("quarter page has no CSV link or period")
    start = datetime.strptime(period.group(1), "%B %d, %Y").date()
    end = datetime.strptime(period.group(2), "%B %d, %Y").date()
    return start, end, csv_link.group(1)


class RealOurCommonsClient:
    def __init__(self, *, base_url: str, user_agent: str, timeout_s: float, rps: float) -> None:
        self._base = base_url.rstrip("/")
        self._ua = user_agent
        self._timeout = timeout_s
        self._gap = 1.0 / rps if rps > 0 else 0.0
        self._last = 0.0

    def quarters(self) -> list[QuarterRef]:
        # The listing has come back without its links once in testing (a
        # transient partial page); retry a few times, spaced, before
        # failing the run.
        for attempt in range(3):
            refs = parse_quarters(self._get("/ProactiveDisclosure/en/members"))
            if refs:
                return refs
            time.sleep(5 * (attempt + 1))
        raise OurCommonsError("members page links no quarters (3 attempts)")

    def report(self, ref: QuarterRef) -> QuarterReport:
        start, end, csv_path = parse_quarter_page(self._get(ref.path))
        return QuarterReport(ref, start, end, f"{self._base}{csv_path}", self._get(csv_path))

    def _get(self, path: str) -> str:
        wait = self._last + self._gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        req = urllib.request.Request(f"{self._base}{path}", headers={"User-Agent": self._ua})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body: bytes = resp.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            raise OurCommonsError(f"ourcommons.ca failed for {path}: {exc}") from exc
        return body.decode("utf-8-sig")
