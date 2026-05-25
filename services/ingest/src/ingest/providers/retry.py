"""Tenacity retry policy for outbound HTTP per PRD 2.2 §6.4.

3 attempts, exponential backoff (1s, 2s, 4s) with ±25% jitter. Retries on
transport errors and on `RetryableHttpError` (raised by the HTTP client
for 5xx / 429). 4xx other than 429 is fatal — callers raise their own
exception class for that.
"""
from __future__ import annotations

import httpx
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)


class RetryableHttpError(Exception):
    """Marker raised by HTTP-touching clients for responses that warrant a retry.

    Carries the status code and optional `Retry-After` value so the caller
    (or a custom wait policy in future) can react.
    """

    def __init__(self, status_code: int, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


def http_retry_policy(*, max_attempts: int = 3) -> Retrying:
    return Retrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=1.0, max=8.0, jitter=0.25),
        retry=retry_if_exception_type((httpx.TransportError, RetryableHttpError)),
        reraise=True,
    )
