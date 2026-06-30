"""Tenacity retry policy for BigQuery jobs.

3 attempts, exponential backoff (2s, 4s, 8s) with ±25% jitter. Retries
on transient BQ errors (`InternalServerError`, `ServiceUnavailable`,
`TooManyRequests`, `GatewayTimeout`). Validation / NotFound / permission
errors are fatal.
"""
from __future__ import annotations

import logging

from google.api_core import exceptions as gax
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

_RETRYABLE: tuple[type[BaseException], ...] = (
    gax.InternalServerError,
    gax.ServiceUnavailable,
    gax.TooManyRequests,
    gax.GatewayTimeout,
)


def bq_retry_policy(*, max_attempts: int = 3) -> Retrying:
    return Retrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=2.0, max=16.0, jitter=0.25),
        retry=retry_if_exception_type(_RETRYABLE),
        before_sleep=before_sleep_log(
            logging.getLogger("semantic_enrich.retry"), logging.WARNING
        ),
        reraise=True,
    )
