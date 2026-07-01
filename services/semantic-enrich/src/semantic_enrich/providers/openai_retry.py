"""Tenacity retry policy for OpenAI calls.

Same shape as `bq_retry_policy` — bounded exponential backoff with
jitter, per-attempt logging. The predicate matches OpenAI's
`RateLimitError` and `APIStatusError` where the HTTP status is 5xx.

4xx errors (auth failures, malformed requests) propagate immediately:
retrying an unauthorised call just burns time.
"""
from __future__ import annotations

import logging

from openai import APIStatusError, RateLimitError
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        return isinstance(status, int) and status >= 500
    return False


def openai_retry_policy(*, max_attempts: int = 3) -> Retrying:
    return Retrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=2.0, max=16.0, jitter=0.25),
        retry=retry_if_exception(_is_retryable),
        before_sleep=before_sleep_log(
            logging.getLogger("semantic_enrich.openai_retry"), logging.WARNING
        ),
        reraise=True,
    )
