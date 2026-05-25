"""structlog configuration per PRD 2.2 §12.3.

JSON to stdout by default (Cloud Logging picks this up automatically
when the Phase B Cloud Run job runs); pretty console renderer when
`INGEST_DEV=1` so local runs are readable.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog


def configure_logging(*, dev: bool | None = None, level: int = logging.INFO) -> None:
    """Call once at process start. Idempotent."""
    if dev is None:
        dev = os.environ.get("INGEST_DEV") == "1"

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: Any = (
        structlog.dev.ConsoleRenderer() if dev else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> Any:
    """Return a configured logger. Call after `configure_logging()`."""
    return structlog.get_logger(name)
