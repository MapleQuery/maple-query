"""`agent-service` CLI shim.

Usage:
  agent-service               # bind 0.0.0.0:$PORT (defaults to 8080)
  agent-service --port 9000   # override

The Cloud Run container uses `python -m uvicorn agent_service.app:app`
directly; this module exists for local `uv run agent-service` ergonomics.
"""
from __future__ import annotations

import argparse

import uvicorn

from agent_service.config import AgentServiceSettings


def main() -> None:
    settings = AgentServiceSettings()
    parser = argparse.ArgumentParser(prog="agent-service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Restart on file change. Local dev only.",
    )
    args = parser.parse_args()

    uvicorn.run(
        "agent_service.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
