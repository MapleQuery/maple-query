"""Bearer-token auth dependency.

Shared token; single value. Rejects with 401 on missing / wrong token.
Health routes bypass by not depending on `require_bearer`.
"""
from __future__ import annotations

import hmac

from fastapi import Depends, Header, HTTPException, Request, status


def require_bearer(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    """Reject the request unless the `Authorization` header matches the
    configured token via a constant-time compare.

    Constant-time to keep timing attacks off the table — the token is
    public by construction (bundled into the FE), but the check itself
    should still not leak information about near-matches."""
    expected = request.app.state.api_token
    if expected is None:
        # Refuse to serve traffic when the service isn't configured. A
        # 500 here is the right signal — the operator is missing a
        # required env var.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="api_token_not_configured",
        )
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing_bearer",
            headers={"WWW-Authenticate": "Bearer"},
        )
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="malformed_bearer",
            headers={"WWW-Authenticate": "Bearer"},
        )
    presented = authorization[len(prefix) :].strip()
    if not hmac.compare_digest(presented, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid_bearer",
            headers={"WWW-Authenticate": "Bearer"},
        )


BearerAuth = Depends(require_bearer)
