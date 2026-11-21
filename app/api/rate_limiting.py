"""Per-API-key rate limiting for the secured routers (slowapi).

The consumer identity is the ``X-API-Key`` header, falling back to the client IP when absent:
several clients behind one NAT must not share a bucket, and one key must not dodge its limit
by rotating IPs. Limits are declared per route with ``@limiter.limit(...)`` and a custom
handler returns a JSON 429 with a ``Retry-After`` hint.
"""

import structlog
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request
from starlette.responses import JSONResponse

log = structlog.get_logger()

# Every limit in this app is per minute, so the client can retry after a full window
_RETRY_AFTER_SECONDS = 60


def api_key_identifier(request: Request) -> str:
    """Rate-limit bucket key: the API key, falling back to the client IP."""
    return request.headers.get("X-API-Key") or get_remote_address(request)


limiter = Limiter(key_func=api_key_identifier)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Return a 429 with a ``Retry-After`` header and an informative body."""
    log.warning("rate_limit_exceeded", path=request.url.path, limit=str(exc.limit.limit))
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Rate limit exceeded.",
            "limit": str(exc.limit.limit),
            "retry_after_seconds": _RETRY_AFTER_SECONDS,
        },
        headers={"Retry-After": str(_RETRY_AFTER_SECONDS)},
    )
