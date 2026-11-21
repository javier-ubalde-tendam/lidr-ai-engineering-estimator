from uuid import uuid4

from fastapi import Request


def get_request_id(request: Request) -> str:
    """The correlation id bound by the request-id middleware (a fresh one outside of it)."""
    return getattr(request.state, "request_id", None) or str(uuid4())
