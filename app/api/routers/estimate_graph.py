"""``POST /v1/estimate/graph``: transcript -> estimate through the LangGraph workflow (Session 13).

Thin transport: validation in ``GraphEstimateRequest`` (422), auth in ``require_estimate_key``
(401), rate limit in the decorator (429). The compiled graph is built once at startup and lives in
``app.state.graph`` (see ``app/api/graph_runtime.py``); ``None`` there means 503. A graph that
raises is a 502. The business outcome (``validated`` / ``needs_review``) travels in a 200 body.
"""

from typing import Any
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.rate_limiting import limiter
from app.api.security import require_estimate_key
from app.domain.schemas.graph_estimation import GraphEstimateRequest, GraphEstimateResponse

log = structlog.get_logger()

router = APIRouter(prefix="/v1/estimate", tags=["estimate-graph"])


@router.post(
    "/graph",
    response_model=GraphEstimateResponse,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("5/minute")
async def estimate_graph(
    request: Request,  # slowapi needs the Starlette request under this exact name
    payload: GraphEstimateRequest,
) -> GraphEstimateResponse:
    """Run the estimation graph; ``estimation_id`` is the persisted thread id of the run."""
    graph: Any = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(status_code=503, detail="The estimation graph is not available.")

    estimation_id = payload.estimation_id or str(uuid4())
    config = {"configurable": {"thread_id": estimation_id}}
    try:
        result = await graph.ainvoke(
            {"transcript": payload.transcript, "estimation_id": estimation_id}, config
        )
    except Exception as exc:
        log.error("graph_failed", estimation_id=estimation_id, error_type=type(exc).__name__)
        raise HTTPException(status_code=502, detail="The estimation graph failed.") from exc

    return GraphEstimateResponse(
        estimation_id=estimation_id,
        status=result.get("status") or "needs_review",
        estimate=result.get("estimate"),
        requirements=result.get("requirements", []),
        components=result.get("components", []),
        budget_matches=result.get("budget_matches", []),
        errors=result.get("errors", []),
    )
