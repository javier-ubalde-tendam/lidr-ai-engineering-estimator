"""``POST /v1/estimate/tasks/hours``: per-task hours by vector search.

The structure-only generation yields modules -> tasks WITHOUT hours. This endpoint fills them in:
each task is matched against the historical task corpus and gets a distance-weighted consensus of
its nearest neighbours' hours plus a reliability score. A task with no neighbour under the
distance threshold comes back with ``has_match=False`` (no hours) so a UI can flag the row.

Thin transport, same posture as the sibling estimate routers: validation in ``TaskHoursRequest``
(422), auth in ``require_estimate_key`` (401), rate limit in the decorator (429), pipeline failures
-> 502. The consensus lives in ``generation/rag/task_hours.py``.
"""

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.rate_limiting import limiter
from app.api.request_context import get_request_id
from app.api.security import require_estimate_key
from app.dependencies import get_task_hours_estimator
from app.generation.rag.errors import RagError
from app.generation.rag.observability import log_stage
from app.generation.rag.schemas import TaskHoursRequest, TaskHoursResult
from app.generation.rag.task_hours import TaskHoursEstimator

log = structlog.get_logger()

router = APIRouter(prefix="/v1/estimate/tasks", tags=["estimate-tasks"])


@router.post(
    "/hours",
    response_model=TaskHoursResult,
    dependencies=[Depends(require_estimate_key)],
)
@limiter.limit("30/minute")
async def hours(
    request: Request,  # slowapi needs the Starlette request under this exact name
    payload: TaskHoursRequest,
    estimator: TaskHoursEstimator = Depends(get_task_hours_estimator),
) -> TaskHoursResult:
    """Estimate hours for every submitted task from the historical task corpus."""
    task_count = sum(len(module.tasks) for module in payload.modules)
    try:
        with log_stage("task_hours", get_request_id(request), tasks=task_count):
            return await estimator.estimate_all(payload.modules)
    except RagError as exc:
        log.error("stage_failed", stage="task_hours", error_type=type(exc).__name__)
        raise HTTPException(status_code=502, detail="Failed to estimate task hours.") from exc
