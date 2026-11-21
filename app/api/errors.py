import structlog
from fastapi import HTTPException

logger = structlog.get_logger(__name__)


def service_error(event: str, exc: Exception, detail: str) -> HTTPException:
    # Al log solo tipo y mensaje truncado (nunca cuerpos ni claves); al cliente un mensaje genérico
    logger.error(event, error_type=type(exc).__name__, error=str(exc)[:200])
    return HTTPException(status_code=500, detail=detail)
