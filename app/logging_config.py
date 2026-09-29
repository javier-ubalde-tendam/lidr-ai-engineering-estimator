import logging

import structlog

from app.config import get_settings

# Nivel personalizado por debajo de DEBUG (10), para logs muy verbosos (ej. volcar cada chunk)
TRACE = 5
logging.addLevelName(TRACE, "TRACE")


def configure_logging() -> None:
    settings = get_settings()
    logging.basicConfig(format="%(message)s", level=settings.LOG_LEVEL)

    # JSON en producción (fácil de parsear por herramientas de logs); consola legible en desarrollo
    renderer = (
        structlog.processors.JSONRenderer()
        if settings.APP_ENV == "production"
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        # Sin esto, structlog no filtra por nivel: imprimiría TRACE/DEBUG igual con LOG_LEVEL=INFO
        wrapper_class=structlog.make_filtering_bound_logger(settings.LOG_LEVEL),
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
    )