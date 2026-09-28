import logging

import structlog

from app.config import get_settings


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
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
    )