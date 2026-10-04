import structlog

from app.sessions.models import ConversationHistory, Session

logger = structlog.get_logger(__name__)


class SessionNotFoundError(Exception):
    """No existe ninguna sesión con ese session_id (nunca existió, o el proceso se reinició)."""


class SessionStore:
    """Almacén de sesiones en memoria del proceso.

    Deliberadamente un dict[str, Session] y no Redis/una BBDD: en esta fase se acepta perder
    las sesiones al reiniciar el proceso (son conversaciones de una estimación, no datos de
    negocio) a cambio de no añadir infraestructura todavía; la persistencia real se abordará
    más adelante si hace falta. Importante: al ser un dict en memoria, SOLO funciona con un
    único worker de uvicorn (varios workers = procesos distintos = dicts distintos); con
    `uvicorn --workers N>1` cada request podría caer en un worker que nunca vio esa sesión.
    """

    def __init__(self, max_conversation_turns: int) -> None:
        self._max_conversation_turns = max_conversation_turns
        self._sessions: dict[str, Session] = {}

    def create(self) -> Session:
        session = Session(history=ConversationHistory(max_turns=self._max_conversation_turns))
        self._sessions[session.session_id] = session
        logger.info("session_created", session_id=session.session_id)
        return session

    def get_or_404(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        return session

    def __len__(self) -> int:
        return len(self._sessions)
