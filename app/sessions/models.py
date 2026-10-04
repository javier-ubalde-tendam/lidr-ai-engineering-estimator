from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Un turno de la conversación ya resuelto (el system prompt no es un Message)."""

    role: Literal["user", "assistant"]
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ConversationHistory(BaseModel):
    """Ventana deslizante de mensajes user/assistant, sin el system prompt."""

    max_turns: int
    messages: list[Message] = Field(default_factory=list)

    def append(self, *, user: str, assistant: str) -> None:
        # Siempre se añaden en pareja: así nunca queda un user sin su assistant (o viceversa)
        self.messages.append(Message(role="user", content=user))
        self.messages.append(Message(role="assistant", content=assistant))
        self._trim()

    def _trim(self) -> None:
        limit = self.max_turns * 2
        if len(self.messages) <= limit:
            return
        # Se descarta siempre de dos en dos (pares completos) para no romper la alternancia user/assistant
        excess_pairs = (len(self.messages) - limit + 1) // 2
        del self.messages[: excess_pairs * 2]

    def to_messages_list(self, system_prompt: str) -> list[dict]:
        # El system prompt se regenera cada turno a partir del metadata actual: nunca se guarda aquí
        return [{"role": "system", "content": system_prompt}] + [
            {"role": m.role, "content": m.content} for m in self.messages
        ]


class ProjectMetadata(BaseModel):
    """Hechos acumulados sobre el proyecto, extraídos de la conversación turno a turno."""

    project_name: str | None = None
    assumed_team_size: int | None = Field(default=None, ge=1, le=50)
    mentioned_technologies: list[str] = Field(default_factory=list)
    agreed_scope: str | None = None

    def is_empty(self) -> bool:
        return (
            self.project_name is None
            and self.assumed_team_size is None
            and not self.mentioned_technologies
            and self.agreed_scope is None
        )

    def merge_with(self, update: "ProjectMetadata") -> "ProjectMetadata":
        # Escalares: solo se sobrescriben si el update trae un valor no nulo (no se "olvida" lo ya sabido)
        merged_technologies = list(self.mentioned_technologies)
        known_lower = {t.lower() for t in merged_technologies}
        for tech in update.mentioned_technologies:
            if tech.lower() not in known_lower:
                merged_technologies.append(tech)
                known_lower.add(tech.lower())

        return ProjectMetadata(
            project_name=update.project_name or self.project_name,
            assumed_team_size=update.assumed_team_size or self.assumed_team_size,
            mentioned_technologies=merged_technologies,
            agreed_scope=update.agreed_scope or self.agreed_scope,
        )


class Session(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    history: ConversationHistory
    metadata: ProjectMetadata = Field(default_factory=ProjectMetadata)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
