from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Un turno de la conversación ya resuelto (el system prompt no es un Message)."""

    role: Literal["user", "assistant"]
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


SUMMARY_PREFIX = "[Earlier conversation summary — the recent turns below are the live thread]"


class ConversationHistory(BaseModel):
    """Memoria híbrida: resumen acumulativo + anclas literales + ventana reciente (sin system prompt)."""

    max_turns: int
    messages: list[Message] = Field(default_factory=list)
    # Pares user/assistant con un compromiso duradero (NDA, alcance congelado...): nunca se olvidan
    anchors: list[Message] = Field(default_factory=list)
    # Resumen de los turnos expulsados de la ventana que no eran anclas
    summary: str | None = None

    def append(self, *, user: str, assistant: str) -> None:
        # Siempre se añaden en pareja: así nunca queda un user sin su assistant (o viceversa).
        # No recorta: qué se olvida lo decide en exclusiva la política de compresión
        self.messages.append(Message(role="user", content=user))
        self.messages.append(Message(role="assistant", content=assistant))

    def to_messages_list(self, system_prompt: str) -> list[dict]:
        """Orden: system, resumen, anclas, ventana reciente.

        De lo más antiguo y comprimido a lo más reciente y literal: el modelo da más peso a lo
        último que lee, así que el hilo vivo va al final y el resumen (lo menos fiable) al principio.
        El resumen viaja como mensaje user y no en el system prompt porque éste se regenera cada
        turno desde el metadata y es lo que fija las reglas de la estimación.
        """
        messages = [{"role": "system", "content": system_prompt}]
        if self.summary:
            messages.append({"role": "user", "content": f"{SUMMARY_PREFIX}\n{self.summary}"})
        messages += [{"role": m.role, "content": m.content} for m in self.anchors]
        messages += [{"role": m.role, "content": m.content} for m in self.messages]
        return messages


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
    # Último tier resuelto y la regla que lo decidió (para depuración y para el panel de Streamlit)
    last_resolved_tier: str | None = None
    last_tier_rule: str | None = None
