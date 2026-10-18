from typing import Literal

from pydantic import BaseModel, Field

# Literal cerrado ~ enum de Java, pero sin clase: Pydantic valida el valor y Swagger lista las
# opciones. Los valores son el universo real de data/budgets_sample.json.
Sector = Literal["finance", "ecommerce", "healthcare", "industrial"]
Complexity = Literal["low", "medium", "high"]


class ClientMetadata(BaseModel):
    name: str = Field(description="Nombre del cliente")
    sector: Sector = Field(description="Sector de negocio del cliente")
    country: str = Field(description="País del cliente (código ISO, p. ej. ES)")


class BudgetComponent(BaseModel):
    component_id: str = Field(description="Identificador del componente dentro del presupuesto")
    name: str = Field(description="Nombre corto del componente")
    description: str = Field(description="Descripción técnica del componente")
    tech_stack: list[str] = Field(description="Tecnologías usadas por el componente")
    estimated_hours: int = Field(ge=0, description="Horas estimadas del componente")
    complexity: Complexity = Field(description="Complejidad estimada del componente")
    dependencies: list[str] = Field(description="component_id de los que depende")


class Budget(BaseModel):
    budget_id: str = Field(description="Identificador único del presupuesto")
    client_metadata: ClientMetadata = Field(description="Datos del cliente")
    project_summary: str = Field(description="Resumen del proyecto")
    main_technology: str = Field(description="Tecnología principal del proyecto")
    year: int = Field(ge=2000, le=2100, description="Año del presupuesto")
    total_estimated_hours: int = Field(ge=0, description="Horas totales estimadas")
    components: list[BudgetComponent] = Field(min_length=1, description="Componentes del proyecto")


class Chunk(BaseModel):
    chunk_id: str = Field(description="Identificador del chunk: '{budget_id}::{component_id}'")
    text: str = Field(description="Texto que se embebe (header del presupuesto + componente)")
    # Tipado explícito: con mypy strict, dict sin parámetros de tipo no compila
    metadata: dict[str, str | int] = Field(
        description="Metadatos filtrables. No se embeben: se guardan junto al vector"
    )
    token_count: int = Field(description="Tokens del texto según el tokenizer del modelo")


class EmbeddedChunk(Chunk):
    embedding: list[float] = Field(description="Vector de embedding del texto")


class IngestRequest(BaseModel):
    budgets: list[Budget] = Field(min_length=1, description="Presupuestos a trocear y embeber")


class IngestStats(BaseModel):
    total_budgets: int = Field(description="Presupuestos recibidos")
    total_chunks: int = Field(description="Chunks generados")
    total_tokens: int = Field(description="Suma de tokens de todos los chunks")
    estimated_cost_usd: float = Field(description="Coste estimado de los embeddings en USD")


class IngestResponse(BaseModel):
    chunks: list[EmbeddedChunk] = Field(description="Chunks con su embedding")
    stats: IngestStats = Field(description="Estadísticas de la ingesta")
