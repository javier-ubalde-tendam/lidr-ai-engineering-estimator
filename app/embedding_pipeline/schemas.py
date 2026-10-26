from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

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
    # Identidad/procedencia dada por el cliente: el servidor nunca abre esta ruta
    source_path: str = Field(
        min_length=1,
        max_length=1000,
        description="Identidad del documento, p. ej. 'data/budgets_sample.json::BUD-2024-001'",
    )
    document_type: str = Field(
        min_length=1, max_length=50, description="Tipo de documento, p. ej. 'historical_budget'"
    )
    content: Budget = Field(description="Presupuesto completo a trocear y embeber")

    @field_validator("source_path", "document_type")
    @classmethod
    def not_blank(cls, value: str) -> str:
        # strip() antes de comprobar: "   " cumple min_length=1 pero no identifica nada
        stripped = value.strip()
        if not stripped:
            raise ValueError("no puede estar vacío")
        return stripped


class IngestResponse(BaseModel):
    document_id: int = Field(description="ID BIGINT del documento en la tabla documents")
    chunks_created: int = Field(description="Chunks persistidos")
    embedding_dimension: int = Field(description="Dimensión de los vectores almacenados")
    ingestion_time_ms: int = Field(description="Duración de la ingesta (reloj monotónico)")


class DuplicateDocumentResponse(BaseModel):
    detail: str = Field(description="Siempre 'Document already ingested'")
    document_id: int = Field(description="ID del documento ya existente")


class SearchRequest(BaseModel):
    query: str = Field(max_length=8000, description="Texto de la consulta")
    k: int = Field(default=5, ge=1, le=100, description="Número de resultados")

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("no puede estar vacío")
        return stripped


class SearchResult(BaseModel):
    chunk_id: int
    document_id: int
    chunk_type: str
    content: str
    distance: float = Field(description="Distancia coseno (0 = misma dirección; menor = más cercano)")
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    k: int
    search_time_ms: int = Field(description="Embedding de la query + consulta SQL")
    results: list[SearchResult]
