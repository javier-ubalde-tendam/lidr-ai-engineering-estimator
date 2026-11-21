import uuid
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Literal cerrado ~ enum de Java, pero sin clase: Pydantic valida el valor y Swagger lista las
# opciones. Los cuatro primeros son el universo de data/budgets_sample.json; los otros cuatro los
# usa el corpus sintético de tareas (data/task_corpus.json, scripts/build_task_corpus.py)
Sector = Literal[
    "finance",
    "ecommerce",
    "healthcare",
    "industrial",
    "logistics",
    "education",
    "media",
    "government",
]
Complexity = Literal["low", "medium", "high"]


class ClientMetadata(BaseModel):
    name: str = Field(description="Nombre del cliente")
    sector: Sector = Field(description="Sector de negocio del cliente")
    country: str = Field(description="País del cliente (código ISO, p. ej. ES)")


class BudgetComponent(BaseModel):
    component_id: str = Field(description="Identificador del componente dentro del presupuesto")
    name: str = Field(description="Nombre corto del componente")
    description: str = Field(description="Descripción técnica del componente")
    # Functional block of a task-granular component (per-task hours corpus); None for budgets
    module: str | None = Field(
        default=None, description="Bloque funcional al que pertenece la tarea (p. ej. 'Payments')"
    )
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
    # Tipado explícito: con mypy strict, dict sin parámetros de tipo no compila. Las listas las usan
    # las colecciones multi-índice (p. ej. `speakers` de las transcripciones)
    metadata: dict[str, str | int | list[str]] = Field(
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
    chunk_type: str = Field(
        default="budget_component",
        min_length=1,
        max_length=50,
        description="chunk_type de cada chunk (filtrable); 'historical_task' para el corpus de "
        "tareas",
    )

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


SearchMode = Literal["vector", "hybrid"]


class RetrievedChunk(BaseModel):
    id: int = Field(description="BIGINT id of the chunk row")
    content: str
    # sector / project_year / budget_id are budget-centric: they default for the other collections
    sector: str = Field(default="unknown", description="Client sector of the parent budget")
    project_year: int = Field(default=0, description="Year of the parent budget")
    chunk_type: str
    distance: float = Field(
        description="Cosine distance to the query. 1.0 is a sentinel for chunks found only by "
        "the lexical branch (no vector distance); fusion and reranking never use it"
    )
    budget_id: str | None = Field(
        default=None, description="Parent budget (metadata->>'budget_id')"
    )
    # Session 10 multi-index fields: all default-valued, so every single-collection caller is unchanged
    collection: str = Field(
        default="budget",
        description="Provenance label: collection the chunk came from (budget/transcript/technical_doc)",
    )
    source_id: str | None = Field(
        default=None,
        description="Traceable id of the parent document in any collection (budget_id, transcript_id, doc_id)",
    )
    relevance_score: float | None = Field(
        default=None,
        description="Final score of the advanced pipeline (higher = better); None on the plain paths",
    )
    document_date: date | None = Field(
        default=None, description="Date used by temporal decay; None when the collection has none"
    )
    # Session 11: the hallucination gate anchors a line's hours on what the cited chunks record
    estimated_hours: int | None = Field(
        default=None,
        description="Historical hours recorded for this chunk (JSONB metadata); None when the "
        "collection carries none",
    )


class RetrievalResult(BaseModel):
    chunks: list[RetrievedChunk] = Field(description="Best first")
    low_confidence: bool = Field(description="True when nothing was retrieved (soft-fail)")
    candidates_evaluated: int = Field(description="Chunks with an embedding considered by k-NN")
    search_mode: SearchMode = "vector"
    rerank: bool = False
    search_time_ms: int = Field(default=0, description="Retrieval + fusion + rerank, no embedding")
    vector_hits: int = Field(default=0, description="Rows returned by the vector branch")
    lexical_hits: int = Field(default=0, description="Rows returned by the lexical branch")


class RetrievalRequest(BaseModel):
    # extra="forbid": unknown fields are a client bug, not something to ignore silently
    model_config = ConfigDict(extra="forbid")

    query_text: str = Field(min_length=10, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=30)
    distance_threshold: float = Field(default=0.6, ge=0, le=2, description="Cosine distance")
    search_mode: SearchMode | None = Field(
        default=None,
        description="None = runtime override (PUT /api/v1/config/retrieval) or RETRIEVAL_SEARCH_MODE",
    )
    rerank: bool | None = Field(
        default=None,
        description="None = runtime override (PUT /api/v1/config/retrieval) or RERANKER_ENABLED",
    )
    # Optional metadata filters: applied before the ranking, None = do not filter on that axis
    sectors: list[Sector] | None = Field(default=None, description="Client sectors to keep")
    project_year_min: int | None = Field(default=None, ge=2010, le=2100)
    project_year_max: int | None = Field(default=None, ge=2010, le=2100)
    chunk_types: list[str] | None = Field(default=None, description="chunk_type values to keep")

    @model_validator(mode="after")
    def year_range_is_ordered(self) -> "RetrievalRequest":
        if (
            self.project_year_min is not None
            and self.project_year_max is not None
            and self.project_year_min > self.project_year_max
        ):
            raise ValueError("project_year_min must not be greater than project_year_max")
        return self


# ---------------------------------------------------------------------------
# Session 9 - RAG estimation from a transcript (query understanding -> generation).
# They coexist with the euros/weeks EstimationResult of app.domain.schemas: this Estimate is
# hours-based (engineer-days) and every line carries verifiable citations.
# ---------------------------------------------------------------------------

Scale = Literal["small", "medium", "large", "unknown"]
Confidence = Literal["high", "medium", "low", "insufficient"]
Relevance = Literal["primary", "supporting", "tangential"]
Impact = Literal["high", "medium", "low"]


class EstimationQuery(BaseModel):
    """Structured brief distilled from a raw meeting transcript.

    A transcript is full of digressions: only what drives retrieval (what to build, with which
    technologies, under which constraints) is extracted, instead of embedding the raw text.
    """

    function: str = Field(description="Functional summary of the project.")
    technologies: list[str] = Field(default_factory=list)
    sector: str | None = None
    scale: Scale = "unknown"
    country: str | None = None
    regulations: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class SourceCitation(BaseModel):
    """A reference from the estimate back to a retrieved chunk (estimate-global layer)."""

    source_id: int = Field(description="DB id of the cited chunk (a RetrievedChunk.id).")
    relevance: Relevance
    used_for: str = Field(description="What this source contributed to the estimate.")


class Assumption(BaseModel):
    """An estimate component NOT backed by any retrieved source."""

    description: str
    impact: Impact
    rationale: str


class SourceReference(BaseModel):
    """A verifiable, line-level citation back to a single retrieved chunk."""

    chunk_id: str = Field(
        description="Id of the retrieved chunk supporting this line: the `id` attribute of the "
        "<source> element it was taken from."
    )
    document_id: str = Field(
        description="Historical budget the chunk belongs to: the `document_id` attribute of the "
        "<source> element."
    )
    evidence: str = Field(
        description="Verbatim span or figure copied from the source that backs this line. "
        "Quote the source; do not paraphrase."
    )


class TaskItem(BaseModel):
    """One concrete engineering task inside a functional module, in engineer-days.

    Each task is an estimate line. A grounded line derives from historical evidence and carries
    at least one SourceReference; an ungrounded one has no sufficient source data, so it must
    not invent hours and should surface its scope as an Assumption instead.
    """

    name: str
    description: str | None = Field(default=None, description="One-line scope of the task.")
    engineer_days: int | None = Field(
        default=None,
        ge=0,
        description="Effort in engineer-days. Must be None when `grounded` is False (no source "
        "data => no invented hours).",
    )
    grounded: bool = Field(
        default=False,
        description="True iff this line is backed by retrieved source data.",
    )
    sources: list[SourceReference] = Field(
        default_factory=list,
        description="Per-line citations. Non-empty iff `grounded` is True.",
    )

    @model_validator(mode="after")
    def grounding_integrity(self) -> "TaskItem":
        # Raising makes Instructor re-prompt the model instead of letting an unverifiable
        # line through
        if self.grounded and not self.sources:
            raise ValueError("a grounded line must cite at least one source")
        if not self.grounded and self.sources:
            raise ValueError("an ungrounded line (grounded=False) must not carry sources")
        if not self.grounded and self.engineer_days is not None:
            raise ValueError(
                "an ungrounded line (grounded=False) must leave engineer_days null; "
                "do not invent hours without source data"
            )
        return self


class WorkModule(BaseModel):
    """A functional block (Auth, Payments, Data, Frontend, Infra, QA, PM...) of tasks."""

    name: str
    description: str | None = Field(default=None, description="What this functional block covers.")
    tasks: list[TaskItem] = Field(default_factory=list)


class Estimate(BaseModel):
    """Grounded estimate produced from retrieved historical budgets (engineer-days).

    When the retrieved context is insufficient, ``confidence="insufficient"``, the numeric totals
    are None and ``modules`` is empty (enforced by ``validation.check_coherence``).
    """

    total_engineer_days: int | None = None
    modules: list[WorkModule] = Field(default_factory=list)
    duration_weeks: int | None = None
    sources: list[SourceCitation] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)
    confidence: Confidence
    reasoning: str = Field(description="How the estimate was derived from the sources.")
    insufficient_context_explanation: str | None = None


LineCitationStatus = Literal["grounded", "dangling", "insufficient"]


class LineCitation(BaseModel):
    """The citation status of a single estimate line."""

    module: str = Field(description="Module the line belongs to.")
    component: str = Field(description="The line (task) name.")
    status: LineCitationStatus = Field(
        description="grounded = every cited chunk was retrieved; dangling = at least one cited "
        "chunk_id was never in the context; insufficient = the line has no source data."
    )
    cited_chunk_ids: list[str] = Field(default_factory=list)
    dangling_chunk_ids: list[str] = Field(
        default_factory=list, description="Cited ids that were never retrieved."
    )


class CitationReport(BaseModel):
    """Result of verifying an estimate's citations against the context it was given."""

    total_lines: int = Field(ge=0)
    grounded_lines: int = Field(ge=0, description="Lines whose every citation was retrieved.")
    dangling_lines: int = Field(ge=0, description="Lines with at least one fabricated citation.")
    insufficient_lines: int = Field(ge=0, description="Lines flagged as no sufficient data.")
    verified_citations: int = Field(
        ge=0, description="Individual citations that resolved to a retrieved chunk."
    )
    dangling_citations: list[str] = Field(
        default_factory=list,
        description="Sorted, de-duplicated cited ids that were never retrieved "
        "(line-level and estimate-global). Empty means every citation is real.",
    )
    lines: list[LineCitation] = Field(default_factory=list)

    @property
    def has_dangling(self) -> bool:
        return bool(self.dangling_citations)


class LineVerdict(BaseModel):
    """Strict-judge verdict for a single estimate line."""

    module: str = Field(description="Module the line belongs to (echoed for matching).")
    component: str = Field(description="The line (task) name (echoed for matching).")
    entailed: bool = Field(
        description="True iff the cited evidence supports the line's hours and scope."
    )
    reason: str = Field(description="One-sentence justification for the verdict.")


LineGateStatus = Literal["grounded", "degraded", "insufficient"]


class LineGate(BaseModel):
    """Graded semantic-gate result for one estimate line."""

    module: str
    component: str
    status: LineGateStatus = Field(
        description="grounded = anchor and judge both accept the line; degraded = a grounded line "
        "whose hours are not entailed by the cited evidence (numeric anchor miss or judge "
        "rejection); insufficient = the line carried no data."
    )
    numeric_deviation: float | None = Field(
        default=None,
        ge=0.0,
        description="|line - anchor| / anchor; None when no numeric anchor was available.",
    )
    reason: str = Field(default="", description="Why the line was degraded (empty when grounded).")


class HallucinationReport(BaseModel):
    """Aggregate of the semantic gate over an estimate's lines."""

    total_lines: int = Field(ge=0)
    grounded_lines: int = Field(ge=0, description="Lines the anchor and the judge both accept.")
    degraded_lines: int = Field(
        ge=0, description="Lines citing a real source whose number it does not entail."
    )
    insufficient_lines: int = Field(ge=0, description="Lines flagged as no sufficient data.")
    lines: list[LineGate] = Field(default_factory=list)

    @property
    def has_degraded(self) -> bool:
        return self.degraded_lines > 0


class TaskNeighbor(BaseModel):
    """One historical task that matched a task being estimated (input of the synthesis)."""

    source_id: int = Field(description="DB id of the matched historical chunk.")
    budget_id: str | None = Field(default=None, description="Traceable parent corpus id.")
    estimated_hours: int = Field(ge=0, description="Hours recorded for this historical task.")
    distance: float = Field(description="Cosine distance to the task (lower = closer).")


class HourRange(BaseModel):
    """Hour range emitted when the historical sources contradict each other.

    A weighted consensus hides disagreement (40h vs 90h -> a misleading 65h point); the range
    keeps the spread visible together with the reason.
    """

    low: int = Field(ge=0, description="Low end of the plausible hours.")
    high: int = Field(ge=0, description="High end of the plausible hours.")
    reason: str = Field(description="Why the sources disagree (the conflict, named).")


class EstimateRequest(BaseModel):
    """Payload of ``POST /v1/estimate/from-transcript``."""

    transcript: str = Field(min_length=100, max_length=50_000)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128)


# ---------------------------------------------------------------------------
# Per-stage endpoints (/v1/estimate/stages/*): each pipeline stage as a stateless request/response
# pair. Retrieval reuses RetrievalRequest / RetrievalResult.
# ---------------------------------------------------------------------------


class ReformulateRequest(BaseModel):
    """Payload of ``POST /v1/estimate/stages/reformulate``."""

    transcript: str = Field(min_length=100, max_length=50_000)


class ReformulationResult(BaseModel):
    """Output of the query-understanding stage: the brief plus the text that gets embedded."""

    query: EstimationQuery
    search_text: str = Field(description="Corpus-aligned text fed to the embedder.")


class AssembleRequest(BaseModel):
    """Payload of ``POST /v1/estimate/stages/assemble``.

    ``max_context_tokens`` defaults to ``MAX_CONTEXT_TOKENS`` when omitted; a small value lets a
    demo show the whole-chunk truncation.
    """

    chunks: list[RetrievedChunk]
    max_context_tokens: int | None = Field(default=None, ge=256, le=64_000)
    augment: bool = Field(default=False, description="Apply the augmentation passes first.")
    compress: bool = Field(default=True, description="Compress each source to its key points.")
    reorder: bool = Field(default=True, description="Edge-load reorder (lost-in-the-middle).")


class AssembleResult(BaseModel):
    """Output of the assembly stage: the ``<source>`` block plus what survived the budget."""

    context_block: str
    kept_chunks: list[RetrievedChunk]
    dropped_count: int = Field(ge=0, description="Chunks dropped by the token budget.")
    token_count: int = Field(ge=0, description="Tokens in the assembled context block.")
    augmented: bool = Field(default=False, description="Whether the augmentation passes ran.")


class StructureRequest(BaseModel):
    """Payload of ``POST /v1/estimate/stages/structure`` (free module -> task decomposition)."""

    query: EstimationQuery


class GenerateRequest(BaseModel):
    """Payload of ``POST /v1/estimate/stages/generate``.

    ``kept_chunks`` are the chunks the context block was built from: they validate the citations.
    """

    context_block: str = Field(min_length=1)
    query: EstimationQuery
    kept_chunks: list[RetrievedChunk] = Field(default_factory=list)
    include_hours: bool = Field(
        default=True,
        description="False = module -> task structure WITHOUT hours (they are derived later by "
        "per-task vector search).",
    )


class GenerateResult(BaseModel):
    """Output of the generation stage: the estimate plus the grounding signals."""

    estimate: Estimate
    fabricated_source_ids: list[str] = Field(
        default_factory=list,
        description="Cited chunk_ids not present in kept_chunks (empty = clean). Mirrors "
        "``citation_report.dangling_citations``.",
    )
    citation_report: CitationReport | None = Field(
        default=None,
        description="Per-line citation report (None for the structure stage: no sources).",
    )
    coherent: bool = Field(description="False when an insufficient estimate still carries numbers.")


class VerifyRequest(BaseModel):
    """Payload of ``POST /v1/estimate/stages/verify`` (semantic hallucination gate)."""

    estimate: Estimate
    kept_chunks: list[RetrievedChunk] = Field(default_factory=list)
    use_judge: bool = Field(
        default=True,
        description="False = only the deterministic numeric anchor runs (no LLM call).",
    )


# ---------------------------------------------------------------------------
# Per-task hours (/v1/estimate/tasks/hours): each task of a hour-less structure is matched against
# the historical task corpus (chunk_type "historical_task"); the hours are a weighted consensus of
# the nearest neighbours. No neighbour under the distance threshold = no hours (red flag).
# ---------------------------------------------------------------------------


class TaskHoursEstimate(BaseModel):
    """Hours derived for one task from the historical task corpus."""

    module: str = Field(description="Module the task belongs to (echoed back).")
    task: str = Field(description="Task name (echoed back).")
    estimated_hours: int | None = Field(
        default=None, ge=0, description="Weighted-consensus hours, or null when no match."
    )
    reliability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="0..1 confidence in the hours (closeness x neighbour agreement).",
    )
    has_match: bool = Field(description="Whether any neighbour crossed the distance threshold.")
    dispersion: float | None = Field(
        default=None,
        ge=0.0,
        description="Coefficient of variation of the neighbour hours; higher = less agreement.",
    )
    neighbors: list[TaskNeighbor] = Field(
        default_factory=list, description="The historical tasks the hours were derived from."
    )
    hours_range: HourRange | None = Field(
        default=None,
        description="Present only when the historical sources contradict; the point "
        "``estimated_hours`` still carries the consensus.",
    )


class TaskHoursTaskInput(BaseModel):
    """One task to estimate (name + optional description)."""

    name: str = Field(min_length=1)
    description: str | None = None


class TaskHoursModuleInput(BaseModel):
    """One module with its tasks, as reviewed by the human before the hours."""

    name: str = Field(min_length=1)
    tasks: list[TaskHoursTaskInput] = Field(default_factory=list)


class TaskHoursRequest(BaseModel):
    """Payload of ``POST /v1/estimate/tasks/hours``."""

    modules: list[TaskHoursModuleInput] = Field(min_length=1)


class TaskHoursResult(BaseModel):
    """Per-task hours estimates, in the order the tasks were submitted."""

    tasks: list[TaskHoursEstimate] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Incremental corpus expansion (/embeddings/index/*): add NEW budgets to the corpus (same
# embedding model, same table) as a background job, and report per-collection corpus stats.
# ---------------------------------------------------------------------------

IndexJobStatus = Literal["pending", "running", "completed", "failed"]


class IndexRunRequest(BaseModel):
    """Payload of ``POST /embeddings/index/runs``: a batch of new documents."""

    documents: list[Budget] = Field(
        min_length=1, max_length=200, description="New budget documents to add to the corpus."
    )
    document_type: str = Field(
        default="historical_budget",
        min_length=1,
        max_length=50,
        description="Document family stamped on every new document.",
    )
    chunk_type: str = Field(
        default="budget_component",
        min_length=1,
        max_length=50,
        description="chunk_type stamped on every new chunk (filterable).",
    )


class IndexRunResponse(BaseModel):
    """Response of ``POST /embeddings/index/runs`` (HTTP 202)."""

    job_id: uuid.UUID
    documents_total: int = Field(ge=0, description="Documents submitted for indexing.")
    status: IndexJobStatus


class IndexJobView(BaseModel):
    """Response of ``GET /embeddings/index/jobs/{job_id}``: progress snapshot."""

    job_id: uuid.UUID
    status: IndexJobStatus
    documents_processed: int = Field(
        ge=0, description="Documents processed so far (indexed + skipped duplicates)."
    )
    error_message: str | None = None
    started_at: datetime
    finished_at: datetime | None = None


class CollectionStats(BaseModel):
    """Corpus size and index state of one collection table."""

    collection: str
    documents: int = Field(ge=0, description="Distinct documents with chunks in this collection.")
    chunks: int = Field(ge=0, description="Chunk rows in this collection.")
    hnsw_indexed: bool = Field(description="Whether the collection carries an HNSW vector index.")


class CorpusStats(BaseModel):
    """Whole-corpus snapshot, per collection."""

    collections: list[CollectionStats] = Field(default_factory=list)
    total_chunks: int = Field(ge=0)
