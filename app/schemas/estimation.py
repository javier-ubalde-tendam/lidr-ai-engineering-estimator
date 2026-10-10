from enum import Enum
from pydantic import BaseModel, Field, model_validator


class ProjectType(str, Enum):
    MOBILE_APP = "mobile_app"
    WEB_SAAS = "web_saas"
    INTERNAL_TOOL = "internal_tool"
    DATA_PIPELINE = "data_pipeline"

class DetailLevel(str, Enum):
    SUMMARY = "summary"
    MEDIUM = "medium"
    DETAILED = "detailed"

class OutputFormat(str, Enum):
    PHASES_TABLE = "phases_table"
    LINE_ITEMS = "line_items"
    NARRATIVE = "narrative"

class EstimationRequest(BaseModel):
    description: str = Field(min_length=20, max_length=10000)
    project_type: ProjectType
    detail_level: DetailLevel
    output_format: OutputFormat

OUT_OF_SCOPE_PREFIX = "Out of scope:"
LOW_CONFIDENCE_THRESHOLD = 30

class Phase(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    duration_weeks: int = Field(ge=1, le=52)
    cost_eur: int = Field(ge=0, le=1_000_000)
    summary: str = Field(min_length=10, max_length=600)


class EstimationResult(BaseModel):
    summary: str = Field(min_length=10, max_length=1200)
    confidence_pct: int = Field(ge=0, le=100)
    phases: list[Phase] = Field(min_length=1, max_length=8)
    total_duration_weeks: int = Field(ge=1, le=104)
    # description viaja en el JSON Schema, así que también orienta al modelo
    total_cost_eur: int = Field(
        ge=0, le=2_000_000, description="Exact sum of every phase.cost_eur"
    )

    def totals_mismatch(self) -> str | None:
        # Ya no es un validator: un desajuste no rechaza el resultado, se avisa al usuario
        phases_cost = sum(p.cost_eur for p in self.phases)
        if self.total_cost_eur == phases_cost:
            return None
        return f"total_cost_eur is {self.total_cost_eur} but the phases sum to {phases_cost}"

    @model_validator(mode="after")  # corre con el objeto ya construido y puede cruzar campos
    def low_confidence_requires_out_of_scope_prefix(self) -> "EstimationResult":
        if self.confidence_pct < LOW_CONFIDENCE_THRESHOLD and not self.summary.startswith(
            OUT_OF_SCOPE_PREFIX
        ):
            raise ValueError(
                f"confidence_pct < {LOW_CONFIDENCE_THRESHOLD} requires summary to "
                f"start with {OUT_OF_SCOPE_PREFIX!r}; refuse the estimation if the "
                f"description is too vague to size"
            )
        return self

class EstimationResponse(BaseModel):
    result: EstimationResult
    prompt_version: str
    cached: bool = False


class ConversationalEstimationResponse(EstimationResponse):
    # Observabilidad de la llamada; opcionales para no romper a clientes que ya consumen EstimationResponse
    latency_ms: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    model: str | None = None
    provider: str | None = None