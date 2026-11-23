"""Pydantic contracts of the estimation agent (Session 12).

Three groups live here:

- validated **tool arguments** (what the model sends to each tool),
- the **final answer** (``AgentEstimate``) and the strict JSON schema derived from it,
- the **trace** (``AgentStep`` / ``AgentResult``) that records every turn of the loop.

Hours are engineer-hours: the historical corpus stores hours, not euros.
"""

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

# Same tolerance everywhere a reported total is compared with the sum of its components
TOTAL_TOLERANCE_HOURS = 0.5

# RetrievalRequest.query_text requires at least 10 characters: reject shorter queries here so the
# model gets a clear error instead of a retrieval-layer validation failure
MIN_QUERY_LENGTH = 10

SearchQuery = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=MIN_QUERY_LENGTH, max_length=2000)
]


class _ToolArgs(BaseModel):
    # The model must not invent parameters: unknown keys are reported back as an error observation
    model_config = ConfigDict(extra="forbid")


# --- Tool arguments -----------------------------------------------------------------------------


class SearchBudgetsFilters(_ToolArgs):
    sectors: list[str] | None = None
    project_year_min: int | None = None
    project_year_max: int | None = None


class SearchBudgetsArgs(_ToolArgs):
    query: SearchQuery
    filters: SearchBudgetsFilters | None = None


class ReferenceItem(_ToolArgs):
    """One historical item chosen as reference: where it comes from and the hours it took."""

    budget_id: str = Field(min_length=1)
    estimated_hours: float = Field(ge=0)


class TaskReferences(_ToolArgs):
    """One engineering task and the similar historical items found for it."""

    name: str = Field(min_length=1)
    references: list[ReferenceItem]


class ComponentTasks(_ToolArgs):
    """A project component, estimated as the sum of its tasks."""

    name: str = Field(min_length=1)
    tasks: list[TaskReferences] = Field(min_length=1)


class CalculateEstimateArgs(_ToolArgs):
    components: list[ComponentTasks] = Field(min_length=1)


class TaskDraft(TaskReferences):
    estimated_hours: float


class ComponentDraft(_ToolArgs):
    name: str = Field(min_length=1)
    tasks: list[TaskDraft] = Field(min_length=1)


class ValidateEstimateArgs(_ToolArgs):
    components: list[ComponentDraft] = Field(min_length=1)
    total_hours: float


# --- Final answer -------------------------------------------------------------------------------


class ComponentEstimate(BaseModel):
    name: str
    hours: float
    reference_budget_ids: list[str]

    @field_validator("hours")
    @classmethod
    def hours_are_not_negative(cls, hours: float) -> float:
        if hours < 0:
            raise ValueError("hours must not be negative")
        return hours


class AgentEstimate(BaseModel):
    """Structured answer of the agent: no field has a default so the strict schema stays simple."""

    components: list[ComponentEstimate]
    total_hours: float
    notes: str

    @model_validator(mode="after")
    def total_matches_components(self) -> "AgentEstimate":
        if not self.components:
            raise ValueError("components must not be empty")
        expected = sum(component.hours for component in self.components)
        if abs(self.total_hours - expected) > TOTAL_TOLERANCE_HOURS:
            raise ValueError(
                f"total_hours ({self.total_hours}) must equal the sum of the components "
                f"({round(expected, 1)}) within {TOTAL_TOLERANCE_HOURS} h"
            )
        return self


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema of ``model`` accepted by OpenAI ``strict`` structured outputs.

    Strict mode needs every property in ``required`` and ``additionalProperties: false`` on every
    object (Pydantic emits neither). Constraints such as ``ge`` are not part of the model's
    schema on purpose; Pydantic validates them after the call.
    """
    schema = model.model_json_schema()
    _make_strict(schema)
    return schema


def _make_strict(node: Any) -> None:
    if isinstance(node, dict):
        if "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"])
        for value in node.values():
            _make_strict(value)
    elif isinstance(node, list):
        for value in node:
            _make_strict(value)


# --- Trace --------------------------------------------------------------------------------------


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class AgentAction(BaseModel):
    """One tool call of a turn and what came back from it."""

    tool: str
    call_id: str
    arguments: dict[str, Any]
    observation: dict[str, Any]
    duration_ms: int
    error: str | None = None


class AgentWarning(BaseModel):
    """A deterministic, non-blocking finding of the loop about the final answer."""

    code: str
    message: str


class AgentStep(BaseModel):
    """One model turn: what the agent decided (reasoning), did (actions) and learned."""

    step: int
    reasoning: str
    actions: list[AgentAction] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    duration_ms: int = 0
    # Set when the final answer of this turn was rejected by validation and re-requested
    validation_error: str | None = None
    # Findings about the final answer of this turn (reference ids that differ from the calculation,
    # a stale calculation that could no longer be fixed, ...)
    warnings: list[AgentWarning] = Field(default_factory=list)


class Coverage(BaseModel):
    """How much of the task list the history could price (computed by the system, not the model)."""

    budgeted_tasks: int
    total_tasks: int
    ratio: float


class AgentResult(BaseModel):
    status: Literal["done", "max_steps_exceeded", "error"]
    estimate: AgentEstimate | None = None
    trace: list[AgentStep] = Field(default_factory=list)
    iterations: int = 0
    usage: TokenUsage = Field(default_factory=TokenUsage)
    error: str | None = None
    # Computed by the loop from the LAST calculate_estimate, never filled in by the model
    coverage: Coverage | None = None
    unbudgeted_tasks: list[str] = Field(default_factory=list)
    # True when some task has no historical reference: it costs 0 h, so the total is a floor
    total_is_lower_bound: bool = False
    # Findings about the accepted final answer (e.g. ``stale_calculation``)
    warnings: list[AgentWarning] = Field(default_factory=list)
