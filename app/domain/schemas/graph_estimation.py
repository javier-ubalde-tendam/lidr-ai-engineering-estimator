"""HTTP contract of ``POST /v1/estimate/graph`` (Session 13).

Own Pydantic models on purpose: this layer must not import ``domain/graph`` (the state's
TypedDicts). Hours are engineer-hours (the historical corpus stores hours, not euros).
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class GraphEstimateRequest(BaseModel):
    transcript: str = Field(min_length=100, max_length=50_000)
    # Becomes the LangGraph thread_id: reuse it to find the persisted run again
    estimation_id: str | None = Field(default=None, max_length=128)


class ComponentOut(BaseModel):
    name: str
    category: str


class BudgetMatchOut(BaseModel):
    component: str
    reference_budget_id: str | None = None
    amount: float
    distance: float


class GraphEstimateResponse(BaseModel):
    estimation_id: str
    status: Literal["validated", "needs_review"]
    estimate: dict[str, Any] | None = None
    requirements: list[str] = Field(default_factory=list)
    components: list[ComponentOut] = Field(default_factory=list)
    budget_matches: list[BudgetMatchOut] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
