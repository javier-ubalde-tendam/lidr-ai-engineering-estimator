"""Feedback estructurado del Critic: es el contrato entre Critic y Boss.

El Boss decide leyendo `verdict`, `severity` y `category`; con texto libre tendría que parsear prosa.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

CriticIssueCategory = Literal[
    "math_error",
    "hallucination",
    "scope_mismatch",
    "phase_imbalance",
    "missing_assumption",
    "unrealistic_estimate",
    "tier_mismatch",
]
CriticIssueSeverity = Literal["critical", "major", "minor"]
CriticVerdict = Literal["accept", "needs_iteration", "reject"]


class CriticIssue(BaseModel):
    category: CriticIssueCategory
    severity: CriticIssueSeverity
    # Campo de EstimationResult al que se refiere: "summary", "total_cost_eur", "phases[2].cost_eur"...
    field_path: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=5, max_length=500)
    suggested_fix: str | None = Field(default=None, max_length=300)


class CriticFeedback(BaseModel):
    verdict: CriticVerdict
    issues: list[CriticIssue] = Field(default_factory=list, max_length=12)
    confidence_in_review: int = Field(ge=0, le=100)

    @model_validator(mode="after")
    def verdict_matches_issues(self) -> "CriticFeedback":
        # Con solo issues minor no vale la pena otra llamada al actor: debe aceptar
        if self.verdict == "needs_iteration" and not any(
            issue.severity in {"critical", "major"} for issue in self.issues
        ):
            raise ValueError("verdict 'needs_iteration' requires at least one critical or major issue")
        if self.verdict == "reject" and not self.issues:
            raise ValueError("verdict 'reject' requires at least one issue explaining why")
        return self
