"""Fakes and builders for the graph tests: a scripted LLM wrapper and a scripted search backend."""

from typing import Any

from app.config import get_settings
from app.domain.graph import GraphDependencies
from app.domain.graph.schemas import (
    ComponentClassification,
    ComponentModel,
    RequirementsExtraction,
)
from app.generation.agentic.agent_schemas import AgentEstimate, SearchBudgetsArgs

TRANSCRIPT = "The client needs a customer portal with payments and reporting. " * 3


class ScriptedLLM:
    """Stands in for ``LLMWrapper``: returns the scripted model for each ``response_model``."""

    def __init__(self, estimate: AgentEstimate, components: list[tuple[str, str]]) -> None:
        self.answers: dict[type, Any] = {
            RequirementsExtraction: RequirementsExtraction(requirements=["pay online", "report"]),
            ComponentClassification: ComponentClassification(
                components=[ComponentModel(name=n, category=c) for n, c in components]
            ),
            AgentEstimate: estimate,
        }
        self.calls: list[dict[str, Any]] = []

    def complete_structured(
        self, *, system_prompt: str, user_message: str, response_model: type, **kwargs: Any
    ) -> tuple[Any, None]:
        self.calls.append({"model": response_model, "user_message": user_message, **kwargs})
        return self.answers[response_model], None


class ScriptedSearch:
    """Search backend keyed by the component name that starts the query; can fail on demand."""

    def __init__(self, hours: dict[str, list[float]], failing: tuple[str, ...] = ()) -> None:
        self.hours = hours
        self.failing = failing
        self.queries: list[str] = []

    async def __call__(self, args: SearchBudgetsArgs) -> list[dict[str, Any]]:
        self.queries.append(args.query)
        name = args.query.split(" (")[0]
        if name in self.failing:
            raise RuntimeError("retrieval down")
        return [
            {"budget_id": f"B-{name[:3]}-{i}", "estimated_hours": h, "distance": 0.2}
            for i, h in enumerate(self.hours.get(name, []))
        ]


def estimate_of(*parts: tuple[str, float, list[str]]) -> AgentEstimate:
    return AgentEstimate.model_validate(
        {
            "components": [
                {"name": n, "hours": h, "reference_budget_ids": ids} for n, h, ids in parts
            ],
            "total_hours": sum(h for _, h, _ in parts),
            "notes": "scripted",
        }
    )


def make_deps(llm: ScriptedLLM, search: ScriptedSearch) -> GraphDependencies:
    return GraphDependencies(
        llm_wrapper=llm,  # type: ignore[arg-type]
        search_backend=search,
        settings=get_settings(),
    )
