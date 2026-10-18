import tiktoken

from app.config import get_settings
from app.embedding_pipeline.schemas import Budget, BudgetComponent, Chunk


class JSONStructuralChunker:
    """Un componente del presupuesto = un chunk (sin overlap ni división de descripciones)."""

    def __init__(self) -> None:
        # El encoding se resuelve una vez por instancia (el DI la hace singleton): cargar el
        # vocabulario BPE es caro y no debe repetirse por chunk.
        self._encoding = tiktoken.encoding_for_model(get_settings().EMBEDDING_MODEL)

    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        # Comprehension con dos `for`: equivale a dos bucles anidados con list.add() en Java
        return [
            self._build_chunk(budget, component)
            for budget in budgets
            for component in budget.components
        ]

    def _build_chunk(self, budget: Budget, component: BudgetComponent) -> Chunk:
        sector = budget.client_metadata.sector
        # Header contextual del padre: sin él, un componente genérico ("Backend services")
        # sería ambiguo; con él el embedding captura proyecto, sector y tecnología.
        text = (
            f"[Project: {budget.project_summary}]\n"
            f"[Client sector: {sector} | Year: {budget.year} | "
            f"Main tech: {budget.main_technology}]\n"
            f"\n"
            f"Component: {component.name}\n"
            f"Description: {component.description}\n"
            f"Tech stack: {', '.join(component.tech_stack)}\n"
            f"Complexity: {component.complexity}\n"
            f"Estimated hours: {component.estimated_hours}"
        )
        return Chunk(
            chunk_id=f"{budget.budget_id}::{component.component_id}",
            text=text,
            # Metadata filtrable (no se embebe): sirve para filtrar por sector/año al buscar
            metadata={
                "budget_id": budget.budget_id,
                "component_id": component.component_id,
                "client_sector": sector,
                "main_technology": budget.main_technology,
                "year": budget.year,
                "complexity": component.complexity,
                "estimated_hours": component.estimated_hours,
            },
            token_count=len(self._encoding.encode(text)),
        )
