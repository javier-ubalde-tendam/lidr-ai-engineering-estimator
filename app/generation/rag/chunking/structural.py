import tiktoken

from app.config import get_settings
from app.generation.rag.chunking.base import Chunker
from app.generation.rag.schemas import Budget, BudgetComponent, Chunk


def render_component_text(budget: Budget, component: BudgetComponent) -> str:
    """Parent context header + component detail: the text every strategy embeds."""
    # Header contextual del padre: sin él, un componente genérico ("Backend services")
    # sería ambiguo; con él el embedding captura proyecto, sector y tecnología.
    # Only task-granular components carry a module: budgets keep the exact same text/metadata
    module_line = f"Module: {component.module}\n" if component.module else ""
    return (
        f"[Project: {budget.project_summary}]\n"
        f"[Client sector: {budget.client_metadata.sector} | Year: {budget.year} | "
        f"Main tech: {budget.main_technology}]\n"
        f"\n"
        f"{module_line}"
        f"Component: {component.name}\n"
        f"Description: {component.description}\n"
        f"Tech stack: {', '.join(component.tech_stack)}\n"
        f"Complexity: {component.complexity}\n"
        f"Estimated hours: {component.estimated_hours}"
    )


def serialize_budget(budget: Budget) -> str:
    """Whole budget as one readable text blob (header + one block per component).

    Used by the strategies that split a whole document instead of iterating components
    (recursive, semantic, hierarchical parent level, contextual retrieval).
    """
    header = (
        f"[Project: {budget.project_summary}]\n"
        f"[Client: {budget.client_metadata.name} | "
        f"Sector: {budget.client_metadata.sector} | Year: {budget.year} | "
        f"Main tech: {budget.main_technology}]"
    )
    blocks = [header]
    for component in budget.components:
        blocks.append(
            f"Component: {component.name}\n"
            f"Description: {component.description}\n"
            f"Tech stack: {', '.join(component.tech_stack)}\n"
            f"Complexity: {component.complexity}\n"
            f"Estimated hours: {component.estimated_hours}"
        )
    return "\n\n".join(blocks)


def component_metadata(
    budget: Budget, component: BudgetComponent
) -> dict[str, str | int | list[str]]:
    """Filterable metadata (not embedded) that travels with a component-level chunk."""
    metadata: dict[str, str | int | list[str]] = {
        "budget_id": budget.budget_id,
        "component_id": component.component_id,
        "client_sector": budget.client_metadata.sector,
        "main_technology": budget.main_technology,
        "year": budget.year,
        "complexity": component.complexity,
        "estimated_hours": component.estimated_hours,
    }
    # Chunk.metadata does not allow None values: the key only exists for task-granular components
    if component.module:
        metadata["module"] = component.module
    return metadata


class JSONStructuralChunker(Chunker):
    """Un componente del presupuesto = un chunk (sin overlap ni división de descripciones)."""

    strategy_name = "structural"

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
        text = render_component_text(budget, component)
        return Chunk(
            chunk_id=f"{budget.budget_id}::{component.component_id}",
            text=text,
            # Metadata filtrable (no se embebe): sirve para filtrar por sector/año al buscar
            metadata=component_metadata(budget, component),
            token_count=len(self._encoding.encode(text)),
        )
