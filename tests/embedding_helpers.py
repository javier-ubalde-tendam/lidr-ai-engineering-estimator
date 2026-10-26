import math
from collections.abc import Callable
from types import SimpleNamespace

from app.config import EMBEDDING_DIMENSION, SUPPORTED_EMBEDDING_MODEL
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.schemas import Budget


def make_budget(budget_id: str = "BUD-1", n_components: int = 2) -> Budget:
    return Budget.model_validate(
        {
            "budget_id": budget_id,
            "client_metadata": {"name": "Acme", "sector": "finance", "country": "ES"},
            "project_summary": "Mobile banking API",
            "main_technology": "python",
            "year": 2024,
            "total_estimated_hours": 100,
            "components": [
                {
                    "component_id": f"C-{i}",
                    "name": f"Component {i}",
                    "description": "Does something",
                    "tech_stack": ["python", "redis"],
                    "estimated_hours": 10,
                    "complexity": "medium",
                    "dependencies": [],
                }
                for i in range(n_components)
            ],
        }
    )


def angle_vector(angle: float) -> list[float]:
    """Vector de 1536 dims con norma 1 en el plano (e0, e1): distancia coseno a e0 = 1 - cos(angle)."""
    vector = [0.0] * EMBEDDING_DIMENSION
    vector[0] = math.cos(angle)
    vector[1] = math.sin(angle)
    return vector


class FakeOpenAIClient:
    """Imita `client.embeddings.create`: registra las llamadas y devuelve vectores de 1536 dims."""

    def __init__(self, vector_for: Callable[[str, int], list[float]] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self._vector_for = vector_for or (lambda _text, index: angle_vector(0.01 * (index + 1)))
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, model: str, input: list[str]) -> SimpleNamespace:
        self.calls.append({"model": model, "input": list(input)})
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=self._vector_for(text, index))
                for index, text in enumerate(input)
            ]
        )


def make_embedder(client: FakeOpenAIClient) -> OpenAIEmbedder:
    return OpenAIEmbedder(client=client, model=SUPPORTED_EMBEDDING_MODEL)  # type: ignore[arg-type]
