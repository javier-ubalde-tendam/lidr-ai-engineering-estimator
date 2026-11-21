"""Fakes for the corpus-expansion tests (no DB, no embeddings provider)."""

from collections.abc import Collection
from typing import Any

from app.generation.rag.errors import DocumentAlreadyIngestedError
from app.generation.rag.schemas import IngestRequest, IngestResponse


def budget_dict(budget_id: str, components: int = 2) -> dict[str, Any]:
    return {
        "budget_id": budget_id,
        "client_metadata": {"name": "Acme", "sector": "finance", "country": "ES"},
        "project_summary": "A new historical project",
        "main_technology": "Python",
        "year": 2025,
        "total_estimated_hours": 100 * components,
        "components": [
            {
                "component_id": f"C-{i}",
                "name": f"Component {i}",
                "description": "work",
                "module": "Core",
                "tech_stack": ["Python"],
                "estimated_hours": 100,
                "complexity": "medium",
                "dependencies": [],
            }
            for i in range(1, components + 1)
        ],
    }


class FakeIngester:
    """Raises ``DocumentAlreadyIngestedError`` for ``duplicates`` and a RuntimeError for ``fail_on``."""

    def __init__(self, duplicates: Collection[str] = (), fail_on: str | None = None) -> None:
        self.duplicates = duplicates
        self.fail_on = fail_on
        self.requests: list[IngestRequest] = []

    async def ingest(self, request: IngestRequest) -> IngestResponse:
        self.requests.append(request)
        budget_id = request.content.budget_id
        if budget_id in self.duplicates:
            raise DocumentAlreadyIngestedError(1)
        if budget_id == self.fail_on:
            raise RuntimeError("embedding provider down")
        return IngestResponse(
            document_id=len(self.requests),
            chunks_created=len(request.content.components),
            embedding_dimension=1536,
            ingestion_time_ms=1,
        )
