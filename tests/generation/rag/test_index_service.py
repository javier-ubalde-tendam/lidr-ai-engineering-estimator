"""CorpusIndexService: the batch loop (indexed vs skipped, chunk count, progress) over a fake ingester."""

import pytest

from app.generation.rag.index_service import CorpusIndexService
from app.generation.rag.schemas import Budget
from tests.index_helpers import FakeIngester, budget_dict

pytestmark = pytest.mark.anyio


def budget(budget_id: str, components: int = 2) -> Budget:
    return Budget.model_validate(budget_dict(budget_id, components))


async def test_expand_counts_indexed_documents_chunks_and_progress() -> None:
    service = CorpusIndexService(FakeIngester())
    progress: list[int] = []

    result = await service.expand([budget("NEW-1", 2), budget("NEW-2", 3)], on_progress=progress.append)

    assert (result.documents_indexed, result.documents_skipped, result.chunks_created) == (2, 0, 5)
    assert progress == [1, 2]


async def test_expand_skips_duplicates_without_failing_the_batch() -> None:
    service = CorpusIndexService(FakeIngester(duplicates={"DUP"}))
    progress: list[int] = []

    result = await service.expand(
        [budget("NEW-1", 1), budget("DUP", 4), budget("NEW-2", 1)], on_progress=progress.append
    )

    assert (result.documents_indexed, result.documents_skipped) == (2, 1)
    assert result.chunks_created == 2  # the duplicate's chunks are not counted
    assert progress == [1, 2, 3]


async def test_expand_stamps_the_source_path_and_the_chunk_type() -> None:
    ingester = FakeIngester()
    service = CorpusIndexService(ingester)

    await service.expand([budget("NEW-1", 1)], chunk_type="historical_task", source_prefix="batch-7")

    request = ingester.requests[0]
    assert request.source_path == "batch-7::NEW-1" and request.chunk_type == "historical_task"
    assert request.document_type == "historical_budget"


async def test_an_ingest_error_other_than_a_duplicate_propagates() -> None:
    service = CorpusIndexService(FakeIngester(fail_on="BOOM"))

    try:
        await service.expand([budget("NEW-1", 1), budget("BOOM", 1)])
    except RuntimeError as exc:
        assert "provider down" in str(exc)
    else:
        raise AssertionError("the failure must reach the caller (the job marks itself failed)")
