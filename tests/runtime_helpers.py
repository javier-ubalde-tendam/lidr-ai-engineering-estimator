"""Fakes of the Redis-backed runtime config so endpoint tests never touch Redis."""

from app.config import get_settings


class FakeRuntimeRetrievalConfig:
    """Stands in for ``RuntimeRetrievalConfig``: an optional override, else the live Settings.

    Settings are read on every call (like the real object does after a Redis miss), so tests
    that ``monkeypatch.setenv`` keep working.
    """

    def __init__(
        self,
        search_mode: str | None = None,
        rerank: bool | None = None,
        routing: bool | None = None,
        query_transform: bool | None = None,
        temporal_decay: bool | None = None,
        augmentation: bool | None = None,
        hallucination_gate: bool | None = None,
        synthesis: bool | None = None,
        task_hours_top_k: int | None = None,
        task_hours_distance_threshold: float | None = None,
    ) -> None:
        self.search_mode = search_mode
        self.rerank = rerank
        self.routing = routing
        self.query_transform = query_transform
        self.temporal_decay = temporal_decay
        self.augmentation = augmentation
        self.hallucination_gate = hallucination_gate
        self.synthesis = synthesis
        self.task_hours_top_k = task_hours_top_k
        self.task_hours_distance_threshold = task_hours_distance_threshold

    def effective_search_mode(self) -> str:
        return self.search_mode or get_settings().RETRIEVAL_SEARCH_MODE

    def effective_rerank(self) -> bool:
        return get_settings().RERANKER_ENABLED if self.rerank is None else self.rerank

    def effective_routing(self) -> bool:
        return get_settings().RETRIEVAL_ROUTING_ENABLED if self.routing is None else self.routing

    def effective_query_transform(self) -> bool:
        settings = get_settings()
        return settings.QUERY_TRANSFORM_ENABLED if self.query_transform is None else self.query_transform

    def effective_temporal_decay(self) -> bool:
        settings = get_settings()
        return settings.TEMPORAL_DECAY_ENABLED if self.temporal_decay is None else self.temporal_decay

    def effective_augmentation(self) -> bool:
        settings = get_settings()
        return settings.AUGMENTATION_ENABLED if self.augmentation is None else self.augmentation

    def effective_hallucination_gate(self) -> bool:
        settings = get_settings()
        enabled = settings.HALLUCINATION_GATE_ENABLED
        return enabled if self.hallucination_gate is None else self.hallucination_gate

    def effective_synthesis(self) -> bool:
        settings = get_settings()
        return settings.SYNTHESIS_ENABLED if self.synthesis is None else self.synthesis

    def effective_task_hours_top_k(self) -> int:
        top_k = self.task_hours_top_k
        return get_settings().TASK_HOURS_TOP_K if top_k is None else top_k

    def effective_task_hours_distance_threshold(self) -> float:
        threshold = self.task_hours_distance_threshold
        return get_settings().TASK_HOURS_DISTANCE_THRESHOLD if threshold is None else threshold


class FakeRuntimeModelConfig:
    """Stands in for ``RuntimeModelConfig``: overrides by key, else the live Settings."""

    def __init__(self, overrides: dict[str, str] | None = None) -> None:
        self.overrides = overrides or {}

    def effective(self, key: str) -> str:
        return self.overrides.get(key) or str(getattr(get_settings(), key))
