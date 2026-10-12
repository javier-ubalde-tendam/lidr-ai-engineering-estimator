"""Métricas binarias y deterministas del stress test del CAG.

A diferencia de `evals/metrics.py` (caso dorado + resultado de estimación), aquí se evalúan la
observación de un turno (`turn_observation` de /estimate) y el snapshot de memoria de la sesión
(`GET /sessions/{id}/memory`). Reutilizan `MetricResult`. Los datos ausentes o con tipo inválido
dan un resultado `passed=False` con el motivo en `details`, no una excepción: en un stress test
un dato roto es una medida más.
"""

import json
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from evals.metrics import MetricResult

MemoryField = Literal["summary", "anchors", "metadata"]
DEFAULT_MEMORY_FIELDS: tuple[MemoryField, ...] = ("summary", "anchors", "metadata")


def _number(observation: Mapping[str, Any] | None, key: str) -> float | None:
    """Valor numérico de la observación, o None si falta o no es un número (bool no cuenta)."""
    if not isinstance(observation, Mapping):
        return None
    value = observation.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


class LatencyBudgetMetric:
    """Pasa si `latency_ms <= budget_ms` (el límite es inclusivo)."""

    name = "latency_budget"

    def __init__(self, budget_ms: float) -> None:
        if budget_ms < 0:
            raise ValueError("budget_ms must be >= 0")
        self.budget_ms = budget_ms

    def evaluate(self, observation: Mapping[str, Any] | None) -> MetricResult:
        latency = _number(observation, "latency_ms")
        if latency is None:
            return MetricResult(self.name, False, "latency_ms missing or not a number")
        passed = latency <= self.budget_ms
        return MetricResult(self.name, passed, f"latency_ms={latency:g} budget_ms={self.budget_ms:g}")


class CostBudgetMetric:
    """Pasa si `cost_usd <= budget_usd` (el límite es inclusivo)."""

    name = "cost_budget"

    def __init__(self, budget_usd: float) -> None:
        if budget_usd < 0:
            raise ValueError("budget_usd must be >= 0")
        self.budget_usd = budget_usd

    def evaluate(self, observation: Mapping[str, Any] | None) -> MetricResult:
        cost = _number(observation, "cost_usd")
        if cost is None:
            return MetricResult(self.name, False, "cost_usd missing or not a number")
        passed = cost <= self.budget_usd
        return MetricResult(self.name, passed, f"cost_usd={cost:g} budget_usd={self.budget_usd:g}")


def _strings(value: Any) -> Iterable[str]:
    """Todos los textos de un valor anidado (valores, no claves)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _strings(item)
    elif value is not None:
        yield json.dumps(value)  # números y bools: "5" debe poder encontrarse como texto


class MemoryDriftMetric:
    """Pasa si `fact` aparece (sin distinguir mayúsculas) en alguno de los campos `where` del snapshot.

    Búsqueda de subcadena literal: sin embeddings ni LLM. Es sensible al formato ("30000" no
    encuentra "30,000"), por eso los hechos de los escenarios son strings que el LLM no suele reformatear.
    """

    name = "memory_drift"

    def __init__(self, fact: str, where: Iterable[MemoryField] = DEFAULT_MEMORY_FIELDS) -> None:
        if not fact or not fact.strip():
            raise ValueError("fact must be a non-empty string")
        self.fact = fact
        self.where = (where,) if isinstance(where, str) else tuple(where)
        unknown = set(self.where) - set(DEFAULT_MEMORY_FIELDS)
        if not self.where or unknown:
            raise ValueError(f"where must be a non-empty subset of {DEFAULT_MEMORY_FIELDS}, got {self.where}")

    def evaluate(self, snapshot: Mapping[str, Any] | None) -> MetricResult:
        if not isinstance(snapshot, Mapping):
            return MetricResult(self.name, False, f"snapshot missing or invalid; fact={self.fact!r}")
        needle = self.fact.lower()
        found_in = [
            field for field in self.where if any(needle in text.lower() for text in _strings(snapshot.get(field)))
        ]
        if found_in:
            return MetricResult(self.name, True, f"fact={self.fact!r} found in {found_in}")
        return MetricResult(self.name, False, f"fact={self.fact!r} not found in {list(self.where)}")
