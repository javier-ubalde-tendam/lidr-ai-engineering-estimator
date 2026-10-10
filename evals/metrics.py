"""Métricas binarias y deterministas sobre la respuesta de un caso dorado.

Se prefieren a un LLM-as-judge porque no cuestan, no varían entre ejecuciones y dicen exactamente
por qué falla un caso. Reciben el JSON de `result` de la respuesta para que un resultado que no
cumple el schema sea un fallo de métrica y no una excepción.
"""

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from app.schemas.estimation import OUT_OF_SCOPE_PREFIX, EstimationResult
from evals.dataset import GoldenCase


@dataclass
class MetricResult:
    name: str
    passed: bool
    details: str


def _parse(payload: dict[str, Any]) -> EstimationResult | None:
    try:
        return EstimationResult.model_validate(payload)
    except ValidationError:
        return None


def _in_range(value: int, bounds: tuple[int, int]) -> bool:
    return bounds[0] <= value <= bounds[1]


class SchemaAdherenceMetric:
    """Valida contra EstimationResult y comprueba que el nº de fases esté en el rango esperado."""

    name = "schema_adherence"

    def evaluate(self, case: GoldenCase, payload: dict[str, Any]) -> MetricResult:
        try:
            result = EstimationResult.model_validate(payload)
        except ValidationError as exc:
            return MetricResult(self.name, False, f"does not match EstimationResult: {exc.error_count()} error(s)")
        bounds = case.expected_phase_count_range
        if bounds and not _in_range(len(result.phases), bounds):
            return MetricResult(self.name, False, f"{len(result.phases)} phases outside {list(bounds)}")
        return MetricResult(self.name, True, "schema ok")


class CostBoundsMetric:
    """Coste y duración dentro de rango; en un caso fuera de alcance basta el prefijo 'Out of scope:'."""

    name = "cost_bounds"

    def evaluate(self, case: GoldenCase, payload: dict[str, Any]) -> MetricResult:
        result = _parse(payload)
        if result is None:
            return MetricResult(self.name, False, "unparseable result")

        if case.expected_out_of_scope:
            ok = result.summary.startswith(OUT_OF_SCOPE_PREFIX)
            return MetricResult(self.name, ok, "out-of-scope prefix present" if ok else "missing 'Out of scope:' prefix")

        problems: list[str] = []
        if case.expected_cost_range_eur and not _in_range(result.total_cost_eur, case.expected_cost_range_eur):
            problems.append(f"cost {result.total_cost_eur} outside {list(case.expected_cost_range_eur)}")
        if case.expected_duration_weeks_range and not _in_range(
            result.total_duration_weeks, case.expected_duration_weeks_range
        ):
            problems.append(
                f"duration {result.total_duration_weeks}w outside {list(case.expected_duration_weeks_range)}"
            )
        return MetricResult(self.name, not problems, "; ".join(problems) or "cost and duration in range")


class ContentRecallMetric:
    """El texto menciona lo esperado: todos los términos de expected_in_summary y alguna tecnología."""

    name = "content_recall"

    def evaluate(self, case: GoldenCase, payload: dict[str, Any]) -> MetricResult:
        if case.expected_out_of_scope:
            return MetricResult(self.name, True, "skipped: out-of-scope case")
        result = _parse(payload)
        if result is None:
            return MetricResult(self.name, False, "unparseable result")

        text = " ".join([result.summary, *(f"{p.name} {p.summary}" for p in result.phases)]).lower()
        problems: list[str] = []
        missing = [term for term in case.expected_in_summary if term.lower() not in text]
        if missing:
            problems.append(f"missing terms: {missing}")
        technologies = case.expected_technologies_any_of
        if technologies and not any(tech.lower() in text for tech in technologies):
            problems.append(f"none of {technologies} mentioned")
        return MetricResult(self.name, not problems, "; ".join(problems) or "expected content present")


_METRICS = (SchemaAdherenceMetric(), CostBoundsMetric(), ContentRecallMetric())


def run_all_metrics(case: GoldenCase, payload: dict[str, Any]) -> list[MetricResult]:
    return [metric.evaluate(case, payload) for metric in _METRICS]
