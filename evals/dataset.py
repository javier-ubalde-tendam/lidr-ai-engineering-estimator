"""Dataset dorado de los evals: casos tipados + cargador del JSON."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

TierName = Literal["auto", "executive", "pm", "developer", "default"]


class GoldenCase(BaseModel):
    """Una transcripción y el rango esperado de la respuesta.

    Casi todo son rangos y no igualdades exactas: un LLM tiene margen legítimo al dimensionar
    una fase. El objetivo es cazar regresiones obvias, no clavar cada cifra.
    """

    id: str = Field(min_length=3, max_length=64)
    transcript: str = Field(min_length=20, max_length=80_000)
    project_type: Literal["mobile_app", "web_saas", "internal_tool", "data_pipeline"]
    detail_level: Literal["summary", "medium", "detailed"] = "medium"
    output_format: Literal["phases_table", "line_items", "narrative"] = "phases_table"
    tier: TierName = "auto"  # "auto" = sin override: lo decide el resolver

    expected_out_of_scope: bool = False
    expected_in_summary: list[str] = Field(default_factory=list)
    expected_technologies_any_of: list[str] = Field(default_factory=list)
    expected_phase_count_range: tuple[int, int] | None = None
    expected_cost_range_eur: tuple[int, int] | None = None
    expected_duration_weeks_range: tuple[int, int] | None = None
    expected_tier: TierName | None = None  # se compara con last_resolved_tier de la sesión


_DEFAULT_PATH = Path(__file__).resolve().parent / "golden_dataset.json"


def load_dataset(path: Path | str | None = None) -> list[GoldenCase]:
    raw = json.loads(Path(path or _DEFAULT_PATH).read_text(encoding="utf-8"))
    return [GoldenCase.model_validate(entry) for entry in raw]
