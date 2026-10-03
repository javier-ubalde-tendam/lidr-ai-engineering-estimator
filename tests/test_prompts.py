import re

import pytest
from jinja2 import TemplateNotFound

from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import DetailLevel, EstimationRequest, EstimationResult, OutputFormat, ProjectType


def make_request(**overrides) -> EstimationRequest:
    fields = {
        "description": "Mobile app with login, chat and push notifications.",
        "project_type": ProjectType.MOBILE_APP,
        "detail_level": DetailLevel.SUMMARY,
        "output_format": OutputFormat.NARRATIVE,
    }
    return EstimationRequest(**{**fields, **overrides})


def test_user_prompt_wraps_description_literally_in_project_description_block():
    description = "Mobile app with login, chat and push notifications."

    _, user = render_estimation_prompt(make_request(description=description))

    # Se comprueba el bloque completo, no solo que la etiqueta y el texto aparezcan por separado
    assert f"<project_description>\n{description}\n</project_description>" in user


def test_user_prompt_keeps_description_untouched_even_with_special_characters():
    description = "Portal <b>B2B</b> & 'API' con {{ variable }} y {% bloque %} en el texto."

    _, user = render_estimation_prompt(make_request(description=description))

    # La description es un dato, no parte del template: ni se escapa ni se interpreta
    assert f"<project_description>\n{description}\n</project_description>" in user


def test_phases_table_format_includes_keyword_and_table_instructions():
    system, _ = render_estimation_prompt(make_request(output_format=OutputFormat.PHASES_TABLE))

    # "phases_table" también sale en la cabecera, por eso se comprueba además el texto propio de la rama
    assert "phases_table" in system
    assert "Render a Markdown table" in system


def test_narrative_format_excludes_phases_table_keyword():
    system, _ = render_estimation_prompt(make_request(output_format=OutputFormat.NARRATIVE))

    # "confidence_pct" no sirve para discriminar: aparece siempre en output_schema y en los ejemplos
    assert "phases_table" not in system
    assert "Do not use tables" in system


def test_detailed_level_asks_for_assumptions_per_phase():
    system, _ = render_estimation_prompt(make_request(detail_level=DetailLevel.DETAILED))

    assert "list assumptions per phase" in system


def test_summary_level_does_not_ask_for_assumptions_per_phase():
    system, _ = render_estimation_prompt(make_request(detail_level=DetailLevel.SUMMARY))

    assert "list assumptions per phase" not in system


def test_system_prompt_includes_examples_block():
    system, _ = render_estimation_prompt(make_request())

    assert "<examples>" in system


def test_unknown_prompt_version_raises():
    with pytest.raises(TemplateNotFound):
        render_estimation_prompt(make_request(), version="v999")


def test_prompt_version_defaults_to_settings(monkeypatch):
    monkeypatch.setenv("PROMPT_VERSION", "v999")

    with pytest.raises(TemplateNotFound):
        render_estimation_prompt(make_request())


def test_explicit_version_overrides_settings(monkeypatch):
    monkeypatch.setenv("PROMPT_VERSION", "v999")

    system, _ = render_estimation_prompt(make_request(), version="v1")

    assert "<examples>" in system


def test_v2_prompt_ignores_output_format_because_it_only_affects_presentation():
    renders = {
        render_estimation_prompt(make_request(output_format=output_format), version="v2")
        for output_format in OutputFormat
    }

    # Un solo prompt para los tres formatos: la caché se comparte y el LLM no recibe instrucciones de maquetación
    assert len(renders) == 1
    system, _ = renders.pop()
    assert "Render a Markdown table" not in system


def test_v2_detail_level_changes_the_text_budget_rules():
    detailed, _ = render_estimation_prompt(make_request(detail_level=DetailLevel.DETAILED), version="v2")
    summary, _ = render_estimation_prompt(make_request(detail_level=DetailLevel.SUMMARY), version="v2")

    assert "at least three risks" in detailed
    assert "at least three risks" not in summary


def test_v2_examples_are_valid_estimation_results():
    system, _ = render_estimation_prompt(make_request(), version="v2")

    blocks = re.findall(r"<estimation>\n(.*?)\n</estimation>", system, flags=re.DOTALL)

    # Si el schema cambia, los ejemplos del prompt dejan de ser válidos y este test lo avisa
    assert len(blocks) == 4
    for block in blocks:
        EstimationResult.model_validate_json(block)