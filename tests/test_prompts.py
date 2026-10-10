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

# --- Prompt conversacional v4: <audience> por tier y <critic_feedback> opcional ---


def _conversational(tier, critic_feedback=None, metadata=None):
    from app.prompts.loader import render_conversational_prompt
    from app.sessions.models import ProjectMetadata

    return render_conversational_prompt(
        transcript="We need a web portal for gym bookings with payments.",
        project_type="web_saas",
        detail_level="medium",
        metadata=metadata or ProjectMetadata(),
        tier=tier,
        critic_feedback=critic_feedback,
        version="v4",
    )


def _feedback():
    from app.schemas.critic import CriticFeedback, CriticIssue

    return CriticFeedback(
        verdict="needs_iteration",
        issues=[
            CriticIssue(
                category="math_error",
                severity="critical",
                field_path="total_cost_eur",
                description="The phases add up to 10000 but the total says 10500.",
                suggested_fix="Set total_cost_eur to the sum of the phases.",
            ),
            CriticIssue(
                category="tier_mismatch",
                severity="minor",
                field_path="summary",
                description="Too much jargon for the audience.",
            ),
        ],
        confidence_in_review=80,
    )


@pytest.mark.parametrize(
    ("tier", "marker"),
    [
        ("executive", "C-level sponsor"),
        ("pm", "project manager"),
        ("developer", "tech lead"),
        ("default", "mixed stakeholders"),
        (None, "mixed stakeholders"),
    ],
)
def test_v4_audience_block_changes_with_the_tier(tier, marker):
    system, _ = _conversational(tier)

    audience = system.split("<audience>")[1].split("</audience>")[0]
    assert marker in audience
    other_markers = {"C-level sponsor", "project manager", "tech lead", "mixed stakeholders"} - {marker}
    assert not any(other in audience for other in other_markers)


def test_v4_audience_block_comes_after_the_project_metadata():
    system, _ = _conversational("pm")

    assert system.index("</project_metadata>") < system.index("<audience>")


def test_v4_treats_the_metadata_as_established_facts():
    system, _ = _conversational("default")

    assert "Treat the project_metadata as established facts" in system


def test_v4_renders_with_empty_metadata():
    system, user = _conversational("default")

    assert "(no project context yet)" in system
    assert "<project_description>" in user


def test_v4_critic_feedback_block_only_appears_when_there_is_feedback():
    _, without_feedback = _conversational("default")
    _, with_feedback = _conversational("default", critic_feedback=_feedback())

    assert "<critic_feedback>" not in without_feedback
    assert "<critic_feedback>" in with_feedback


def test_v4_critic_feedback_lists_every_issue_in_the_expected_format():
    _, user = _conversational("default", critic_feedback=_feedback())

    assert (
        "- [critical] math_error at total_cost_eur: The phases add up to 10000 but the total says 10500. "
        "(suggested fix: Set total_cost_eur to the sum of the phases.)"
    ) in user
    # Sin suggested_fix no se añade el paréntesis vacío
    assert "- [minor] tier_mismatch at summary: Too much jargon for the audience.\n" in user


def test_v3_prompt_still_renders_with_the_new_context_variables():
    from app.prompts.loader import render_conversational_prompt
    from app.sessions.models import ProjectMetadata

    system, user = render_conversational_prompt(
        transcript="We need a web portal for gym bookings.",
        project_type="web_saas",
        detail_level="medium",
        metadata=ProjectMetadata(),
        tier="executive",
        critic_feedback=None,
        version="v3",
    )

    assert "<audience>" not in system
    assert "<critic_feedback>" not in user
