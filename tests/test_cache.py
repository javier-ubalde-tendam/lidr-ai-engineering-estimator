from app.cache import build_cache_key
from app.prompts.loader import render_estimation_prompt
from app.schemas.estimation import DetailLevel, EstimationRequest, OutputFormat, ProjectType


def make_request(**overrides) -> EstimationRequest:
    fields = {
        "description": "Mobile app with login, chat and push notifications.",
        "project_type": ProjectType.MOBILE_APP,
        "detail_level": DetailLevel.SUMMARY,
        "output_format": OutputFormat.NARRATIVE,
    }
    return EstimationRequest(**{**fields, **overrides})


def test_cache_key_is_stable_for_the_same_request():
    assert build_cache_key(*render_estimation_prompt(make_request())) == build_cache_key(
        *render_estimation_prompt(make_request())
    )


def test_cache_key_changes_with_each_form_field():
    base = build_cache_key(*render_estimation_prompt(make_request()))

    variants = [
        make_request(project_type=ProjectType.WEB_SAAS),
        make_request(detail_level=DetailLevel.DETAILED),
        make_request(output_format=OutputFormat.LINE_ITEMS),
    ]
    for variant in variants:
        assert build_cache_key(*render_estimation_prompt(variant)) != base
