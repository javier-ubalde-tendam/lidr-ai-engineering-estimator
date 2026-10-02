import pytest
from pydantic import ValidationError

from app.schemas.estimation import EstimationRequest, ProjectType

VALID_FIELDS = {
    "description": "Aplicación web SaaS de facturación para pequeños negocios",
    "project_type": "web_saas",
    "detail_level": "summary",
    "output_format": "narrative",
}

def test_description_too_short_is_rejected():
    # Se pasan strings: Pydantic los convierte a Enum. pytest.raises equivale a assertThrows
    with pytest.raises(ValidationError):
        EstimationRequest(
            description="corta",
            project_type="web_saas",
            detail_level="summary",
            output_format="narrative",
        )


def test_valid_request_is_accepted():
    request = EstimationRequest(**VALID_FIELDS)  # ** desempaqueta el dict como argumentos con nombre
    assert request.project_type == ProjectType.WEB_SAAS


def test_unknown_project_type_is_rejected():
    with pytest.raises(ValidationError):
        EstimationRequest(**{**VALID_FIELDS, "project_type": "no_existe"})


def test_description_too_long_is_rejected():
    with pytest.raises(ValidationError):
        EstimationRequest(**{**VALID_FIELDS, "description": "x" * 2001})