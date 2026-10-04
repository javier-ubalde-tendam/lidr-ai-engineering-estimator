"""Utilidades compartidas por los tests de sesiones conversacionales (no es un módulo de tests:
no sigue el patrón test_*.py, así que pytest no lo recolecta, solo se importa desde otros tests)."""

import json

import instructor
import litellm
from fpdf import FPDF


def make_pdf_bytes(text: str) -> bytes:
    # PDF mínimo válido generado con fpdf2 (dev-dependency ligera): suficiente para que pypdf
    # pueda extraer el texto en los tests, sin necesitar un fichero de ejemplo en el repo
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, text)
    return bytes(pdf.output())


class FakeModerationResult:
    def __init__(self, flagged: bool = False) -> None:
        self.flagged = flagged
        self.categories = type("Categories", (), {"model_dump": lambda self: {}})()


class FakeModerationResponse:
    def __init__(self, flagged: bool = False) -> None:
        self.results = [FakeModerationResult(flagged)]


class FakeModerations:
    def create(self, input: str) -> FakeModerationResponse:
        return FakeModerationResponse(flagged=False)


class FakeOpenAIClient:
    """Evita llamadas reales a la Moderation API en los tests de sesiones conversacionales."""

    def __init__(self) -> None:
        self.moderations = FakeModerations()


def _tool_call_response(tool_name: str, payload: dict) -> litellm.ModelResponse:
    # Misma forma que usa tests/test_llm_service.py: el JSON viaja en tool_calls, no en content
    return litellm.ModelResponse(
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": json.dumps(payload)},
                        }
                    ],
                },
            }
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        model="fake",
    )


def use_fake_estimation_llm(monkeypatch, payloads: list):
    """Sustituye llm_service._structured_client por un fake que registra los `messages` recibidos.

    `payloads` acepta dicts (resultado fijo) o callables `messages -> dict` (para variar la
    respuesta según el contenido de los mensajes, p. ej. si contienen texto de un adjunto).
    """
    from app.services import llm_service

    calls: list[list[dict]] = []

    def fake_completion(*args, **kwargs):
        calls.append(kwargs["messages"])
        payload = payloads[min(len(calls) - 1, len(payloads) - 1)]
        if callable(payload):
            payload = payload(kwargs["messages"])
        return _tool_call_response("EstimationResult", payload)

    monkeypatch.setattr(llm_service, "_structured_client", instructor.from_litellm(fake_completion))
    return calls


def use_fake_metadata_llm(monkeypatch, payloads: list[dict]):
    """Sustituye metadata_extractor._structured_client; devuelve los `messages` de cada llamada."""
    from app.sessions import metadata_extractor

    calls: list[list[dict]] = []

    def fake_completion(*args, **kwargs):
        calls.append(kwargs["messages"])
        payload = payloads[min(len(calls) - 1, len(payloads) - 1)]
        return _tool_call_response("ProjectMetadata", payload)

    monkeypatch.setattr(metadata_extractor, "_structured_client", instructor.from_litellm(fake_completion))
    return calls


EMPTY_METADATA_PAYLOAD = {
    "project_name": None,
    "assumed_team_size": None,
    "mentioned_technologies": [],
    "agreed_scope": None,
}

GOOD_ESTIMATION_PAYLOAD = {
    "summary": "Web platform with billing and reporting.",
    "confidence_pct": 80,
    "phases": [
        {"name": "Design", "duration_weeks": 2, "cost_eur": 4000, "summary": "Design the platform."},
        {"name": "Build", "duration_weeks": 3, "cost_eur": 6000, "summary": "Build the platform."},
    ],
    "total_duration_weeks": 5,
    "total_cost_eur": 10000,
}
