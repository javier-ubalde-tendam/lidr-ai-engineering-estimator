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


ACCEPT_CRITIC_PAYLOAD = {"verdict": "accept", "issues": [], "confidence_in_review": 90}
SUMMARY_PAYLOAD = {"summary": "Fake running summary of the earlier turns."}
NOT_ANCHOR_PAYLOAD = {"is_anchor": False, "reason": "fake"}

# Respuesta por defecto de cada response_model: un test solo registra las que necesita variar
_DEFAULT_PAYLOADS: dict[str, dict] = {
    "EstimationResult": GOOD_ESTIMATION_PAYLOAD,
    "ProjectMetadata": EMPTY_METADATA_PAYLOAD,
    "CriticFeedback": ACCEPT_CRITIC_PAYLOAD,
    "SummaryEnvelope": SUMMARY_PAYLOAD,
    "AnchorClassification": NOT_ANCHOR_PAYLOAD,
}


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


class FakeLLM:
    """Un único LLM falso para todas las llamadas estructuradas; responde según el response_model.

    `payloads[nombre]` acepta dicts (respuesta fija), callables `messages -> dict` (para variar la
    respuesta según los mensajes) o excepciones (se lanzan). Se usa el último si hay más llamadas.
    """

    def __init__(self) -> None:
        self.calls: dict[str, list[list[dict]]] = {name: [] for name in _DEFAULT_PAYLOADS}
        self.payloads: dict[str, list] = {}

    def completion(self, *args, **kwargs):
        # Instructor convierte response_model en una tool con el nombre de la clase
        name = kwargs["tools"][0]["function"]["name"]
        self.calls[name].append(kwargs["messages"])
        options = self.payloads.get(name) or [_DEFAULT_PAYLOADS[name]]
        payload = options[min(len(self.calls[name]) - 1, len(options) - 1)]
        if callable(payload):
            payload = payload(kwargs["messages"])
        if isinstance(payload, Exception):
            raise payload
        return _tool_call_response(name, payload)


_current_fake: tuple[FakeLLM, object] | None = None


def use_fake_llm(monkeypatch) -> FakeLLM:
    """Instala (una vez por test) el FakeLLM en llm_wrapper; llamadas repetidas devuelven el mismo."""
    global _current_fake
    from app.services import llm_wrapper

    if _current_fake is not None and llm_wrapper._structured_client is _current_fake[1]:
        return _current_fake[0]
    fake = FakeLLM()
    client = instructor.from_litellm(fake.completion)
    monkeypatch.setattr(llm_wrapper, "_structured_client", client)
    _current_fake = (fake, client)
    return fake


def use_fake_estimation_llm(monkeypatch, payloads: list):
    """Registra las respuestas de EstimationResult; devuelve los `messages` de cada llamada."""
    fake = use_fake_llm(monkeypatch)
    fake.payloads["EstimationResult"] = payloads
    return fake.calls["EstimationResult"]


def use_fake_metadata_llm(monkeypatch, payloads: list[dict]):
    """Registra las respuestas de ProjectMetadata; devuelve los `messages` de cada llamada."""
    fake = use_fake_llm(monkeypatch)
    fake.payloads["ProjectMetadata"] = payloads
    return fake.calls["ProjectMetadata"]