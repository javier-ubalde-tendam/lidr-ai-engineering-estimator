import json

from fastapi.testclient import TestClient

from app.main import app
from app.schemas.estimation import EstimationResult
from app.services.llm_service import EstimationFailedError

client = TestClient(app)

VALID_BODY = {
    "description": "Aplicación móvil de reservas para un gimnasio con pagos online",
    "project_type": "mobile_app",
    "detail_level": "summary",
    "output_format": "narrative",
}

VALID_RESULT = {
    "summary": "Mobile booking app with online payments.",
    "confidence_pct": 80,
    "phases": [
        {"name": "Design", "duration_weeks": 2, "cost_eur": 4000, "summary": "Design the app."},
        {"name": "Build", "duration_weeks": 3, "cost_eur": 6000, "summary": "Build the app."},
    ],
    "total_duration_weeks": 5,
    "total_cost_eur": 10000,
}


def fake_estimate_project(request):
    return EstimationResult.model_validate(VALID_RESULT), 100, 200, "gpt-4o-mini"


def test_estimate_returns_response_contract(monkeypatch):
    # Se parchea el nombre donde se USA (router), no donde se define (llm_service),
    # porque "from x import f" copia la referencia al módulo que importa
    monkeypatch.setattr("app.routers.estimations.estimate_project", fake_estimate_project)
    response = client.post("/api/v1/estimate", json=VALID_BODY)
    assert response.status_code == 200
    assert response.json() == {
        "result": VALID_RESULT,
        "prompt_version": "v1",
        "cached": False,
    }


def test_estimate_reports_prompt_version_from_settings(monkeypatch):
    monkeypatch.setattr("app.routers.estimations.estimate_project", fake_estimate_project)
    monkeypatch.setenv("PROMPT_VERSION", "v7")
    response = client.post("/api/v1/estimate", json=VALID_BODY)
    assert response.json()["prompt_version"] == "v7"


def test_estimate_returns_502_when_llm_output_is_not_valid(monkeypatch):
    def failing_estimate_project(request):
        raise EstimationFailedError("No valid estimation after retries")

    monkeypatch.setattr("app.routers.estimations.estimate_project", failing_estimate_project)
    response = client.post("/api/v1/estimate", json=VALID_BODY)
    assert response.status_code == 502
    assert "No valid estimation" not in response.text  # no se filtra el detalle interno


def test_estimate_rejects_short_description_without_calling_llm(monkeypatch):
    calls = []
    # La lista registra cada llamada: si el endpoint llegara al servicio, calls dejaría de estar vacía
    monkeypatch.setattr(
        "app.routers.estimations.estimate_project",
        lambda request: calls.append(request),
    )
    response = client.post("/api/v1/estimate", json={**VALID_BODY, "description": "corta"})
    assert response.status_code == 422
    assert calls == []


def parse_sse(text: str) -> list[tuple[str, dict]]:
    # Cada evento SSE son dos líneas ("event: x" y "data: {...}") separadas del siguiente por una línea en blanco
    events = []
    for block in text.strip().split("\n\n"):
        event_line, data_line = block.split("\n")
        events.append((event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: "))))
    return events


# Lo que emite Instructor mientras el JSON está a medias: campos None aunque el tipo declarado sea int
PARTIAL_RESULT = EstimationResult.model_construct(
    summary="Mobile booking",
    confidence_pct=None,
    phases=[],
    total_duration_weeks=None,
    total_cost_eur=None,
)


def test_estimate_stream_emits_partial_complete_and_metrics_events(monkeypatch):
    final = EstimationResult.model_validate(VALID_RESULT)

    def fake_stream(request, stats):
        stats["cache_hit"] = False
        yield "partial", PARTIAL_RESULT
        yield "partial", final
        yield "complete", final

    monkeypatch.setattr("app.routers.estimations.estimate_project_stream_structured", fake_stream)
    response = client.post("/api/v1/estimate/stream", json=VALID_BODY)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["partial", "partial", "complete", "metrics"]
    assert events[0][1]["total_cost_eur"] is None  # el parcial viaja con campos a None
    assert events[2][1] == VALID_RESULT  # mismo shape que EstimationResponse.result
    assert events[3][1] == {"cache_hit": False, "prompt_version": "v1"}


def test_estimate_stream_emits_error_event_when_the_final_validation_fails(monkeypatch):
    def failing_stream(request, stats):
        stats["cache_hit"] = False
        yield "partial", PARTIAL_RESULT
        raise EstimationFailedError("No valid estimation after retries")

    monkeypatch.setattr("app.routers.estimations.estimate_project_stream_structured", failing_stream)
    response = client.post("/api/v1/estimate/stream", json=VALID_BODY)

    assert response.status_code == 200  # las cabeceras ya se enviaron: el fallo viaja como evento
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["partial", "error", "metrics"]
    assert "No valid estimation" not in response.text  # no se filtra el detalle interno
