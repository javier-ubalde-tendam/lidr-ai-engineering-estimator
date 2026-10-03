from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_BODY = {
    "description": "Aplicación móvil de reservas para un gimnasio con pagos online",
    "project_type": "mobile_app",
    "detail_level": "summary",
    "output_format": "narrative",
}

def test_estimate_returns_response_contract(monkeypatch):
    # Se parchea el nombre donde se USA (router), no donde se define (llm_service),
    # porque "from x import f" copia la referencia al módulo que importa
    monkeypatch.setattr(
        "app.routers.estimations.estimate_project",
        lambda request: ("# Estimación", 100, 200, "gpt-4o-mini"),
    )
    response = client.post("/api/v1/estimate", json=VALID_BODY)
    assert response.status_code == 200
    assert response.json() == {
        "estimation": "# Estimación",
        "prompt_version": "v1",
        "model": "gpt-4o-mini",
        "provider": "openai",  # fijado en conftest.py
        "tokens_input": 100,
        "tokens_output": 200,
    }


def test_estimate_reports_prompt_version_from_settings(monkeypatch):
    monkeypatch.setattr(
        "app.routers.estimations.estimate_project",
        lambda request: ("# Estimación", 1, 2, "gpt-4o-mini"),
    )
    monkeypatch.setenv("PROMPT_VERSION", "v7")
    response = client.post("/api/v1/estimate", json=VALID_BODY)
    assert response.json()["prompt_version"] == "v7"


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


def test_estimate_stream_emits_token_and_metrics_events(monkeypatch):
    def fake_stream(request, stats):
        stats["cache_hit"] = False
        yield "Hola "
        yield "mundo"

    monkeypatch.setattr("app.routers.estimations.estimate_project_stream", fake_stream)
    response = client.post("/api/v1/estimate/stream", json=VALID_BODY)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.count("event: token") == 2
    assert 'event: metrics\ndata: {"cache_hit": false, "prompt_version": "v1"}' in response.text