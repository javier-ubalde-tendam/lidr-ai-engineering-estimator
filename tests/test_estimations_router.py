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
        lambda description: ("# Estimación", 100, 200, "gpt-4o-mini"),
    )
    response = client.post("/api/v1/estimate", json=VALID_BODY)
    assert response.status_code == 200
    assert response.json() == {
        "estimation": "# Estimación",
        "prompt_version": "v0",
        "model": "gpt-4o-mini",
        "provider": "openai",  # fijado en conftest.py
        "tokens_input": 100,
        "tokens_output": 200,
    }


def test_estimate_rejects_short_description_without_calling_llm(monkeypatch):
    calls = []
    # La lista registra cada llamada: si el endpoint llegara al servicio, calls dejaría de estar vacía
    monkeypatch.setattr(
        "app.routers.estimations.estimate_project",
        lambda description: calls.append(description),
    )
    response = client.post("/api/v1/estimate", json={**VALID_BODY, "description": "corta"})
    assert response.status_code == 422
    assert calls == []