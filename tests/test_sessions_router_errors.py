from fastapi.testclient import TestClient

from app.dependencies import get_session_store
from app.main import app
from app.services.llm_service import EstimationFailedError
from app.sessions.store import SessionStore
from tests.sessions_helpers import EMPTY_METADATA_PAYLOAD, FakeOpenAIClient, use_fake_metadata_llm

client = TestClient(app)

FORM_FIELDS = {"project_type": "web_saas", "detail_level": "summary", "output_format": "narrative"}
TRANSCRIPT = "Kickoff call transcript for the new client portal project."


def test_get_session_returns_404_for_unknown_session():
    app.dependency_overrides[get_session_store] = lambda: SessionStore(max_conversation_turns=6)
    try:
        # Caso 404 puro: no hace falta compartir instancia porque no se crea ninguna sesión antes
        response = client.get("/sessions/does-not-exist")
        assert response.status_code == 404
        assert response.json()["detail"] == "session_not_found"
    finally:
        app.dependency_overrides.clear()


def test_estimate_returns_404_for_unknown_session():
    app.dependency_overrides[get_session_store] = lambda: SessionStore(max_conversation_turns=6)
    try:
        # Caso 404 puro: no hace falta compartir instancia porque no se crea ninguna sesión antes
        response = client.post(
            "/sessions/does-not-exist/estimate", data={**FORM_FIELDS, "transcript": TRANSCRIPT}
        )
        assert response.status_code == 404
        assert response.json()["detail"] == "session_not_found"
    finally:
        app.dependency_overrides.clear()


def test_estimate_rejects_transcript_with_pii(monkeypatch):
    store = SessionStore(max_conversation_turns=6)
    app.dependency_overrides[get_session_store] = lambda: store
    monkeypatch.setattr("app.services.llm_service.get_openai_client", FakeOpenAIClient)
    use_fake_metadata_llm(monkeypatch, [EMPTY_METADATA_PAYLOAD])
    try:
        session_id = client.post("/sessions").json()["session_id"]
        response = client.post(
            f"/sessions/{session_id}/estimate",
            data={**FORM_FIELDS, "transcript": "Contact me at juan.perez@example.com for more project details"},
        )
        assert response.status_code == 400
        assert response.json()["detail"]["reason"] == "pii"
    finally:
        app.dependency_overrides.clear()


def test_estimate_returns_422_when_attachment_extraction_fails():
    store = SessionStore(max_conversation_turns=6)
    app.dependency_overrides[get_session_store] = lambda: store
    try:
        session_id = client.post("/sessions").json()["session_id"]
        response = client.post(
            f"/sessions/{session_id}/estimate",
            data={**FORM_FIELDS, "transcript": TRANSCRIPT},
            files=[("attachments", ("broken.pdf", b"not a pdf", "application/pdf"))],
        )
        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_estimate_returns_502_when_the_llm_fails(monkeypatch):
    store = SessionStore(max_conversation_turns=6)
    app.dependency_overrides[get_session_store] = lambda: store

    def failing_estimate_conversational(**kwargs):
        raise EstimationFailedError("No valid estimation after retries")

    monkeypatch.setattr("app.routers.sessions.estimate_conversational", failing_estimate_conversational)
    try:
        session_id = client.post("/sessions").json()["session_id"]
        response = client.post(
            f"/sessions/{session_id}/estimate",
            data={**FORM_FIELDS, "transcript": TRANSCRIPT},
        )
        assert response.status_code == 502
        assert "No valid estimation" not in response.text
    finally:
        app.dependency_overrides.clear()
