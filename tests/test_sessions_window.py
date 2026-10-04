from fastapi.testclient import TestClient

from app.dependencies import get_session_store
from app.main import app
from app.sessions.store import SessionStore
from tests.sessions_helpers import (
    EMPTY_METADATA_PAYLOAD,
    GOOD_ESTIMATION_PAYLOAD,
    FakeOpenAIClient,
    use_fake_estimation_llm,
    use_fake_metadata_llm,
)

client = TestClient(app)

FORM_FIELDS = {"project_type": "web_saas", "detail_level": "summary", "output_format": "narrative"}
MAX_TURNS = 3


def test_sliding_window_caps_messages_sent_to_the_llm_and_stored_in_history(monkeypatch):
    store = SessionStore(max_conversation_turns=MAX_TURNS)
    app.dependency_overrides[get_session_store] = lambda: store
    monkeypatch.setattr("app.services.llm_service.get_openai_client", FakeOpenAIClient)
    estimation_calls = use_fake_estimation_llm(monkeypatch, [GOOD_ESTIMATION_PAYLOAD])
    use_fake_metadata_llm(monkeypatch, [EMPTY_METADATA_PAYLOAD])

    try:
        session_id = client.post("/sessions").json()["session_id"]

        for turn in range(8):
            response = client.post(
                f"/sessions/{session_id}/estimate",
                data={**FORM_FIELDS, "transcript": f"Turn {turn} update with enough detail to pass validation."},
            )
            assert response.status_code == 200

        # 1 (system) + 2*MAX_TURNS (historial) + 1 (user actual): ninguna llamada debe superar este tope
        max_messages_per_call = 1 + 2 * MAX_TURNS + 1
        for call_messages in estimation_calls:
            assert len(call_messages) <= max_messages_per_call

        info = client.get(f"/sessions/{session_id}").json()
        assert info["message_count"] <= 2 * MAX_TURNS
        assert info["message_count"] == 2 * MAX_TURNS  # tras 8 turnos ya se alcanzó el límite
    finally:
        app.dependency_overrides.clear()
