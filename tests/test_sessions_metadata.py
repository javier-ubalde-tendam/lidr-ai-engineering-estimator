from fastapi.testclient import TestClient

from app.dependencies import get_session_store
from app.main import app
from app.sessions.store import SessionStore
from tests.sessions_helpers import (
    GOOD_ESTIMATION_PAYLOAD,
    FakeOpenAIClient,
    use_fake_estimation_llm,
    use_fake_metadata_llm,
)

client = TestClient(app)

TURN_1_METADATA = {
    "project_name": "Acme Portal",
    "assumed_team_size": None,
    "mentioned_technologies": ["React"],
    "agreed_scope": None,
}
TURN_2_METADATA = {
    "project_name": None,
    "assumed_team_size": 5,
    "mentioned_technologies": ["Postgres"],
    "agreed_scope": None,
}

FORM_FIELDS = {"project_type": "web_saas", "detail_level": "summary", "output_format": "narrative"}


def test_metadata_accumulates_across_turns_and_is_injected_in_the_next_system_prompt(monkeypatch):
    store = SessionStore(max_conversation_turns=6)
    app.dependency_overrides[get_session_store] = lambda: store
    monkeypatch.setattr("app.services.llm_service.get_openai_client", FakeOpenAIClient)
    estimation_calls = use_fake_estimation_llm(monkeypatch, [GOOD_ESTIMATION_PAYLOAD])
    use_fake_metadata_llm(monkeypatch, [TURN_1_METADATA, TURN_2_METADATA])

    try:
        session_id = client.post("/sessions").json()["session_id"]

        response_1 = client.post(
            f"/sessions/{session_id}/estimate",
            data={**FORM_FIELDS, "transcript": "We are building Acme Portal, a web SaaS."},
        )
        assert response_1.status_code == 200

        info_after_turn_1 = client.get(f"/sessions/{session_id}").json()
        assert info_after_turn_1["metadata"]["project_name"] == "Acme Portal"
        assert info_after_turn_1["metadata"]["mentioned_technologies"] == ["React"]

        response_2 = client.post(
            f"/sessions/{session_id}/estimate",
            data={**FORM_FIELDS, "transcript": "The team will have 5 people working on it."},
        )
        assert response_2.status_code == 200

        info_after_turn_2 = client.get(f"/sessions/{session_id}").json()
        # project_name del turno 1 se conserva: el turno 2 no lo vuelve a mencionar
        assert info_after_turn_2["metadata"]["project_name"] == "Acme Portal"
        assert info_after_turn_2["metadata"]["assumed_team_size"] == 5
        # Las tecnologías se acumulan entre turnos, no se sobrescriben
        assert info_after_turn_2["metadata"]["mentioned_technologies"] == ["React", "Postgres"]

        # El system prompt de la 2ª llamada de estimación debe reflejar el metadata acumulado tras el turno 1
        second_call_system_prompt = estimation_calls[1][0]["content"]
        assert "<project_metadata>" in second_call_system_prompt
        assert "Acme Portal" in second_call_system_prompt
        assert "React" in second_call_system_prompt
    finally:
        app.dependency_overrides.clear()


def test_first_turn_system_prompt_shows_no_project_context_yet(monkeypatch):
    store = SessionStore(max_conversation_turns=6)
    app.dependency_overrides[get_session_store] = lambda: store
    monkeypatch.setattr("app.services.llm_service.get_openai_client", FakeOpenAIClient)
    estimation_calls = use_fake_estimation_llm(monkeypatch, [GOOD_ESTIMATION_PAYLOAD])
    use_fake_metadata_llm(monkeypatch, [TURN_1_METADATA])

    try:
        session_id = client.post("/sessions").json()["session_id"]
        client.post(
            f"/sessions/{session_id}/estimate",
            data={**FORM_FIELDS, "transcript": "We are building Acme Portal, a web SaaS."},
        )

        first_call_system_prompt = estimation_calls[0][0]["content"]
        assert "(no project context yet)" in first_call_system_prompt
    finally:
        app.dependency_overrides.clear()
