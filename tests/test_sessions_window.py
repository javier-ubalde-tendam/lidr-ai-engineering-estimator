from fastapi.testclient import TestClient

from app.dependencies import get_session_store
from app.main import app
from app.sessions.models import SUMMARY_PREFIX
from app.sessions.store import SessionStore
from tests.sessions_helpers import (
    EMPTY_METADATA_PAYLOAD,
    GOOD_ESTIMATION_PAYLOAD,
    FakeOpenAIClient,
    use_fake_estimation_llm,
    use_fake_llm,
    use_fake_metadata_llm,
)

client = TestClient(app)

FORM_FIELDS = {"project_type": "web_saas", "detail_level": "summary", "output_format": "narrative"}
MAX_TURNS = 3


def _has_summary(call_messages: list[dict]) -> bool:
    return len(call_messages) > 1 and call_messages[1]["content"].startswith(SUMMARY_PREFIX)


def _post_turns(session_id: str, transcripts: list[str]) -> None:
    for transcript in transcripts:
        response = client.post(f"/sessions/{session_id}/estimate", data={**FORM_FIELDS, "transcript": transcript})
        assert response.status_code == 200


def _setup(monkeypatch):
    store = SessionStore(max_conversation_turns=MAX_TURNS)
    app.dependency_overrides[get_session_store] = lambda: store
    monkeypatch.setattr("app.services.llm_service.get_openai_client", FakeOpenAIClient)
    estimation_calls = use_fake_estimation_llm(monkeypatch, [GOOD_ESTIMATION_PAYLOAD])
    use_fake_metadata_llm(monkeypatch, [EMPTY_METADATA_PAYLOAD])
    return store, estimation_calls


def test_window_caps_the_messages_and_the_summary_appears_once_it_is_exceeded(monkeypatch):
    _store, estimation_calls = _setup(monkeypatch)

    try:
        session_id = client.post("/sessions").json()["session_id"]
        _post_turns(session_id, [f"Turn {turn} update with enough detail to pass validation." for turn in range(8)])

        # system + resumen (si hay) + anclas (ninguna aquí) + ventana + user actual
        for call_messages in estimation_calls:
            summary_messages = 1 if _has_summary(call_messages) else 0
            assert len(call_messages) <= 1 + summary_messages + 0 + 2 * MAX_TURNS + 1

        # Hasta llenar la ventana no hay resumen; la llamada siguiente al primer desbordamiento ya lo lleva
        assert not _has_summary(estimation_calls[MAX_TURNS])
        assert _has_summary(estimation_calls[MAX_TURNS + 1])

        info = client.get(f"/sessions/{session_id}").json()
        assert info["message_count"] == 2 * MAX_TURNS  # la compresión deja la ventana en su límite
        assert info["summary_chars"] > 0
        assert info["anchors_count"] == 0
    finally:
        app.dependency_overrides.clear()


def test_anchor_turns_survive_verbatim_and_widen_the_limit(monkeypatch):
    store, estimation_calls = _setup(monkeypatch)
    fake = use_fake_llm(monkeypatch)

    try:
        session_id = client.post("/sessions").json()["session_id"]
        transcripts = ["We signed the contract with the client yesterday, so the scope is final."] + [
            f"Turn {turn} update with enough detail to pass validation." for turn in range(1, 8)
        ]
        _post_turns(session_id, transcripts)

        info = client.get(f"/sessions/{session_id}").json()
        assert info["anchors_count"] == 2  # el par user+assistant del turno ancla
        assert info["message_count"] == 2 * MAX_TURNS

        last_call = estimation_calls[-1]
        anchors = len(store.get_or_404(session_id).history.anchors)
        assert len(last_call) <= 1 + 1 + anchors + 2 * MAX_TURNS + 1
        # El turno ancla viaja literal en el contexto aunque ya haya salido de la ventana
        assert any("We signed the contract" in message["content"] for message in last_call)
        # El resumen solo recibe los turnos que no eran anclas
        summary_inputs = " ".join(m["content"] for messages in fake.calls["SummaryEnvelope"] for m in messages)
        assert "We signed the contract" not in summary_inputs
    finally:
        app.dependency_overrides.clear()
