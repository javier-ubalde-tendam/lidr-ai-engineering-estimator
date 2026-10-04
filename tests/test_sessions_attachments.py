from fastapi.testclient import TestClient

from app.dependencies import get_session_store
from app.main import app
from app.sessions.store import SessionStore
from tests.sessions_helpers import (
    EMPTY_METADATA_PAYLOAD,
    GOOD_ESTIMATION_PAYLOAD,
    FakeOpenAIClient,
    make_pdf_bytes,
    use_fake_estimation_llm,
    use_fake_metadata_llm,
)

client = TestClient(app)

FORM_FIELDS = {"project_type": "web_saas", "detail_level": "summary", "output_format": "narrative"}
TRANSCRIPT = "Kickoff call transcript for the new client portal project."
ATTACHMENT_MARKER = "SPECIAL_BUDGET_MARKER_50000_EUR"


def _estimation_payload_that_reacts_to_the_attachment(messages: list[dict]) -> dict:
    # Simula que el LLM "vio" el texto del adjunto: distingue la respuesta si el marcador aparece
    combined = " ".join(m["content"] for m in messages)
    if ATTACHMENT_MARKER in combined:
        return {**GOOD_ESTIMATION_PAYLOAD, "summary": "Web platform with billing and reporting. Budget confirmed."}
    return GOOD_ESTIMATION_PAYLOAD


def _setup(monkeypatch, estimation_payloads=None):
    store = SessionStore(max_conversation_turns=6)
    app.dependency_overrides[get_session_store] = lambda: store
    monkeypatch.setattr("app.services.llm_service.get_openai_client", FakeOpenAIClient)
    estimation_calls = use_fake_estimation_llm(
        monkeypatch, estimation_payloads or [_estimation_payload_that_reacts_to_the_attachment]
    )
    use_fake_metadata_llm(monkeypatch, [EMPTY_METADATA_PAYLOAD])
    return estimation_calls


def test_attachment_text_reaches_the_llm_messages_with_the_delimiter(monkeypatch):
    estimation_calls = _setup(monkeypatch)
    try:
        session_id = client.post("/sessions").json()["session_id"]
        pdf_bytes = make_pdf_bytes(ATTACHMENT_MARKER)

        response = client.post(
            f"/sessions/{session_id}/estimate",
            data={**FORM_FIELDS, "transcript": TRANSCRIPT},
            files=[("attachments", ("budget.pdf", pdf_bytes, "application/pdf"))],
        )

        assert response.status_code == 200
        last_user_message = estimation_calls[-1][-1]["content"]
        assert "--- attachment: budget.pdf ---" in last_user_message
        assert ATTACHMENT_MARKER in last_user_message
        assert "--- end attachment ---" in last_user_message
    finally:
        app.dependency_overrides.clear()


def test_result_differs_from_the_same_request_without_an_attachment(monkeypatch):
    estimation_calls = _setup(monkeypatch)
    try:
        session_id_with = client.post("/sessions").json()["session_id"]
        pdf_bytes = make_pdf_bytes(ATTACHMENT_MARKER)
        response_with = client.post(
            f"/sessions/{session_id_with}/estimate",
            data={**FORM_FIELDS, "transcript": TRANSCRIPT},
            files=[("attachments", ("budget.pdf", pdf_bytes, "application/pdf"))],
        )

        session_id_without = client.post("/sessions").json()["session_id"]
        response_without = client.post(
            f"/sessions/{session_id_without}/estimate",
            data={**FORM_FIELDS, "transcript": TRANSCRIPT},
        )

        assert response_with.status_code == 200
        assert response_without.status_code == 200
        assert response_with.json()["result"]["summary"] != response_without.json()["result"]["summary"]
        assert len(estimation_calls) == 2
    finally:
        app.dependency_overrides.clear()


def test_unsupported_attachment_extension_returns_415(monkeypatch):
    _setup(monkeypatch, estimation_payloads=[GOOD_ESTIMATION_PAYLOAD])
    try:
        session_id = client.post("/sessions").json()["session_id"]

        response = client.post(
            f"/sessions/{session_id}/estimate",
            data={**FORM_FIELDS, "transcript": TRANSCRIPT},
            files=[("attachments", ("notes.txt", b"plain text notes", "text/plain"))],
        )

        assert response.status_code == 415
    finally:
        app.dependency_overrides.clear()
