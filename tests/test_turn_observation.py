from fastapi.testclient import TestClient
from structlog.testing import capture_logs

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
OBSERVATION_FIELDS = {
    "turn_index",
    "session_id",
    "enriched_transcript_chars",
    "attachments_total_chars",
    "messages_in_window",
    "anchors_count",
    "summary_chars",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "latency_ms",
    "cache_hit_kind",
    "last_resolved_tier",
}
MAX_TURNS = 2


def _setup(monkeypatch):
    store = SessionStore(max_conversation_turns=MAX_TURNS)
    app.dependency_overrides[get_session_store] = lambda: store
    monkeypatch.setattr("app.services.llm_service.get_openai_client", FakeOpenAIClient)
    use_fake_estimation_llm(monkeypatch, [GOOD_ESTIMATION_PAYLOAD])
    use_fake_metadata_llm(monkeypatch, [EMPTY_METADATA_PAYLOAD])


def _turn(session_id: str, transcript: str, **kwargs):
    return client.post(
        f"/sessions/{session_id}/estimate", data={**FORM_FIELDS, "transcript": transcript}, **kwargs
    )


def test_the_response_carries_the_observation_with_every_field(monkeypatch):
    _setup(monkeypatch)
    try:
        session_id = client.post("/sessions").json()["session_id"]
        transcript = "We are building Northwind Booking, a web SaaS for gyms."

        body = _turn(session_id, transcript).json()

        observation = body["turn_observation"]
        assert set(observation) == OBSERVATION_FIELDS
        assert observation["turn_index"] == 1
        assert observation["session_id"] == session_id
        assert observation["enriched_transcript_chars"] == len(transcript)
        assert observation["attachments_total_chars"] == 0
        assert observation["messages_in_window"] == 2
        assert observation["anchors_count"] == 0
        assert observation["summary_chars"] == 0
        assert observation["cache_hit_kind"] == "none"
        assert observation["last_resolved_tier"] == "default"
        # Coinciden con los campos que la respuesta ya exponía (misma fuente: LLMCallMeta)
        assert observation["tokens_in"] == body["tokens_in"] == 10
        assert observation["tokens_out"] == body["tokens_out"] == 5
        assert observation["cost_usd"] == body["cost_usd"]
        assert observation["latency_ms"] == body["latency_ms"]
    finally:
        app.dependency_overrides.clear()


def test_the_existing_response_fields_are_unchanged(monkeypatch):
    _setup(monkeypatch)
    try:
        session_id = client.post("/sessions").json()["session_id"]

        body = _turn(session_id, "We are building Northwind Booking, a web SaaS for gyms.").json()

        assert {"result", "prompt_version", "cached", "latency_ms", "model", "provider"} <= set(body)
    finally:
        app.dependency_overrides.clear()


def test_turn_index_grows_and_the_window_and_summary_reflect_the_compression(monkeypatch):
    _setup(monkeypatch)
    try:
        session_id = client.post("/sessions").json()["session_id"]

        observations = [
            _turn(session_id, f"Turn {turn} update with enough detail to pass validation.").json()["turn_observation"]
            for turn in range(1, 5)
        ]

        assert [o["turn_index"] for o in observations] == [1, 2, 3, 4]
        assert [o["messages_in_window"] for o in observations] == [2, 4, 4, 4]  # tope = 2 * MAX_TURNS
        assert [o["summary_chars"] > 0 for o in observations] == [False, False, True, True]
    finally:
        app.dependency_overrides.clear()


def test_attachment_chars_are_counted_inside_the_enriched_transcript(monkeypatch):
    _setup(monkeypatch)
    try:
        session_id = client.post("/sessions").json()["session_id"]
        pdf = make_pdf_bytes("Attachment body with the budget breakdown.")

        observation = _turn(
            session_id,
            "Here is the budget breakdown from the client.",
            files=[("attachments", ("budget.pdf", pdf, "application/pdf"))],
        ).json()["turn_observation"]

        assert observation["attachments_total_chars"] > 0
        assert observation["enriched_transcript_chars"] > observation["attachments_total_chars"]
    finally:
        app.dependency_overrides.clear()


def test_turn_observed_is_emitted_as_a_structured_log_event(monkeypatch):
    _setup(monkeypatch)
    try:
        session_id = client.post("/sessions").json()["session_id"]

        with capture_logs() as logs:
            body = _turn(session_id, "We are building Northwind Booking, a web SaaS for gyms.").json()

        (event,) = [entry for entry in logs if entry["event"] == "turn_observed"]
        assert {k: v for k, v in event.items() if k not in ("event", "log_level")} == body["turn_observation"]
    finally:
        app.dependency_overrides.clear()


def test_a_failed_turn_does_not_advance_the_turn_counter(monkeypatch):
    _setup(monkeypatch)
    try:
        session_id = client.post("/sessions").json()["session_id"]
        failed = _turn(session_id, "Contact me at juan.perez@example.com for more project details")
        assert failed.status_code == 400

        observation = _turn(session_id, "We are building Northwind Booking, a web SaaS for gyms.").json()[
            "turn_observation"
        ]

        assert observation["turn_index"] == 1
    finally:
        app.dependency_overrides.clear()


def test_memory_endpoint_exposes_summary_anchors_and_metadata(monkeypatch):
    _setup(monkeypatch)
    try:
        session_id = client.post("/sessions").json()["session_id"]
        _turn(session_id, "We signed the contract with the client yesterday, so the scope is final.")
        for turn in range(1, 4):
            _turn(session_id, f"Turn {turn} update with enough detail to pass validation.")

        snapshot = client.get(f"/sessions/{session_id}/memory").json()

        assert set(snapshot) == {"session_id", "summary", "anchors", "metadata"}
        assert snapshot["summary"]  # los turnos no ancla expulsados acaban en el resumen
        assert len(snapshot["anchors"]) == 2  # el par user + assistant del turno ancla
        assert "We signed the contract" in snapshot["anchors"][0]
        assert snapshot["metadata"]["mentioned_technologies"] == []
    finally:
        app.dependency_overrides.clear()


def test_memory_endpoint_returns_404_for_an_unknown_session():
    app.dependency_overrides[get_session_store] = lambda: SessionStore(max_conversation_turns=6)
    try:
        response = client.get("/sessions/does-not-exist/memory")

        assert response.status_code == 404
        assert response.json()["detail"] == "session_not_found"
    finally:
        app.dependency_overrides.clear()
