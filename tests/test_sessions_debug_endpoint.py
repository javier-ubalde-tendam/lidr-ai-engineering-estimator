from fastapi.testclient import TestClient

from app.dependencies import get_session_store
from app.main import app
from app.sessions.store import SessionStore
from tests.sessions_helpers import (
    GOOD_ESTIMATION_PAYLOAD,
    FakeOpenAIClient,
    use_fake_estimation_llm,
    use_fake_llm,
)

client = TestClient(app)

FORM_FIELDS = {"project_type": "web_saas", "detail_level": "summary", "output_format": "narrative"}
EXPECTED_KEYS = {
    "session_id",
    "message_count",
    "max_turns",
    "metadata",
    "anchors_count",
    "summary_chars",
    "last_resolved_tier",
    "last_tier_rule",
}


def _setup(monkeypatch, max_turns: int = 6):
    store = SessionStore(max_conversation_turns=max_turns)
    app.dependency_overrides[get_session_store] = lambda: store
    monkeypatch.setattr("app.services.llm_service.get_openai_client", FakeOpenAIClient)
    use_fake_estimation_llm(monkeypatch, [GOOD_ESTIMATION_PAYLOAD])
    return use_fake_llm(monkeypatch)


def test_new_session_exposes_exactly_the_debug_fields_with_empty_values(monkeypatch):
    _setup(monkeypatch)
    try:
        session_id = client.post("/sessions").json()["session_id"]

        info = client.get(f"/sessions/{session_id}").json()

        assert set(info) == EXPECTED_KEYS
        assert info["message_count"] == 0
        assert info["max_turns"] == 6
        assert info["anchors_count"] == 0
        assert info["summary_chars"] == 0
        assert info["last_resolved_tier"] is None
        assert info["last_tier_rule"] is None
    finally:
        app.dependency_overrides.clear()


def test_the_resolved_tier_and_its_rule_are_reflected_after_a_turn(monkeypatch):
    _setup(monkeypatch)
    try:
        session_id = client.post("/sessions").json()["session_id"]

        client.post(
            f"/sessions/{session_id}/estimate",
            data={**FORM_FIELDS, "transcript": "Everything about this portal is under NDA until launch."},
        )

        info = client.get(f"/sessions/{session_id}").json()
        assert (info["last_resolved_tier"], info["last_tier_rule"]) == ("executive", "nda_detected")
        assert info["message_count"] == 2
    finally:
        app.dependency_overrides.clear()


def test_the_tier_form_field_overrides_the_resolver_and_reaches_the_prompt(monkeypatch):
    fake = _setup(monkeypatch)
    try:
        session_id = client.post("/sessions").json()["session_id"]

        response = client.post(
            f"/sessions/{session_id}/estimate",
            data={**FORM_FIELDS, "transcript": "Everything about this portal is under NDA until launch.", "tier": "developer"},
        )

        assert response.status_code == 200
        info = client.get(f"/sessions/{session_id}").json()
        assert (info["last_resolved_tier"], info["last_tier_rule"]) == ("developer", "explicit_override")
        system_prompt = fake.calls["EstimationResult"][0][0]["content"]
        assert "tech lead" in system_prompt
    finally:
        app.dependency_overrides.clear()


def test_an_unknown_tier_is_rejected_with_422(monkeypatch):
    _setup(monkeypatch)
    try:
        session_id = client.post("/sessions").json()["session_id"]

        response = client.post(
            f"/sessions/{session_id}/estimate",
            data={**FORM_FIELDS, "transcript": "A transcript that is long enough.", "tier": "ceo"},
        )

        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_the_tier_is_resolved_with_the_metadata_known_before_the_turn(monkeypatch):
    fake = _setup(monkeypatch)
    fake.payloads["ProjectMetadata"] = [
        {"project_name": None, "assumed_team_size": 1, "mentioned_technologies": [], "agreed_scope": None}
    ]
    try:
        session_id = client.post("/sessions").json()["session_id"]
        transcript = "We are one freelancer building a booking portal."

        client.post(f"/sessions/{session_id}/estimate", data={**FORM_FIELDS, "transcript": transcript})
        first = client.get(f"/sessions/{session_id}").json()
        client.post(f"/sessions/{session_id}/estimate", data={**FORM_FIELDS, "transcript": transcript})
        second = client.get(f"/sessions/{session_id}").json()

        # El 1er turno no conoce aún el team size (lo extrae el turno); el 2º ya lo usa
        assert first["last_tier_rule"] == "no_rule_matched"
        assert (second["last_resolved_tier"], second["last_tier_rule"]) == ("pm", "low_budget_pm")
    finally:
        app.dependency_overrides.clear()


def test_anchors_and_summary_counters_grow_once_the_window_overflows(monkeypatch):
    _setup(monkeypatch, max_turns=2)
    try:
        session_id = client.post("/sessions").json()["session_id"]
        transcripts = ["We signed the contract with the client, scope is final."] + [
            f"Turn {n} brings more detail about the portal." for n in range(1, 5)
        ]
        for transcript in transcripts:
            client.post(f"/sessions/{session_id}/estimate", data={**FORM_FIELDS, "transcript": transcript})

        info = client.get(f"/sessions/{session_id}").json()
        assert info["anchors_count"] == 2
        assert info["summary_chars"] > 0
        assert info["message_count"] == 4
    finally:
        app.dependency_overrides.clear()


def test_the_conversational_response_carries_the_call_meta(monkeypatch):
    _setup(monkeypatch)
    try:
        session_id = client.post("/sessions").json()["session_id"]

        body = client.post(
            f"/sessions/{session_id}/estimate", data={**FORM_FIELDS, "transcript": "A transcript that is long enough."}
        ).json()

        assert body["prompt_version"] == "v4"
        assert (body["tokens_in"], body["tokens_out"]) == (10, 5)
        assert body["model"] == "gpt-4o-mini"
        assert body["provider"] == "openai"
        assert body["cost_usd"] > 0
        assert body["latency_ms"] >= 0
        assert "acb" not in body
    finally:
        app.dependency_overrides.clear()
