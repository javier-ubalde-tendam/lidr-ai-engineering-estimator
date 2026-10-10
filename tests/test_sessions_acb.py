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
TRANSCRIPT = "Kickoff call transcript for the new client portal project."
TOTALS_MISMATCH = {**GOOD_ESTIMATION_PAYLOAD, "total_cost_eur": 10500}
NEEDS_ITERATION = {
    "verdict": "needs_iteration",
    "issues": [
        {
            "category": "math_error",
            "severity": "critical",
            "field_path": "total_cost_eur",
            "description": "Phases sum to 10000 but the total says 10500.",
            "suggested_fix": "Use the exact sum of the phases.",
        }
    ],
    "confidence_in_review": 85,
}
ACCEPT = {"verdict": "accept", "issues": [], "confidence_in_review": 90}


def _setup(monkeypatch, estimation_payloads=None, critic_payloads=None):
    store = SessionStore(max_conversation_turns=6)
    app.dependency_overrides[get_session_store] = lambda: store
    monkeypatch.setattr("app.services.llm_service.get_openai_client", FakeOpenAIClient)
    use_fake_estimation_llm(monkeypatch, estimation_payloads or [GOOD_ESTIMATION_PAYLOAD])
    fake = use_fake_llm(monkeypatch)
    if critic_payloads:
        fake.payloads["CriticFeedback"] = critic_payloads
    return store, fake


def _post_acb(session_id: str, **extra):
    return client.post(
        f"/sessions/{session_id}/estimate-acb", data={**FORM_FIELDS, "transcript": TRANSCRIPT, **extra.pop("data", {})}, **extra
    )


def test_acb_endpoint_returns_the_estimation_with_the_trace_and_the_call_meta(monkeypatch):
    _, fake = _setup(monkeypatch, critic_payloads=[ACCEPT])
    try:
        session_id = client.post("/sessions").json()["session_id"]

        response = _post_acb(session_id)

        assert response.status_code == 200
        body = response.json()
        assert body["result"]["total_cost_eur"] == 10000
        assert body["prompt_version"] == "v4"
        assert body["cached"] is False
        assert body["acb"]["final_decision"] == "accept"
        assert body["acb"]["iterations_run"] == 1
        assert body["acb"]["iterations"][0]["critic_verdict"] == "accept"
        # Meta agregado de actor + critic (cada llamada falsa: 10 tokens de entrada, 5 de salida)
        assert (body["tokens_in"], body["tokens_out"]) == (20, 10)
        assert body["acb"]["llm_usage"]["tokens_in"] == 20
        assert {"latency_ms", "cost_usd", "model", "provider"} <= body.keys()
        assert len(fake.calls["CriticFeedback"]) == 1
    finally:
        app.dependency_overrides.clear()


def test_acb_feeds_the_critic_feedback_to_the_second_actor_attempt(monkeypatch):
    _, fake = _setup(
        monkeypatch,
        estimation_payloads=[TOTALS_MISMATCH, GOOD_ESTIMATION_PAYLOAD],
        critic_payloads=[NEEDS_ITERATION, ACCEPT],
    )
    try:
        session_id = client.post("/sessions").json()["session_id"]

        response = _post_acb(session_id)

        assert response.status_code == 200
        body = response.json()
        assert body["result"]["total_cost_eur"] == 10000  # el borrador corregido
        assert [i["decision_after"] for i in body["acb"]["iterations"]] == ["iterate", "accept"]
        first_attempt, second_attempt = fake.calls["EstimationResult"]
        assert "<critic_feedback>" not in first_attempt[-1]["content"]
        assert "<critic_feedback>" in second_attempt[-1]["content"]
        assert "[critical] math_error at total_cost_eur" in second_attempt[-1]["content"]
        # 2 llamadas del actor + 2 del critic
        assert body["tokens_in"] == 40
    finally:
        app.dependency_overrides.clear()


def test_acb_appends_a_single_turn_to_the_history_with_the_final_result(monkeypatch):
    store, _ = _setup(
        monkeypatch,
        estimation_payloads=[TOTALS_MISMATCH, GOOD_ESTIMATION_PAYLOAD],
        critic_payloads=[NEEDS_ITERATION, ACCEPT],
    )
    try:
        session_id = client.post("/sessions").json()["session_id"]

        _post_acb(session_id)

        history = store.get_or_404(session_id).history
        assert [m.role for m in history.messages] == ["user", "assistant"]  # UN solo append pese a 2 borradores
        assert '"total_cost_eur":10000' in history.messages[1].content
        assert "<critic_feedback>" not in history.messages[0].content
        assert client.get(f"/sessions/{session_id}").json()["message_count"] == 2
    finally:
        app.dependency_overrides.clear()


def test_acb_exhausted_budget_returns_the_annotated_draft(monkeypatch):
    _setup(monkeypatch, critic_payloads=[NEEDS_ITERATION])
    try:
        session_id = client.post("/sessions").json()["session_id"]

        body = _post_acb(session_id).json()

        assert body["acb"]["final_decision"] == "synthesize"
        assert body["acb"]["iterations_run"] == 3
        assert body["result"]["summary"].startswith("⚠ Open caveats")
        assert body["result"]["confidence_pct"] == 40  # 80 // 2
    finally:
        app.dependency_overrides.clear()


def test_acb_is_fail_open_when_the_critic_breaks(monkeypatch):
    _setup(monkeypatch, critic_payloads=[RuntimeError("critic provider down")])
    try:
        session_id = client.post("/sessions").json()["session_id"]

        response = _post_acb(session_id)

        assert response.status_code == 200
        body = response.json()
        assert body["acb"]["final_decision"] == "accept"
        assert body["acb"]["iterations"][0]["critic_confidence"] == 0
        assert body["tokens_in"] == 10  # solo cuenta la llamada del actor: la del critic falló
    finally:
        app.dependency_overrides.clear()


def test_acb_honours_the_tier_override(monkeypatch):
    _setup(monkeypatch, critic_payloads=[ACCEPT])
    try:
        session_id = client.post("/sessions").json()["session_id"]

        _post_acb(session_id, data={"tier": "pm"})

        info = client.get(f"/sessions/{session_id}").json()
        assert (info["last_resolved_tier"], info["last_tier_rule"]) == ("pm", "explicit_override")
    finally:
        app.dependency_overrides.clear()


def test_acb_returns_404_for_an_unknown_session(monkeypatch):
    _setup(monkeypatch)
    try:
        response = _post_acb("does-not-exist")

        assert response.status_code == 404
        assert response.json()["detail"] == "session_not_found"
    finally:
        app.dependency_overrides.clear()


def test_acb_returns_415_for_an_unsupported_attachment(monkeypatch):
    _setup(monkeypatch)
    try:
        session_id = client.post("/sessions").json()["session_id"]

        response = _post_acb(session_id, files=[("attachments", ("notes.txt", b"plain text", "text/plain"))])

        assert response.status_code == 415
    finally:
        app.dependency_overrides.clear()


def test_acb_returns_400_for_an_input_guardrail_violation(monkeypatch):
    _, fake = _setup(monkeypatch)
    try:
        session_id = client.post("/sessions").json()["session_id"]

        response = _post_acb(session_id, data={"transcript": "Contact me at juan.perez@example.com for the details"})

        assert response.status_code == 400
        assert response.json()["detail"]["reason"] == "pii"
        assert fake.calls["EstimationResult"] == []
    finally:
        app.dependency_overrides.clear()


def test_acb_returns_502_when_the_actor_fails(monkeypatch):
    _setup(monkeypatch, estimation_payloads=[RuntimeError("provider down")])
    try:
        session_id = client.post("/sessions").json()["session_id"]

        response = _post_acb(session_id)

        assert response.status_code == 502
        assert "provider down" not in response.text
    finally:
        app.dependency_overrides.clear()
