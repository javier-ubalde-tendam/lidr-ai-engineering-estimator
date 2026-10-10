import pytest

from app.sessions.compression.anchors import AnchorDetector
from app.sessions.models import Message
from tests.sessions_helpers import use_fake_llm


def _user(text: str) -> Message:
    return Message(role="user", content=text)


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        ("This project is under an NDA, please keep it private.", "nda"),
        ("Hemos firmado un acuerdo de confidencialidad con el cliente.", "nda"),
        ("The client signed the contract last week.", "signed_contract"),
        ("El contrato ya está firmado desde el lunes.", "signed_contract"),
        ("The scope is frozen until the next release.", "scope_frozen"),
        ("El alcance está congelado hasta la fase 2.", "scope_frozen"),
        ("Our budget is locked at 50k EUR.", "budget_locked"),
        ("El presupuesto cerrado es de 50.000 euros.", "budget_locked"),
        ("We must be HIPAA compliant from day one.", "compliance"),
        ("Tenemos que cumplir el RGPD y la ISO 27001.", "compliance"),
        ("There is a hard deadline on March 1st.", "deadline_hard"),
        ("La fecha límite innegociable es el 1 de marzo.", "deadline_hard"),
        ("We are contractually obliged to deliver the audit trail.", "contractual"),
        ("Estamos obligados por contrato a entregar el informe.", "contractual"),
        ("The client agreed to cover the cloud costs.", "explicit_commitment"),
        ("Nos hemos comprometido a entregar un piloto en mayo.", "explicit_commitment"),
    ],
)
def test_heuristic_detects_anchor_phrases_in_english_and_spanish(text, rule):
    match = AnchorDetector(mode="heuristic").detect(_user(text))

    assert match.is_anchor
    assert rule in match.matched_rules


@pytest.mark.parametrize(
    "text",
    [
        "We want a web portal for gym bookings with a member calendar.",
        "Queremos una aplicación móvil con notificaciones push y login.",
        "Can you add a dashboard for the sales team?",
        "Quizá el presupuesto sea de unos 50.000 euros, aún por confirmar.",
    ],
)
def test_heuristic_ignores_ordinary_turns(text):
    match = AnchorDetector(mode="heuristic").detect(_user(text))

    assert not match.is_anchor
    assert match.matched_rules == []


def test_heuristic_reports_every_rule_that_fired():
    text = "The NDA is signed and we need GDPR compliance."

    match = AnchorDetector(mode="heuristic").detect(_user(text))

    assert {"nda", "compliance"} <= set(match.matched_rules)


def test_llm_mode_uses_the_classifier_answer(monkeypatch):
    fake = use_fake_llm(monkeypatch)
    fake.payloads["AnchorClassification"] = [{"is_anchor": True, "reason": "frozen scope"}]

    match = AnchorDetector(mode="llm").detect(_user("Nothing the regexes would catch here, honestly."))

    assert match.is_anchor
    assert match.matched_rules == ["llm_classifier"]
    assert len(fake.calls["AnchorClassification"]) == 1


def test_llm_mode_failure_is_logged_and_treated_as_not_an_anchor(monkeypatch):
    from structlog.testing import capture_logs

    fake = use_fake_llm(monkeypatch)
    fake.payloads["AnchorClassification"] = [RuntimeError("provider down")]

    with capture_logs() as logs:
        # Contiene una frase que la heurística SÍ cazaría: el modo llm no cae a la heurística
        match = AnchorDetector(mode="llm").detect(_user("We signed the contract yesterday."))

    assert not match.is_anchor
    assert any(log["event"] == "anchor_llm_failed" for log in logs)
