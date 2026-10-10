import pytest
from pydantic import ValidationError

from app.schemas.critic import CriticFeedback, CriticIssue
from app.schemas.estimation import EstimationResult
from app.services import critic as critic_service
from app.sessions.models import ProjectMetadata
from app.sessions.tier_resolver import Tier
from tests.sessions_helpers import GOOD_ESTIMATION_PAYLOAD, use_fake_llm


def issue(severity="major", category="scope_mismatch", **overrides) -> dict:
    return {
        "category": category,
        "severity": severity,
        "field_path": "summary",
        "description": "Something is off here.",
        **overrides,
    }


def test_accept_without_issues_is_valid():
    feedback = CriticFeedback(verdict="accept", issues=[], confidence_in_review=90)

    assert feedback.issues == []


def test_accept_may_carry_minor_issues():
    CriticFeedback(verdict="accept", issues=[issue("minor")], confidence_in_review=70)


@pytest.mark.parametrize("severity", ["critical", "major"])
def test_needs_iteration_with_a_blocking_issue_is_valid(severity):
    CriticFeedback(verdict="needs_iteration", issues=[issue(severity)], confidence_in_review=70)


def test_needs_iteration_without_issues_is_rejected():
    with pytest.raises(ValidationError, match="needs_iteration"):
        CriticFeedback(verdict="needs_iteration", issues=[], confidence_in_review=70)


def test_needs_iteration_with_only_minor_issues_is_rejected():
    with pytest.raises(ValidationError, match="critical or major"):
        CriticFeedback(verdict="needs_iteration", issues=[issue("minor")], confidence_in_review=70)


def test_reject_requires_at_least_one_issue():
    with pytest.raises(ValidationError, match="reject"):
        CriticFeedback(verdict="reject", issues=[], confidence_in_review=70)
    CriticFeedback(verdict="reject", issues=[issue("minor")], confidence_in_review=70)


def test_at_most_twelve_issues_and_a_bounded_confidence():
    with pytest.raises(ValidationError):
        CriticFeedback(verdict="accept", issues=[issue("minor")] * 13, confidence_in_review=70)
    with pytest.raises(ValidationError):
        CriticFeedback(verdict="accept", issues=[], confidence_in_review=101)


def test_issue_category_and_severity_are_closed_sets():
    with pytest.raises(ValidationError):
        CriticIssue(**issue(category="typo"))
    with pytest.raises(ValidationError):
        CriticIssue(**issue(severity="blocker"))


def _review(estimation: EstimationResult | None = None):
    return critic_service.review(
        "We need a gym portal.",
        ProjectMetadata(),
        Tier.DEFAULT,
        estimation or EstimationResult.model_validate(GOOD_ESTIMATION_PAYLOAD),
    )


def test_review_returns_the_feedback_and_the_call_meta(monkeypatch):
    fake = use_fake_llm(monkeypatch)
    fake.payloads["CriticFeedback"] = [
        {"verdict": "needs_iteration", "issues": [issue("major")], "confidence_in_review": 75}
    ]

    feedback, meta = _review()

    assert feedback.verdict == "needs_iteration"
    assert meta is not None
    assert meta.tokens_in > 0
    # El prompt del critic incluye la estimación a auditar y el tier resuelto
    prompt = " ".join(m["content"] for m in fake.calls["CriticFeedback"][0])
    assert "phases[0]" in prompt
    assert "<resolved_tier>default</resolved_tier>" in prompt


def test_review_fails_open_with_an_accept_when_the_llm_raises(monkeypatch):
    from structlog.testing import capture_logs

    fake = use_fake_llm(monkeypatch)
    fake.payloads["CriticFeedback"] = [RuntimeError("provider down")]

    with capture_logs() as logs:
        feedback, meta = _review()

    assert (feedback.verdict, feedback.issues, feedback.confidence_in_review) == ("accept", [], 0)
    assert meta is None
    assert any(log["event"] == "critic_failed_fallback_accept" for log in logs)


def test_review_fails_open_when_the_output_never_validates(monkeypatch):
    fake = use_fake_llm(monkeypatch)
    # needs_iteration sin issues critical/major: Instructor reintenta y al final se rinde
    fake.payloads["CriticFeedback"] = [{"verdict": "needs_iteration", "issues": [], "confidence_in_review": 50}]

    feedback, meta = _review()

    assert feedback.verdict == "accept"
    assert meta is None
    assert len(fake.calls["CriticFeedback"]) == 4  # 1 intento + max_retries=3
