import pytest

from app.sessions import tier_resolver
from app.sessions.models import ProjectMetadata
from app.sessions.tier_resolver import Tier, TierRule, resolve_tier

NEUTRAL = "We want a booking portal for a small gym with a member calendar."


def _resolve(transcript: str = NEUTRAL, metadata: ProjectMetadata | None = None, override: Tier | None = None):
    return resolve_tier(transcript=transcript, metadata=metadata or ProjectMetadata(), override=override)


def test_nda_in_the_transcript_resolves_to_executive():
    assert _resolve("Everything here is under NDA until launch.") == (Tier.EXECUTIVE, "nda_detected")


def test_nda_in_the_agreed_scope_also_counts():
    metadata = ProjectMetadata(agreed_scope="Confidential rollout for the board")

    assert _resolve(metadata=metadata) == (Tier.EXECUTIVE, "nda_detected")


def test_regulatory_context_resolves_to_executive():
    assert _resolve("The platform must be HIPAA compliant.") == (Tier.EXECUTIVE, "regulatory_context")


def test_regulatory_term_among_the_technologies_counts():
    metadata = ProjectMetadata(mentioned_technologies=["Postgres", "PCI-DSS"])

    assert _resolve(metadata=metadata) == (Tier.EXECUTIVE, "regulatory_context")


def test_two_distinct_technical_keywords_resolve_to_developer():
    assert _resolve("We deploy on Kubernetes with Docker images.") == (Tier.DEVELOPER, "technical_audience")


def test_one_technical_keyword_is_not_enough():
    assert _resolve("We deploy with Docker, nothing else.") == (Tier.DEFAULT, "no_rule_matched")


def test_the_same_keyword_repeated_does_not_count_twice():
    assert _resolve("Docker, docker and more DOCKER.") == (Tier.DEFAULT, "no_rule_matched")


@pytest.mark.parametrize(("team_size", "expected"), [(1, Tier.PM), (2, Tier.PM), (3, Tier.DEFAULT)])
def test_small_team_resolves_to_pm(team_size, expected):
    tier, rule = _resolve(metadata=ProjectMetadata(assumed_team_size=team_size))

    assert tier == expected
    assert rule == ("low_budget_pm" if expected == Tier.PM else "no_rule_matched")


def test_nothing_matching_resolves_to_default():
    assert _resolve() == (Tier.DEFAULT, "no_rule_matched")


def test_precedence_follows_the_order_of_the_chain():
    transcript = "Under NDA, HIPAA applies, and we use Kafka and Terraform."
    metadata = ProjectMetadata(assumed_team_size=1)

    assert _resolve(transcript, metadata) == (Tier.EXECUTIVE, "nda_detected")
    assert _resolve("HIPAA applies, and we use Kafka and Terraform.", metadata) == (Tier.EXECUTIVE, "regulatory_context")
    assert _resolve("I am the CEO and we use Kafka and Terraform.", metadata) == (Tier.EXECUTIVE, "executive_role")
    assert _resolve("We use Kafka and Terraform.", metadata) == (Tier.DEVELOPER, "technical_audience")
    assert _resolve(NEUTRAL, metadata) == (Tier.PM, "low_budget_pm")


def test_override_always_wins():
    assert _resolve("Everything is under NDA.", override=Tier.DEVELOPER) == (Tier.DEVELOPER, "explicit_override")


@pytest.mark.parametrize(
    "text",
    [
        "Soy director general de la empresa y quiero una landing page.",
        "La directora general quiere ver una propuesta para la web.",
        "Soy el gerente general y necesito un presupuesto.",
        "I'm the CEO and I need a quote for a landing page.",
    ],
)
def test_an_executive_role_resolves_to_executive(text):
    assert _resolve(text) == (Tier.EXECUTIVE, "executive_role")


def test_a_technical_role_is_not_an_executive_role():
    assert _resolve("Soy el CTO y quiero una landing page.") == (Tier.DEFAULT, "no_rule_matched")


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        ("El proyecto está sujeto a un acuerdo de confidencialidad.", "nda_detected"),
        ("Los datos son confidenciales según el contrato.", "nda_detected"),
        ("Tenemos que cumplir el RGPD.", "regulatory_context"),
    ],
)
def test_spanish_variants_of_nda_and_regulation(text, rule):
    assert _resolve(text)[1] == rule


def test_a_failing_predicate_is_logged_and_skipped(monkeypatch):
    from structlog.testing import capture_logs

    def broken(_ctx):
        raise RuntimeError("boom")

    rules = (TierRule("broken_rule", Tier.EXECUTIVE, broken), *tier_resolver._RULES[2:])
    monkeypatch.setattr(tier_resolver, "_RULES", rules)

    with capture_logs() as logs:
        result = _resolve("We deploy on Kubernetes with Docker images.")

    assert result == (Tier.DEVELOPER, "technical_audience")
    assert any(log["event"] == "tier_rule_failed" and log["rule"] == "broken_rule" for log in logs)


def test_tier_resolved_is_logged():
    from structlog.testing import capture_logs

    with capture_logs() as logs:
        _resolve("Under NDA.")

    event = next(log for log in logs if log["event"] == "tier_resolved")
    assert (event["tier"], event["rule"]) == ("executive", "nda_detected")
