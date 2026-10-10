import pytest

from app.schemas.critic import CriticFeedback
from app.schemas.estimation import LOW_CONFIDENCE_THRESHOLD, OUT_OF_SCOPE_PREFIX, EstimationResult
from app.services.boss import SUMMARY_MAX_LENGTH, Boss
from tests.sessions_helpers import GOOD_ESTIMATION_PAYLOAD


def result(**overrides) -> EstimationResult:
    return EstimationResult.model_validate({**GOOD_ESTIMATION_PAYLOAD, **overrides})


def issue(severity="major", category="math_error", field_path="total_cost_eur", description="Totals do not add up."):
    return {"category": category, "severity": severity, "field_path": field_path, "description": description}


def feedback(verdict: str, issues: list[dict] | None = None, confidence: int = 80) -> CriticFeedback:
    return CriticFeedback.model_validate(
        {"verdict": verdict, "issues": issues or [], "confidence_in_review": confidence}
    )


ACCEPT = feedback("accept")
NEEDS_ITERATION = feedback("needs_iteration", [issue("major")])


class ScriptedActor:
    """Devuelve borradores distintos en cada llamada y recuerda el feedback que recibió."""

    def __init__(self, drafts: list[EstimationResult]) -> None:
        self.drafts = drafts
        self.received_feedback: list[CriticFeedback | None] = []

    def __call__(self, critic_feedback: CriticFeedback | None) -> EstimationResult:
        self.received_feedback.append(critic_feedback)
        return self.drafts[min(len(self.received_feedback) - 1, len(self.drafts) - 1)]


def scripted_critic(reviews: list[CriticFeedback]):
    calls: list[EstimationResult] = []

    def critic(draft: EstimationResult) -> CriticFeedback:
        calls.append(draft)
        return reviews[min(len(calls) - 1, len(reviews) - 1)]

    critic.calls = calls  # type: ignore[attr-defined]
    return critic


def test_accept_on_the_first_round_returns_the_draft_untouched():
    draft = result()
    actor, critic = ScriptedActor([draft]), scripted_critic([ACCEPT])

    final, trace = Boss(max_iterations=3).run(actor, critic)

    assert final is draft
    assert len(actor.received_feedback) == 1
    assert (trace.final_decision, trace.iterations_run) == ("accept", 1)
    assert [(i.iteration, i.decision_after, i.critic_verdict) for i in trace.iterations] == [(0, "accept", "accept")]


def test_iterate_then_accept_passes_the_feedback_to_the_actor_on_the_second_call():
    bad, fixed = result(total_cost_eur=10500), result()
    actor, critic = ScriptedActor([bad, fixed]), scripted_critic([NEEDS_ITERATION, ACCEPT])

    final, trace = Boss(max_iterations=3).run(actor, critic)

    assert final is fixed
    assert actor.received_feedback == [None, NEEDS_ITERATION]
    assert critic.calls == [bad, fixed]
    assert trace.final_decision == "accept"
    assert [i.decision_after for i in trace.iterations] == ["iterate", "accept"]
    assert trace.iterations[0].issue_summary == ["[major] math_error @ total_cost_eur"]


def test_exhausted_budget_synthesizes_the_last_draft_with_the_open_issues():
    from structlog.testing import capture_logs

    drafts = [result(summary=f"Draft {n} of the portal estimation.") for n in range(3)]
    actor, critic = ScriptedActor(drafts), scripted_critic([NEEDS_ITERATION])

    with capture_logs() as logs:
        final, trace = Boss(max_iterations=3).run(actor, critic)

    assert len(actor.received_feedback) == 3
    assert trace.final_decision == "synthesize"
    assert [i.decision_after for i in trace.iterations] == ["iterate", "iterate", "synthesize"]
    assert final.summary.startswith("⚠ Open caveats")
    assert "[major] math_error (total_cost_eur)" in final.summary
    assert final.summary.endswith("Draft 2 of the portal estimation.")  # el último borrador, no el primero
    assert any(log["event"] == "boss_iteration_budget_exhausted" for log in logs)


def test_reject_synthesizes_immediately_without_retrying():
    actor = ScriptedActor([result()])
    critic = scripted_critic([feedback("reject", [issue("critical", "scope_mismatch", "summary")])])

    final, trace = Boss(max_iterations=3).run(actor, critic)

    assert len(actor.received_feedback) == 1
    assert (trace.final_decision, trace.iterations_run) == ("synthesize", 1)
    assert "scope_mismatch" in final.summary


def test_synthesized_confidence_is_halved_but_never_below_the_low_confidence_threshold():
    reject = feedback("reject", [issue()])

    high, _ = Boss(max_iterations=1).run(ScriptedActor([result(confidence_pct=90)]), scripted_critic([reject]))
    low, _ = Boss(max_iterations=1).run(ScriptedActor([result(confidence_pct=40)]), scripted_critic([reject]))

    assert high.confidence_pct == 45
    assert low.confidence_pct == LOW_CONFIDENCE_THRESHOLD  # 40 // 2 = 20 caería en el validator de "Out of scope"
    # El resultado sigue siendo válido para el schema
    EstimationResult.model_validate(low.model_dump())


def test_synthesized_summary_respects_the_schema_max_length():
    long_summary = "x" * SUMMARY_MAX_LENGTH
    many_issues = [issue("major", field_path=f"phases[{n}].cost_eur", description="d" * 300) for n in range(8)]
    reject = feedback("reject", many_issues)

    final, _ = Boss(max_iterations=1).run(ScriptedActor([result(summary=long_summary)]), scripted_critic([reject]))

    assert len(final.summary) <= SUMMARY_MAX_LENGTH
    EstimationResult.model_validate(final.model_dump())
    # Sigue quedando sitio para el texto original del borrador
    assert final.summary.endswith("x")


def test_synthesize_keeps_the_most_severe_issues_first():
    reject = feedback("reject", [issue("minor", "tier_mismatch", "summary"), issue("critical", "hallucination", "summary")])

    final, _ = Boss(max_iterations=1).run(ScriptedActor([result()]), scripted_critic([reject]))

    assert final.summary.index("hallucination") < final.summary.index("tier_mismatch")


def test_an_out_of_scope_draft_is_not_annotated():
    draft = result(
        summary=f"{OUT_OF_SCOPE_PREFIX} the description is too vague.",
        confidence_pct=10,
        phases=[{"name": "Not estimated", "duration_weeks": 1, "cost_eur": 0, "summary": "Cannot be sized without info."}],
        total_duration_weeks=1,
        total_cost_eur=0,
    )

    final, _ = Boss(max_iterations=1).run(ScriptedActor([draft]), scripted_critic([feedback("reject", [issue()])]))

    assert final.summary == draft.summary
    assert final.confidence_pct == 10


def test_boss_iteration_is_logged_per_round():
    from structlog.testing import capture_logs

    with capture_logs() as logs:
        Boss(max_iterations=3).run(ScriptedActor([result()]), scripted_critic([NEEDS_ITERATION, ACCEPT]))

    rounds = [log for log in logs if log["event"] == "boss_iteration"]
    assert [(r["iteration"], r["decision"]) for r in rounds] == [(0, "iterate"), (1, "accept")]


def test_max_iterations_must_be_positive():
    with pytest.raises(ValueError):
        Boss(max_iterations=0)
