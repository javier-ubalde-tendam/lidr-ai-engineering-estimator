import pytest

from app.guardrails.input import check_input
from app.sessions.compression.anchors import AnchorDetector
from app.sessions.models import Message, ProjectMetadata
from app.sessions.tier_resolver import Tier, resolve_tier
from evals.stress.scenarios import SCENARIOS, Scenario

# El heurístico de anclas es la única pieza que el escenario `contradiction` activa a propósito
_DETECTOR = AnchorDetector(mode="heuristic", model="unused")


def _is_anchor(text: str) -> bool:
    return _DETECTOR.detect(Message(role="user", content=text)).is_anchor


def test_the_three_scenarios_exist():
    assert set(SCENARIOS) == {"growing", "pivot", "contradiction"}


@pytest.mark.parametrize("scenario", SCENARIOS.values(), ids=SCENARIOS.keys())
def test_each_scenario_has_at_least_twenty_valid_turns(scenario: Scenario):
    assert len(scenario.turns) >= 20
    # Form(min_length=20) del endpoint
    assert all(len(turn) >= 20 for turn in scenario.turns)
    assert len(set(scenario.turns)) == len(scenario.turns)


@pytest.mark.parametrize("scenario", SCENARIOS.values(), ids=SCENARIOS.keys())
def test_turns_pass_the_input_guardrails_and_do_not_force_a_tier(scenario: Scenario):
    for turn in scenario.turns:
        check_input(turn)  # sin cliente: solo regex de prompt injection y PII
        assert resolve_tier(transcript=turn, metadata=ProjectMetadata(), override=None)[0] == Tier.DEFAULT


@pytest.mark.parametrize("scenario", SCENARIOS.values(), ids=SCENARIOS.keys())
def test_facts_are_literally_present_in_the_turn_that_declares_them(scenario: Scenario):
    assert scenario.facts
    for fact in scenario.facts:
        assert fact.text.lower() in scenario.turns[fact.declared_at - 1].lower(), fact


@pytest.mark.parametrize("scenario", SCENARIOS.values(), ids=SCENARIOS.keys())
def test_no_fact_is_written_with_thousands_separators(scenario: Scenario):
    assert all("," not in fact.text and "." not in fact.text for fact in scenario.facts)


def test_only_the_contradiction_scenario_creates_anchors_and_only_with_the_first_budget():
    anchored = {name: [i + 1 for i, t in enumerate(s.turns) if _is_anchor(t)] for name, s in SCENARIOS.items()}

    assert anchored == {"growing": [], "pivot": [], "contradiction": [2]}


def test_pivot_replaces_the_stack_around_turn_five():
    pivot = SCENARIOS["pivot"]
    django = next(f for f in pivot.facts if f.text == "Django")
    nestjs = next(f for f in pivot.facts if f.text == "NestJS")

    assert django.superseded_at == nestjs.declared_at == 5
    assert "Django" in pivot.turns[1] and "NestJS" in pivot.turns[4]


def test_contradiction_declares_30000_then_80000_around_turn_eight():
    contradiction = SCENARIOS["contradiction"]

    assert contradiction.turns[1].count("30000") == 1
    assert "80000" in contradiction.turns[7]
    assert "30000" not in contradiction.turns[7]  # la contradicción no menciona el valor antiguo


def test_expected_and_stale_facts_follow_the_declaration_and_supersession_turns():
    contradiction = SCENARIOS["contradiction"]

    assert contradiction.expected_facts(1) == ["Harvest Hub"]
    assert contradiction.expected_facts(7) == ["Harvest Hub", "30000"]
    assert contradiction.expected_facts(8) == ["Harvest Hub", "80000"]
    assert contradiction.stale_facts(7) == []
    assert contradiction.stale_facts(8) == ["30000"]
