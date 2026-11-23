"""The manual agent loop driven by a scripted fake of ``client.responses.create`` (no network)."""

import json
from typing import Any

import httpx
import openai
import pytest

from app.generation.agentic.agent_loop import (
    FINAL_ANSWER_FORMAT,
    NO_REASONING_SUMMARY,
    AgentLoopError,
    build_system_prompt,
    format_trace,
    run_agent,
)
from app.generation.agentic.agent_schemas import (
    AgentAction,
    AgentEstimate,
    AgentResult,
    AgentStep,
    AgentWarning,
    Coverage,
    SearchBudgetsArgs,
    TokenUsage,
)
from app.generation.agentic.agent_tools import TOOL_SCHEMAS, ToolRegistry, build_tool_registry
from tests.agent_helpers import (
    FakeAsyncOpenAI,
    calc_args,
    final_response,
    function_call,
    make_response,
    outputs_of,
    reasoning_item,
    reference,
    valid_estimate,
)

pytestmark = pytest.mark.anyio

SEARCH_ARGS = {"query": "payment gateway integration", "filters": None}


def search(call_id: str = "s") -> Any:
    return function_call("search_budgets", SEARCH_ARGS, call_id)


def calc(call_id: str = "c", args: dict[str, Any] | None = None) -> Any:
    return function_call("calculate_estimate", args or calc_args(), call_id)


def turn(response_id: str, *items: Any) -> Any:
    return make_response(response_id, list(items))


async def backend(args: SearchBudgetsArgs) -> list[dict[str, Any]]:
    return [
        {
            "id": 1,
            "budget_id": "B-1",
            "sector": "finance",
            "project_year": 2023,
            "content_preview": "Payment gateway",
            "estimated_hours": 100.0,
            "distance": 0.2,
        }
    ]


def registry() -> ToolRegistry:
    return build_tool_registry(backend)


async def run(client: FakeAsyncOpenAI, *, max_iterations: int = 10, **overrides: Any) -> AgentResult:
    return await run_agent(
        "transcript text",
        client=client,
        registry=overrides.pop("registry", registry()),
        model="gpt-5-mini",
        effort="minimal",
        max_iterations=max_iterations,
        **overrides,
    )


async def test_a_single_tool_call_then_the_final_answer() -> None:
    client = FakeAsyncOpenAI(
        [
            make_response(
                "resp_1",
                [reasoning_item("Need evidence."), function_call("search_budgets", SEARCH_ARGS, "c1")],
            ),
            turn("resp_2", calc()),
            final_response("resp_3", valid_estimate()),
        ]
    )

    result = await run(client)

    assert result.status == "done" and result.iterations == 3
    assert isinstance(result.estimate, AgentEstimate) and result.estimate.total_hours == 115.0
    first, _, second = result.trace
    assert first.reasoning == "Need evidence."
    assert [a.tool for a in first.actions] == ["search_budgets"]
    assert first.actions[0].observation["count"] == 1 and first.actions[0].error is None
    assert second.actions == []  # the final answer is a step without actions


async def test_the_first_request_sends_the_transcript_instructions_tools_and_the_format() -> None:
    client = FakeAsyncOpenAI([final_response("resp_1", valid_estimate(ids=()))])

    await run(client)

    call = client.calls[0]
    assert call["model"] == "gpt-5-mini"
    assert call["input"] == [{"role": "user", "content": "transcript text"}]
    assert call["instructions"] == build_system_prompt(10)
    assert call["tools"] == TOOL_SCHEMAS
    assert call["reasoning"] == {"effort": "minimal", "summary": "auto"}
    assert call["text"] == {"format": FINAL_ANSWER_FORMAT}
    assert FINAL_ANSWER_FORMAT["strict"] is True
    assert "previous_response_id" not in call
    assert "store" not in call  # chaining with previous_response_id needs the default store=True


async def test_the_instructions_tell_the_model_its_turn_budget() -> None:
    client = FakeAsyncOpenAI([final_response("resp_1", valid_estimate(ids=()))])

    await run(client, max_iterations=7)

    instructions = client.calls[0]["instructions"]
    assert instructions == build_system_prompt(7)
    assert "at most 7 turns" in instructions


def test_the_prompt_asks_for_task_level_search_and_a_stop_rule_without_fixing_search_counts() -> None:
    prompt = build_system_prompt(10)

    assert "functional level" in prompt and "16 to 72" in prompt
    assert "two rephrasings" in prompt and "move on" in prompt  # convergence rule
    assert "the same turn" in prompt  # batch independent searches and rephrasings
    # Deliberately non-directive: no fixed number of searches and no component names
    assert not any(word in prompt.lower() for word in ("sap", "backend", "mobile", "exactly"))


async def test_custom_instructions_override_the_default_prompt() -> None:
    client = FakeAsyncOpenAI([final_response("resp_1", valid_estimate(ids=()))])

    await run(client, instructions="custom")

    assert client.calls[0]["instructions"] == "custom"


async def test_parallel_calls_run_in_one_turn_and_return_in_one_request() -> None:
    client = FakeAsyncOpenAI(
        [
            make_response(
                "resp_1",
                [
                    function_call("search_budgets", SEARCH_ARGS, "call_A"),
                    function_call("search_budgets", {"query": "mobile offline sync app"}, "call_B"),
                ],
            ),
            turn("resp_2", calc()),
            final_response("resp_3", valid_estimate()),
        ]
    )

    result = await run(client)

    assert len(client.calls) == 3  # ONE continuation for both calls (+ calculation + answer)
    sent = outputs_of(client.calls[1])
    assert [item["call_id"] for item in sent] == ["call_A", "call_B"]
    assert all(json.loads(item["output"])["count"] == 1 for item in sent)
    assert [a.call_id for a in result.trace[0].actions] == ["call_A", "call_B"]


async def test_every_turn_chains_previous_response_id_and_resends_instructions_and_tools() -> None:
    client = FakeAsyncOpenAI(
        [
            make_response("resp_1", [function_call("search_budgets", SEARCH_ARGS, "c1")]),
            make_response("resp_2", [function_call("search_budgets", SEARCH_ARGS, "c2")]),
            turn("resp_3", calc()),
            final_response("resp_4", valid_estimate()),
        ]
    )

    await run(client)

    assert [c.get("previous_response_id") for c in client.calls] == [
        None,
        "resp_1",
        "resp_2",
        "resp_3",
    ]
    for call in client.calls:
        assert call["instructions"] == build_system_prompt(10) and call["tools"] == TOOL_SCHEMAS
    # Continuations carry only the new items (outputs + turn notice), never the transcript again
    continuation = client.calls[1]["input"]
    assert [item.get("type") for item in continuation[:-1]] == ["function_call_output"]
    assert continuation[-1]["role"] == "user" and "Turn 2 of 10" in continuation[-1]["content"]


ALLOWED_AT_CLOSE = {
    "type": "allowed_tools",
    "mode": "auto",
    "tools": [
        {"type": "function", "name": "calculate_estimate"},
        {"type": "function", "name": "validate_estimate"},
    ],
}


async def test_only_the_closing_turns_constrain_the_tools() -> None:
    script = [
        make_response(f"resp_{n}", [function_call("search_budgets", SEARCH_ARGS, f"c{n}")])
        for n in range(1, 10)
    ] + [final_response("resp_10", valid_estimate())]
    client = FakeAsyncOpenAI(script)

    result = await run(client, max_iterations=10)

    choices = [call.get("tool_choice") for call in client.calls]
    # Turns 1-7 search freely, 8-9 may only calculate/validate, 10 has no tools
    assert choices == [None] * 7 + [ALLOWED_AT_CLOSE, ALLOWED_AT_CLOSE, "none"]
    assert all(c["tools"] == TOOL_SCHEMAS for c in client.calls)  # list unchanged (prompt cache)
    assert result.status == "done"


async def test_the_turn_notice_tells_the_phase_of_the_next_turn() -> None:
    script = [
        make_response(f"resp_{n}", [function_call("search_budgets", SEARCH_ARGS, f"c{n}")])
        for n in range(1, 10)
    ] + [final_response("resp_10", valid_estimate())]
    client = FakeAsyncOpenAI(script)

    await run(client, max_iterations=10)

    notices = [call["input"][-1]["content"] for call in client.calls[1:]]
    assert "Searching is possible until turn 7" in notices[0]
    assert "Searching is closed" in notices[7]  # notice for turn 9
    assert "LAST turn" in notices[8] and "Turn 10 of 10" in notices[8]


@pytest.mark.parametrize(
    ("max_iterations", "expected"),
    [
        (1, [None]),
        (2, [None, None]),
        (3, [None, None, "none"]),  # 3 // 3 = 1 reserved turn
        (6, [None] * 4 + [ALLOWED_AT_CLOSE, "none"]),  # 6 // 3 = 2 reserved turns
    ],
)
async def test_short_runs_keep_their_search_turns(
    max_iterations: int, expected: list[Any]
) -> None:
    script = [
        make_response(f"resp_{n}", [function_call("search_budgets", SEARCH_ARGS, f"c{n}")])
        for n in range(1, max_iterations + 1)
    ]
    client = FakeAsyncOpenAI(script)

    result = await run(client, max_iterations=max_iterations)

    assert [call.get("tool_choice") for call in client.calls] == expected
    assert result.status == "max_steps_exceeded"  # the fake keeps calling tools regardless


def test_the_prompt_announces_the_reserved_turns_only_when_there_are_any() -> None:
    assert "first 7 turns" in build_system_prompt(10) and "no tools at all" in build_system_prompt(10)
    assert "reserved" not in build_system_prompt(2)


async def test_a_failing_tool_is_an_observation_and_the_loop_continues() -> None:
    async def broken_backend(args: SearchBudgetsArgs) -> list[dict[str, Any]]:
        raise RuntimeError("database is down")

    client = FakeAsyncOpenAI(
        [
            make_response(
                "resp_1",
                [
                    function_call("search_budgets", SEARCH_ARGS, "c1"),
                    function_call("search_budgets", {"query": "short"}, "c2"),
                    function_call("search_budgets", "{broken json", "c3"),
                ],
            ),
            final_response("resp_2", valid_estimate(ids=())),
        ]
    )

    result = await run(client, registry=build_tool_registry(broken_backend))

    assert result.status == "done"
    errors = [a.error for a in result.trace[0].actions]
    assert "database is down" in (errors[0] or "")
    assert "at least 10" in (errors[1] or "")
    assert "not valid JSON" in (errors[2] or "")
    sent = {item["call_id"]: json.loads(item["output"]) for item in outputs_of(client.calls[1])}
    assert set(sent) == {"c1", "c2", "c3"} and all("error" in output for output in sent.values())


async def test_an_unknown_tool_is_reported_back_to_the_model() -> None:
    client = FakeAsyncOpenAI(
        [
            make_response("resp_1", [function_call("send_email", {}, "c1")]),
            final_response("resp_2", valid_estimate(ids=())),
        ]
    )

    result = await run(client)

    assert result.status == "done"
    assert "Unknown tool 'send_email'" in (result.trace[0].actions[0].error or "")


async def test_exhausting_the_iterations_stops_with_a_partial_trace_and_no_extra_calls() -> None:
    client = FakeAsyncOpenAI(
        [
            make_response(f"resp_{n}", [function_call("search_budgets", SEARCH_ARGS, f"c{n}")])
            for n in range(1, 6)
        ]
    )

    result = await run(client, max_iterations=3)

    assert result.status == "max_steps_exceeded" and result.estimate is None
    assert result.iterations == 3 and len(client.calls) == 3
    assert [step.step for step in result.trace] == [1, 2, 3]
    assert result.error and "3 iterations" in result.error


async def test_an_invalid_final_answer_gets_one_correction_then_succeeds() -> None:
    client = FakeAsyncOpenAI(
        [
            final_response("resp_1", {**valid_estimate(120, ids=()), "total_hours": 999}),
            final_response("resp_2", valid_estimate(120, ids=())),
        ]
    )

    result = await run(client)

    assert result.status == "done" and result.iterations == 2  # the correction is an iteration
    assert "total_hours" in (result.trace[0].validation_error or "")
    correction = client.calls[1]
    assert correction["previous_response_id"] == "resp_1"
    assert correction["input"][0]["role"] == "user" and "invalid" in correction["input"][0]["content"]


async def test_a_second_invalid_final_answer_ends_in_error() -> None:
    client = FakeAsyncOpenAI(
        [final_response("resp_1", "not json at all"), final_response("resp_2", "{}")]
    )

    result = await run(client)

    assert result.status == "error" and result.estimate is None
    assert len(client.calls) == 2 and len(result.trace) == 2  # exactly one correction turn
    assert result.error


async def test_missing_reasoning_summaries_do_not_break_the_run() -> None:
    client = FakeAsyncOpenAI(
        [
            make_response("resp_1", [calc("c1")]),
            make_response("resp_2", [], output_text=json.dumps(valid_estimate(ids=()))),
        ]
    )

    result = await run(client)

    assert result.status == "done"
    assert [step.reasoning for step in result.trace] == [NO_REASONING_SUMMARY] * 2


async def test_usage_is_recorded_per_step_and_aggregated() -> None:
    client = FakeAsyncOpenAI(
        [
            make_response(
                "resp_1",
                [calc("c1")],
                input_tokens=100,
                output_tokens=20,
                reasoning_tokens=8,
            ),
            final_response("resp_2", valid_estimate(ids=())),
        ]
    )

    result = await run(client)

    assert result.trace[0].usage == TokenUsage(
        input_tokens=100, output_tokens=20, reasoning_tokens=8, total_tokens=120
    )
    assert result.usage.total_tokens == 120 + 15 and result.usage.input_tokens == 110


async def test_reasoning_summary_can_be_disabled() -> None:
    client = FakeAsyncOpenAI([final_response("resp_1", valid_estimate(ids=()))])

    await run(client, reasoning_summary=None)

    assert client.calls[0]["reasoning"] == {"effort": "minimal"}


async def test_an_openai_failure_raises_with_the_steps_completed_so_far() -> None:
    api_error = openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com"))
    client = FakeAsyncOpenAI(
        [make_response("resp_1", [function_call("search_budgets", SEARCH_ARGS, "c1")]), api_error]
    )

    with pytest.raises(AgentLoopError) as caught:
        await run(client)

    assert len(caught.value.trace) == 1


# --- format_trace -----------------------------------------------------------------------------------


def make_result() -> AgentResult:
    search = AgentAction(
        tool="search_budgets",
        call_id="c1",
        arguments={"query": "payment gateway", "filters": {"sectors": ["finance"]}},
        observation={"summary": "2 historical items; hours 100-300 (median 200)"},
        duration_ms=12,
    )
    calc = AgentAction(
        tool="calculate_estimate",
        call_id="c2",
        arguments={"components": []},
        observation={"error": "Invalid arguments"},
        duration_ms=1,
        error="Invalid arguments",
    )
    return AgentResult(
        status="done",
        estimate=AgentEstimate.model_validate(valid_estimate()),
        trace=[
            AgentStep(step=1, reasoning="Look for evidence.\nStart with payments.", actions=[search, calc]),
            AgentStep(step=2, reasoning="All set."),
        ],
        iterations=2,
        usage=TokenUsage(input_tokens=10, output_tokens=5, reasoning_tokens=2, total_tokens=15),
    )


def test_format_trace_prints_steps_with_reasoning_action_and_observation() -> None:
    text = format_trace(make_result())

    assert text.startswith(
        "STEP 1\n"
        "  reasoning:   Look for evidence.\n"
        "               Start with payments.\n"
        '  action:      search_budgets(query="payment gateway", filters={"sectors": ["finance"]})\n'
        "  observation: 2 historical items; hours 100-300 (median 200)\n"
        "  action:      calculate_estimate(components=[])\n"
        "  observation: ERROR: Invalid arguments\n"
        "\n"
        "STEP 2\n"
        "  reasoning:   All set.\n"
        "  action:      (none: final answer)\n"
        "  observation: final answer produced\n"
        "\n"
        "RESULT\n"
        "  status:      done\n"
        "  iterations:  2\n"
        "  tokens:      input=10 output=5 reasoning=2 total=15\n"
        "  total hours: 115\n"
        "  component:   Backend: 115 h (refs: B-1)\n"
        "  notes:       grounded on B-1\n"
    )


# --- Freshness invariant: the answer must rest on a calculation made after the last search ----------


def codes_of(warnings: list[Any]) -> list[str]:
    return [warning.code for warning in warnings]


async def test_a_search_after_the_last_calculation_gets_a_corrective_turn_and_a_recalculation() -> (
    None
):
    client = FakeAsyncOpenAI(
        [
            turn("r1", search("s1")),
            turn("r2", calc("c1")),
            turn("r3", search("s2")),  # new evidence after the calculation
            final_response("r4", valid_estimate()),  # answers with the stale calculation
            turn("r5", calc("c2")),  # recalculation demanded by the correction
            final_response("r6", valid_estimate()),
        ]
    )

    result = await run(client)

    assert result.status == "done" and result.iterations == 6 and len(client.calls) == 6
    rejected = result.trace[3]
    assert "New evidence arrived after your last calculation" in (rejected.validation_error or "")
    correction = client.calls[4]
    assert correction["previous_response_id"] == "r4"
    content = correction["input"][0]["content"]
    assert "calculate_estimate" in content and "Turn 5 of 10" in content
    assert result.warnings == []  # fixed, nothing left to warn about


async def test_without_a_later_search_the_loop_does_not_intervene() -> None:
    client = FakeAsyncOpenAI(
        [turn("r1", search()), turn("r2", calc()), final_response("r3", valid_estimate())]
    )

    result = await run(client)

    assert result.status == "done" and len(client.calls) == 3
    assert all(step.validation_error is None and not step.warnings for step in result.trace)
    assert result.warnings == []


async def test_a_search_in_the_same_turn_as_the_calculation_still_counts_as_unseen() -> None:
    client = FakeAsyncOpenAI(
        [
            turn("r1", search("s1"), calc("c1")),  # parallel: the calculation cannot see the search
            final_response("r2", valid_estimate()),
            turn("r3", calc("c2")),
            final_response("r4", valid_estimate()),
        ]
    )

    result = await run(client)

    assert result.status == "done" and result.iterations == 4
    assert result.trace[1].validation_error


async def _empty_backend(args: SearchBudgetsArgs) -> list[dict[str, Any]]:
    return []


async def test_a_search_without_items_is_not_new_evidence() -> None:
    client = FakeAsyncOpenAI(
        [
            turn("r1", calc("c1")),
            turn("r2", search("s1")),
            final_response("r3", valid_estimate(ids=())),
        ]
    )

    result = await run(client, registry=build_tool_registry(_empty_backend))

    assert result.status == "done" and len(client.calls) == 3  # no corrective turn
    assert "stale_calculation" not in codes_of(result.warnings)


async def test_a_stale_answer_on_the_last_turn_is_accepted_with_a_warning_and_no_extra_turn() -> (
    None
):
    # max_iterations=4 -> the 4th turn has tool_choice="none": no recalculation is possible
    client = FakeAsyncOpenAI(
        [
            turn("r1", search("s1")),
            turn("r2", calc("c1")),
            turn("r3", search("s2")),
            final_response("r4", valid_estimate()),
        ]
    )

    result = await run(client, max_iterations=4)

    assert client.calls[3]["tool_choice"] == "none"
    assert result.status == "done" and result.iterations == 4 and len(client.calls) == 4
    assert codes_of(result.warnings) == ["stale_calculation"]
    assert codes_of(result.trace[-1].warnings) == ["stale_calculation"]
    assert "stale_calculation" in format_trace(result)


async def test_no_corrective_turn_when_the_only_turn_left_cannot_calculate() -> None:
    # Answer at turn 3 of 4: the 4th turn has no tools, so a recalculation would be impossible
    client = FakeAsyncOpenAI(
        [turn("r1", calc("c1")), turn("r2", search("s1")), final_response("r3", valid_estimate())]
    )

    result = await run(client, max_iterations=4)

    assert len(client.calls) == 3 and result.status == "done"
    assert codes_of(result.warnings) == ["stale_calculation"]


async def test_a_correction_that_can_recalculate_is_given_while_calculate_is_still_allowed() -> (
    None
):
    # max_iterations=6: turn 5 allows only calculate/validate, turn 6 has no tools
    client = FakeAsyncOpenAI(
        [
            turn("r1", search("s1")),
            turn("r2", calc("c1")),
            turn("r3", search("s2")),
            final_response("r4", valid_estimate()),  # turn 4: turn 5 can still recalculate
            turn("r5", calc("c2")),
            final_response("r6", valid_estimate()),
        ]
    )

    result = await run(client, max_iterations=6)

    assert result.status == "done" and result.iterations == 6 and result.warnings == []
    assert client.calls[4]["tool_choice"] != "none"


async def test_the_corrections_are_bounded_and_never_exceed_max_iterations() -> None:
    # The model ignores every correction: after MAX_ANSWER_CORRECTIONS the answer is accepted
    script: list[Any] = [turn("r1", search("s1")), turn("r2", calc("c1")), turn("r3", search("s2"))]
    script += [final_response(f"a{n}", valid_estimate()) for n in range(1, 4)]
    client = FakeAsyncOpenAI(script)

    result = await run(client, max_iterations=10)

    assert result.status == "done" and len(client.calls) == 6
    assert codes_of(result.warnings) == ["stale_calculation"]


async def test_validate_with_a_stale_calculation_adds_a_warning_instead_of_failing() -> None:
    validate = function_call(
        "validate_estimate",
        {
            "components": [
                {
                    "name": "Backend",
                    "tasks": [
                        {
                            "name": "Payments",
                            "estimated_hours": 115.0,
                            "references": [reference("B-1", 100.0)],
                        }
                    ],
                }
            ],
            "total_hours": 115.0,
        },
        "v1",
    )
    client = FakeAsyncOpenAI(
        [
            turn("r1", calc("c1")),
            turn("r2", search("s1")),
            turn("r3", validate),
            turn("r4", calc("c2")),
            turn("r5", validate),
            final_response("r6", valid_estimate()),
        ]
    )

    result = await run(client)

    stale, fresh = result.trace[2].actions[0], result.trace[4].actions[0]
    assert stale.error is None and stale.observation["ok"] is True
    assert [w["code"] for w in stale.observation["warnings"]] == ["stale_calculation"]
    assert "stale_calculation" in stale.observation["summary"]
    sent = json.loads(outputs_of(client.calls[3])[0]["output"])  # what the model read
    assert [w["code"] for w in sent["warnings"]] == ["stale_calculation"]
    assert fresh.observation["warnings"] == []
    assert result.status == "done" and result.warnings == []


# --- Reference ids and total checked against the run --------------------------------------------------


async def test_unknown_reference_ids_get_a_corrective_turn() -> None:
    client = FakeAsyncOpenAI(
        [
            turn("r1", search()),
            turn("r2", calc()),
            final_response("r3", valid_estimate(ids=("B-1", "X-99"))),
            final_response("r4", valid_estimate()),
        ]
    )

    result = await run(client)

    assert result.status == "done" and len(client.calls) == 4
    assert "X-99" in (result.trace[2].validation_error or "")
    assert "X-99" in client.calls[3]["input"][0]["content"]
    assert result.warnings == [] and result.estimate is not None
    assert result.estimate.components[0].reference_budget_ids == ["B-1"]


async def test_unknown_ids_that_persist_are_accepted_with_a_warning_after_the_corrections() -> None:
    bad = valid_estimate(ids=("X-99",))
    client = FakeAsyncOpenAI(
        [turn("r1", search()), turn("r2", calc())]
        + [final_response(f"a{n}", bad) for n in range(1, 4)]
    )

    result = await run(client)

    assert result.status == "done" and len(client.calls) == 5
    assert "unknown_reference_ids" in codes_of(result.warnings)


async def test_ids_never_returned_by_a_search_are_unknown_even_if_the_calculation_used_them() -> None:
    invented = calc_args({"name": "Payments", "references": [reference("INV-1", 100.0)]})
    client = FakeAsyncOpenAI(
        [
            turn("r1", search()),
            turn("r2", calc("c1", invented)),
            final_response("r3", valid_estimate(ids=("INV-1",))),
            final_response("r4", valid_estimate(ids=())),
        ]
    )

    result = await run(client)

    assert "INV-1" in (result.trace[2].validation_error or "")
    assert result.status == "done"


async def test_duplicate_ids_inside_a_component_are_corrected() -> None:
    client = FakeAsyncOpenAI(
        [
            turn("r1", search()),
            turn("r2", calc()),
            final_response("r3", valid_estimate(ids=("B-1", "B-1"))),
            final_response("r4", valid_estimate()),
        ]
    )

    result = await run(client)

    assert "repeats reference ids" in (result.trace[2].validation_error or "")
    assert result.status == "done" and result.warnings == []


async def _two_items_backend(args: SearchBudgetsArgs) -> list[dict[str, Any]]:
    return [
        {"id": n, "budget_id": f"B-{n}", "estimated_hours": 100.0 * n, "distance": 0.1 * n}
        for n in (1, 2)
    ]


async def test_ids_that_differ_from_the_calculation_are_only_a_trace_warning() -> None:
    client = FakeAsyncOpenAI(
        [
            turn("r1", search()),
            turn("r2", calc()),  # the calculation cites B-1 only
            final_response("r3", valid_estimate(ids=("B-1", "B-2"))),  # B-2 exists, not calculated
        ]
    )

    result = await run(client, registry=build_tool_registry(_two_items_backend))

    assert result.status == "done" and len(client.calls) == 3  # no corrective turn
    assert codes_of(result.warnings) == ["reference_ids_differ"]
    assert "reference_ids_differ" in format_trace(result)


async def test_a_total_that_differs_from_the_last_calculation_is_corrected() -> None:
    client = FakeAsyncOpenAI(
        [
            turn("r1", search()),
            turn("r2", calc()),  # 115 h
            final_response("r3", valid_estimate(130)),
            final_response("r4", valid_estimate(115)),
        ]
    )

    result = await run(client)

    assert result.status == "done" and result.estimate is not None
    assert result.estimate.total_hours == 115.0
    assert "differs from your last calculation" in (result.trace[2].validation_error or "")


async def test_a_total_mismatch_on_the_last_turn_is_accepted_with_a_warning() -> None:
    client = FakeAsyncOpenAI(
        [turn("r1", calc()), final_response("r2", valid_estimate(130, ids=()))]
    )

    result = await run(client, max_iterations=2)

    assert result.status == "done" and len(client.calls) == 2
    assert "total_differs_from_calculation" in codes_of(result.warnings)


# --- Coverage computed by the system -------------------------------------------------------------------------


async def test_coverage_and_unbudgeted_tasks_are_computed_from_the_last_calculation() -> None:
    mixed = calc_args(
        {"name": "Payments", "references": [reference("B-1", 100.0)]},
        {"name": "Offline mode", "references": []},
        {"name": "Tariffs", "references": []},
    )
    client = FakeAsyncOpenAI(
        [turn("r1", search()), turn("r2", calc("c1", mixed)), final_response("r3", valid_estimate())]
    )

    result = await run(client)

    assert result.coverage is not None
    assert (result.coverage.budgeted_tasks, result.coverage.total_tasks) == (1, 3)
    assert result.coverage.ratio == pytest.approx(0.333, abs=0.001)
    assert result.unbudgeted_tasks == ["Backend / Offline mode", "Backend / Tariffs"]
    assert result.total_is_lower_bound is True
    # The model-filled estimate is untouched: coverage never goes into AgentEstimate
    assert set(AgentEstimate.model_fields) == {"components", "total_hours", "notes"}


async def test_full_coverage_is_not_a_lower_bound_and_no_calculation_means_no_coverage() -> None:
    full = await run(
        FakeAsyncOpenAI(
            [turn("r1", search()), turn("r2", calc()), final_response("r3", valid_estimate())]
        )
    )
    none = await run(FakeAsyncOpenAI([final_response("r1", valid_estimate(ids=()))]))

    assert full.coverage is not None and full.coverage.ratio == 1.0
    assert full.unbudgeted_tasks == [] and full.total_is_lower_bound is False
    assert none.coverage is None and none.total_is_lower_bound is False


# --- Trace: nothing important is cut, cuts are marked, coverage and search items are auditable --------------------


def action(tool: str, arguments: dict[str, Any], observation: dict[str, Any]) -> AgentAction:
    return AgentAction(
        tool=tool, call_id="x", arguments=arguments, observation=observation, duration_ms=1
    )


def trace_of(*actions: AgentAction) -> str:
    result = AgentResult(
        status="done",
        trace=[AgentStep(step=1, reasoning="r", actions=list(actions))],
        iterations=1,
    )
    return format_trace(result)


def test_calculate_and_validate_actions_are_never_truncated() -> None:
    tasks = [
        {"name": f"Task number {n}", "references": [reference(f"BUD-{n:04d}", 40.0 + n)] * 2}
        for n in range(40)
    ]
    calc_action = action("calculate_estimate", calc_args(*tasks), {"summary": "ok"})
    validate_action = action(
        "validate_estimate",
        {
            "components": [
                {"name": "Backend", "tasks": [{**task, "estimated_hours": 50.5} for task in tasks]}
            ],
            "total_hours": 2020.0,
        },
        {"summary": "0 issues, 0 warnings"},
    )

    text = trace_of(calc_action, validate_action)

    assert "truncated" not in text and "..." not in text
    for n in range(40):
        assert f"BUD-{n:04d}=" in text and f"Task number {n}" in text
    assert "total_hours: 2020.0" in text and "-> 50.5 h" in text
    assert 'calculate_estimate(\n' in text and '  component "Backend":' in text


def test_other_long_actions_are_cut_at_600_characters_with_an_explicit_mark() -> None:
    long_query = "payment " * 200
    text = trace_of(
        action("search_budgets", {"query": long_query, "filters": None}, {"summary": "s"})
    )

    line = next(line for line in text.splitlines() if "search_budgets(" in line)
    assert line.endswith("…[truncated]")
    assert len(line.split("action:", 1)[1].strip()) == 600 + len("…[truncated]")

    fits = trace_of(action("search_budgets", {"query": "x" * 400, "filters": None}, {"summary": "s"}))
    assert "truncated" not in fits  # it used to be cut at 300 characters


def test_search_observations_list_the_three_closest_items_for_auditing() -> None:
    items = [
        {"budget_id": "B-FAR", "estimated_hours": 90.0, "distance": 0.5},
        {"budget_id": "B-NEAR", "estimated_hours": 30.0, "distance": 0.1},
        {"budget_id": "B-MID", "estimated_hours": 45.5, "distance": 0.3},
        {"budget_id": "B-MID2", "estimated_hours": 60.0, "distance": 0.4},
    ]
    obs = {"summary": "4 historical items", "items": items, "count": 4}

    text = trace_of(action("search_budgets", {"query": "q" * 12, "filters": None}, obs))

    assert "closest: B-NEAR 30 h (d=0.1); B-MID 45.5 h (d=0.3); B-MID2 60 h (d=0.4)" in text
    assert "B-FAR" not in text  # only the three closest keep the trace readable


def test_the_result_block_shows_coverage_unbudgeted_tasks_and_warnings() -> None:
    result = make_result()
    result.coverage = Coverage(budgeted_tasks=11, total_tasks=21, ratio=0.524)
    result.unbudgeted_tasks = ["Backend / Tariffs", "Mobile / Offline mode"]
    result.total_is_lower_bound = True
    result.warnings = [AgentWarning(code="stale_calculation", message="search after calculation")]

    text = format_trace(result)

    assert "coverage:    11 of 21 tasks have historical references (52%)" in text
    assert "lower bound: total is a lower bound: yes" in text
    assert "unbudgeted:  - Backend / Tariffs\n               - Mobile / Offline mode" in text
    assert "warning:     stale_calculation: search after calculation" in text


def test_validation_findings_other_than_no_reference_are_visible_in_the_observation() -> None:
    obs = {
        "summary": "1 issue, 2 warnings: out_of_range×1, no_reference×1, stale_calculation×1",
        "issues": [{"where": "Backend / A", "code": "out_of_range", "message": "too high"}],
        "warnings": [
            {"where": "Backend / B", "code": "no_reference", "message": "unbudgeted"},
            {"where": None, "code": "stale_calculation", "message": "recalculate"},
        ],
    }

    text = trace_of(action("validate_estimate", {"components": []}, obs))

    assert "issue out_of_range (Backend / A): too high" in text
    assert "warning stale_calculation: recalculate" in text
    assert "Backend / B" not in text  # no_reference stays in the summary count only


def test_the_prompt_explains_freshness_references_and_warnings_without_fixed_counts() -> None:
    prompt = build_system_prompt(10)

    assert "after your last search" in prompt and "calculate again" in prompt
    assert "grouped round" in prompt
    assert "`warnings` are informative" in prompt and "unbudgeted" in prompt
    assert "cite only ids that a search returned" in prompt
