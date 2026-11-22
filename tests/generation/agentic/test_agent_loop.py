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
    SearchBudgetsArgs,
    TokenUsage,
)
from app.generation.agentic.agent_tools import TOOL_SCHEMAS, ToolRegistry, build_tool_registry
from tests.agent_helpers import (
    FakeAsyncOpenAI,
    final_response,
    function_call,
    make_response,
    outputs_of,
    reasoning_item,
    valid_estimate,
)

pytestmark = pytest.mark.anyio

SEARCH_ARGS = {"query": "payment gateway integration", "filters": None}


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
            final_response("resp_2", valid_estimate()),
        ]
    )

    result = await run(client)

    assert result.status == "done" and result.iterations == 2
    assert isinstance(result.estimate, AgentEstimate) and result.estimate.total_hours == 120.0
    first, second = result.trace
    assert first.reasoning == "Need evidence."
    assert [a.tool for a in first.actions] == ["search_budgets"]
    assert first.actions[0].observation["count"] == 1 and first.actions[0].error is None
    assert second.actions == []  # the final answer is a step without actions


async def test_the_first_request_sends_the_transcript_instructions_tools_and_the_format() -> None:
    client = FakeAsyncOpenAI([final_response("resp_1", valid_estimate())])

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
    client = FakeAsyncOpenAI([final_response("resp_1", valid_estimate())])

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
    client = FakeAsyncOpenAI([final_response("resp_1", valid_estimate())])

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
            final_response("resp_2", valid_estimate()),
        ]
    )

    result = await run(client)

    assert len(client.calls) == 2  # ONE continuation for both calls
    sent = outputs_of(client.calls[1])
    assert [item["call_id"] for item in sent] == ["call_A", "call_B"]
    assert all(json.loads(item["output"])["count"] == 1 for item in sent)
    assert [a.call_id for a in result.trace[0].actions] == ["call_A", "call_B"]


async def test_every_turn_chains_previous_response_id_and_resends_instructions_and_tools() -> None:
    client = FakeAsyncOpenAI(
        [
            make_response("resp_1", [function_call("search_budgets", SEARCH_ARGS, "c1")]),
            make_response("resp_2", [function_call("search_budgets", SEARCH_ARGS, "c2")]),
            final_response("resp_3", valid_estimate()),
        ]
    )

    await run(client)

    assert [c.get("previous_response_id") for c in client.calls] == [None, "resp_1", "resp_2"]
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
            final_response("resp_2", valid_estimate()),
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
            final_response("resp_2", valid_estimate()),
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
            final_response("resp_1", {**valid_estimate(120), "total_hours": 999}),
            final_response("resp_2", valid_estimate(120)),
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
            make_response("resp_1", [function_call("search_budgets", SEARCH_ARGS, "c1")]),
            make_response("resp_2", [], output_text=json.dumps(valid_estimate())),
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
                [function_call("search_budgets", SEARCH_ARGS, "c1")],
                input_tokens=100,
                output_tokens=20,
                reasoning_tokens=8,
            ),
            final_response("resp_2", valid_estimate()),
        ]
    )

    result = await run(client)

    assert result.trace[0].usage == TokenUsage(
        input_tokens=100, output_tokens=20, reasoning_tokens=8, total_tokens=120
    )
    assert result.usage.total_tokens == 120 + 15 and result.usage.input_tokens == 110


async def test_reasoning_summary_can_be_disabled() -> None:
    client = FakeAsyncOpenAI([final_response("resp_1", valid_estimate())])

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
        "  total hours: 120\n"
        "  component:   Backend: 120 h (refs: B-1)\n"
        "  notes:       grounded on B-1\n"
    )
