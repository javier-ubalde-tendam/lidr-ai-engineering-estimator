"""Fakes to drive the agent loop without network: a scripted ``responses.create`` + item builders."""

import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from openai.types.responses import ResponseFunctionToolCall, ResponseReasoningItem
from openai.types.responses.response_reasoning_item import Summary

Script = list[Any]  # SimpleNamespace responses, Exceptions, or callables(kwargs) -> response


def function_call(name: str, arguments: dict[str, Any] | str, call_id: str) -> Any:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return ResponseFunctionToolCall(type="function_call", name=name, arguments=raw, call_id=call_id)


def reasoning_item(*summaries: str) -> Any:
    return ResponseReasoningItem(
        id="rs_1",
        type="reasoning",
        summary=[Summary(text=text, type="summary_text") for text in summaries],
    )


def make_response(
    response_id: str,
    output: list[Any],
    *,
    output_text: str = "",
    input_tokens: int = 10,
    output_tokens: int = 5,
    reasoning_tokens: int = 2,
) -> Any:
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
    )
    return SimpleNamespace(
        id=response_id, output=output, output_text=output_text, usage=usage, status="completed"
    )


def final_response(response_id: str, estimate: dict[str, Any] | str) -> Any:
    text = estimate if isinstance(estimate, str) else json.dumps(estimate)
    return make_response(response_id, [reasoning_item("Done reasoning.")], output_text=text)


def valid_estimate(hours: float = 120.0) -> dict[str, Any]:
    return {
        "components": [{"name": "Backend", "hours": hours, "reference_budget_ids": ["B-1"]}],
        "total_hours": hours,
        "notes": "grounded on B-1",
    }


class FakeResponses:
    """Plays the script in order and records the kwargs of every ``create`` call."""

    def __init__(self, script: Script) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self._script:
            raise AssertionError("The fake client ran out of scripted responses")
        step = self._script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step(kwargs) if isinstance(step, Callable) else step


class FakeAsyncOpenAI:
    def __init__(self, script: Script) -> None:
        self.responses = FakeResponses(script)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.responses.calls


def outputs_of(call: dict[str, Any]) -> list[dict[str, Any]]:
    """The ``function_call_output`` items sent in a continuation request."""
    return [item for item in call["input"] if item.get("type") == "function_call_output"]
