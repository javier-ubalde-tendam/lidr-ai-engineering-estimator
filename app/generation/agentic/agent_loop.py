"""Manual agent loop over the raw OpenAI Responses API (Session 12).

This is the deliberate exception to "every LLM call goes through ``LLMWrapper``": the point of the
exercise is to SEE the loop, so there is no SDK agent, ``tool_runner`` or framework here. We drive
``reason -> act -> observe -> repeat`` ourselves and record every turn in a trace.

Loop contract (verified against the OpenAI *Function calling* guide):

- Each turn is one ``client.responses.create`` call. The first one sends the transcript; the next
  ones send ONLY the new items (``function_call_output`` / a correction) and chain with
  ``previous_response_id`` (the server keeps the earlier items, reasoning items included).
- ``instructions`` are NOT carried over by ``previous_response_id``: they are re-sent every turn,
  together with ``tools``.
- A turn may contain several ``function_call`` items; they run in parallel and ALL their
  ``function_call_output`` go back in a single request, each with its own ``call_id``.
- A turn with no ``function_call`` is the final answer: it must validate against ``AgentEstimate``.
  The strict ``text.format`` json_schema is sent on every call (it only constrains the final
  message, never the function calls) and Pydantic validates the result afterwards. One corrective
  turn is allowed; then the run ends with ``status="error"``.
- The loop is bounded by ``max_iterations`` model turns. To make it converge, the last turns are
  reserved: from ``max_iterations - wrap_up_turns`` on, ``tool_choice`` only allows the
  calculation/validation tools, and the last turn has no tools (``"none"``). Every continuation
  carries a short "turn N of M" notice. ``max_steps_exceeded`` stays as the safety net.
"""

import asyncio
import json
import time
from typing import Any, Literal

import structlog
from openai import OpenAIError
from pydantic import ValidationError

from app.generation.agentic.agent_schemas import (
    AgentAction,
    AgentEstimate,
    AgentResult,
    AgentStep,
    TokenUsage,
    strict_json_schema,
)
from app.generation.agentic.agent_tools import TOOL_SCHEMAS, ToolRegistry, execute_tool

log = structlog.get_logger()

NO_REASONING_SUMMARY = "(no reasoning summary available)"
ARGUMENT_PREVIEW_MAX_CHARS = 300

SYSTEM_PROMPT_TEMPLATE = """\
You are a software estimation agent. You receive the transcript of a client discovery meeting \
and must produce an effort estimate in engineer-hours, grounded on the company's historical \
data instead of on your own guesses.

The historical database holds individual engineering tasks at functional level: a feature or a \
piece of a module that takes roughly 16 to 72 engineer-hours. It does not hold whole systems, \
and it does not hold small chores (documentation, CI setup, configuration) or product-specific \
details. A project component is only comparable to history once it is broken down to that level.

Method:
1. Read the transcript and break the project down into its distinct components, and each \
component into functional tasks of that size (typically 3 to 6 per component). Ignore small talk.
2. Look for similar historical tasks with the search tool, one task per search, with a query \
built from what the task does and the technologies involved. Issue all the searches that do not \
depend on each other in the same turn.
3. Judge the evidence and use only the items that really match the task. When results are \
missing or weak, rephrase with more generic wording (or drop the filters) and search again, \
issuing all the rephrasings of that round together in the same turn. If two rephrasings still \
give no good match, the history has no equivalent: leave the task without references, say so \
in `notes`, and move on.
4. Compute the numbers with the calculation tool, feeding it, for every task, the hours of its \
relevant historical items. Do not do the cost arithmetic yourself.
5. If a validation tool is available, validate the draft and resolve the issues it reports \
before answering.

Turn budget: you have at most {max_iterations} turns in total (a turn is one model response, \
which may contain several tool calls). {reserved}An answer that arrives after the last turn is \
lost.

Rules:
- Never invent hours for a task that has no historical evidence: it costs 0 hours; say so in \
`notes`.
- Each component's `hours` is the sum of its tasks. In `reference_budget_ids`, list the \
budget_id of the historical items that support each component.
- Hours are engineer-hours. `total_hours` is the sum of the component hours.
- Use `notes` for assumptions, gaps in the evidence and risks.
- Your final message must be the JSON estimate and nothing else.
"""

# The last turns are reserved for closing the estimate (see ``_tool_choice``)
WRAP_UP_TURNS = 3
WRAP_UP_TOOLS = ("calculate_estimate", "validate_estimate")


def wrap_up_turns(max_iterations: int) -> int:
    """Turns reserved at the end; a short run keeps (almost) all its turns for searching."""
    return min(WRAP_UP_TURNS, max_iterations // 3)


def build_system_prompt(max_iterations: int) -> str:
    """The agent instructions, telling the model how many turns it has and how they are used."""
    reserved = wrap_up_turns(max_iterations)
    if reserved:
        text = (
            f"Searching is only possible during the first {max_iterations - reserved} turns; "
            "the remaining turns are reserved for the calculation, the validation and the final "
            "answer, and the very last turn has no tools at all. "
        )
    else:
        text = "Make sure the calculation, the validation and the final answer fit inside it. "
    return SYSTEM_PROMPT_TEMPLATE.format(max_iterations=max_iterations, reserved=text)


def _tool_choice(
    iteration: int, max_iterations: int, tools: list[dict[str, Any]]
) -> dict[str, Any] | str | None:
    """Constrain the closing turns so the run converges instead of searching until it runs out.

    Last turn: no tools (the model can only answer). The turns before it, up to
    ``wrap_up_turns``: only the calculation and validation tools (``allowed_tools`` keeps the
    full ``tools`` list unchanged, so the prompt cache is not invalidated).
    """
    turns_after = max_iterations - iteration
    if turns_after >= wrap_up_turns(max_iterations):
        return None
    allowed = [
        {"type": "function", "name": tool["name"]}
        for tool in tools
        if tool.get("name") in WRAP_UP_TOOLS
    ]
    if turns_after == 0 or not allowed:
        return "none"
    return {"type": "allowed_tools", "mode": "auto", "tools": allowed}


def _turn_notice(iteration: int, max_iterations: int, tools: list[dict[str, Any]]) -> str:
    """Short status message for turn ``iteration``, sent together with the tool outputs."""
    left = max_iterations - iteration + 1
    choice = _tool_choice(iteration, max_iterations, tools)
    if choice == "none":
        phase = "This is the LAST turn: tools are disabled, reply now with the final JSON estimate."
    elif choice is not None:
        phase = (
            "Searching is closed: only calculate_estimate and validate_estimate are available; "
            "the final JSON estimate comes in the last turn at the latest."
        )
    else:
        search_turns = max_iterations - wrap_up_turns(max_iterations)
        phase = f"Searching is possible until turn {search_turns}."
    return f"[Turn {iteration} of {max_iterations}; {left} left including this one. {phase}]"


FINAL_ANSWER_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "name": "agent_estimate",
    "schema": strict_json_schema(AgentEstimate),
    "strict": True,
}


class AgentLoopError(Exception):
    """The OpenAI call failed mid-run; ``trace`` keeps the steps completed before the failure."""

    def __init__(self, message: str, trace: list[AgentStep]) -> None:
        super().__init__(message)
        self.trace = trace


# --- Reading a response --------------------------------------------------------------------


def _usage_of(response: Any) -> TokenUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    details = getattr(usage, "output_tokens_details", None)
    return TokenUsage(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        reasoning_tokens=getattr(details, "reasoning_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
    )


def _message_text(item: Any) -> str:
    parts = getattr(item, "content", None) or []
    return "".join(getattr(part, "text", "") or "" for part in parts)


def _reasoning_of(output: list[Any]) -> str:
    """What the agent said about this turn: reasoning summaries, else any text it wrote."""
    summaries = [
        getattr(part, "text", "")
        for item in output
        if getattr(item, "type", None) == "reasoning"
        for part in (getattr(item, "summary", None) or [])
    ]
    text = "\n".join(summary for summary in summaries if summary).strip()
    if text:
        return text
    spoken = "\n".join(
        _message_text(item) for item in output if getattr(item, "type", None) == "message"
    ).strip()
    return spoken or NO_REASONING_SUMMARY


def _arguments_for_trace(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {"raw_arguments": str(raw)}
    return parsed if isinstance(parsed, dict) else {"raw_arguments": parsed}


async def _run_action(registry: ToolRegistry, call: Any) -> AgentAction:
    started = time.perf_counter()
    observation = await execute_tool(registry, call.name, call.arguments)
    duration_ms = round((time.perf_counter() - started) * 1000)
    error = observation["error"] if isinstance(observation.get("error"), str) else None
    log.info(
        "agent_tool_call",
        tool=call.name,
        call_id=call.call_id,
        duration_ms=duration_ms,
        error=error is not None,
    )
    return AgentAction(
        tool=call.name,
        call_id=call.call_id,
        arguments=_arguments_for_trace(call.arguments),
        observation=observation,
        duration_ms=duration_ms,
        error=error,
    )


def _invalid_answer_message(exc: ValidationError) -> str:
    problems = "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'answer'}: {error['msg']}"
        for error in exc.errors()
    )
    return problems[:500]


# --- The loop ------------------------------------------------------------------------------


async def run_agent(
    transcript: str,
    *,
    client: Any,
    registry: ToolRegistry,
    model: str,
    effort: str,
    max_iterations: int,
    tools: list[dict[str, Any]] | None = None,
    instructions: str | None = None,
    reasoning_summary: Literal["auto", "concise", "detailed"] | None = "auto",
) -> AgentResult:
    """Drive the agent until it answers, fails validation twice or runs out of iterations.

    ``client`` is an ``AsyncOpenAI`` (or any object with ``responses.create``). ``instructions``
    defaults to ``build_system_prompt(max_iterations)``. Raises ``AgentLoopError`` when the
    OpenAI call itself fails.
    """
    tool_schemas = TOOL_SCHEMAS if tools is None else tools
    instructions = instructions or build_system_prompt(max_iterations)
    reasoning: dict[str, Any] = {"effort": effort}
    if reasoning_summary:
        reasoning["summary"] = reasoning_summary

    trace: list[AgentStep] = []
    total_usage = TokenUsage()
    previous_response_id: str | None = None
    next_input: list[dict[str, Any]] = [{"role": "user", "content": transcript}]
    correction_sent = False

    for iteration in range(1, max_iterations + 1):
        started = time.perf_counter()
        params: dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": next_input,
            "tools": tool_schemas,
            "reasoning": reasoning,
            "text": {"format": FINAL_ANSWER_FORMAT},
        }
        if previous_response_id:
            params["previous_response_id"] = previous_response_id
        tool_choice = _tool_choice(iteration, max_iterations, tool_schemas)
        if tool_choice is not None:
            params["tool_choice"] = tool_choice
        try:
            response = await client.responses.create(**params)
        except OpenAIError as exc:
            log.error("agent_call_failed", step=iteration, error_type=type(exc).__name__)
            raise AgentLoopError(f"OpenAI call failed at step {iteration}: {exc}", trace) from exc
        if getattr(response, "status", None) == "failed":
            raise AgentLoopError(
                f"OpenAI response failed at step {iteration}: {getattr(response, 'error', None)}",
                trace,
            )

        step_usage = _usage_of(response)
        total_usage += step_usage
        output = list(response.output)
        calls = [item for item in output if getattr(item, "type", None) == "function_call"]
        step = AgentStep(step=iteration, reasoning=_reasoning_of(output), usage=step_usage)
        previous_response_id = response.id

        if not calls:
            try:
                estimate = AgentEstimate.model_validate_json(response.output_text)
            except ValidationError as exc:
                step.validation_error = _invalid_answer_message(exc)
                step.duration_ms = round((time.perf_counter() - started) * 1000)
                trace.append(step)
                log.warning("agent_invalid_answer", step=iteration, error=step.validation_error)
                if correction_sent:
                    return _finish(
                        "error",
                        trace,
                        iteration,
                        total_usage,
                        None,
                        "The final answer did not validate after one correction.",
                    )
                correction_sent = True
                next_input = [
                    {
                        "role": "user",
                        "content": (
                            f"Your final answer is invalid: {step.validation_error}. "
                            "Reply again with the corrected JSON estimate only."
                        ),
                    }
                ]
                continue
            step.duration_ms = round((time.perf_counter() - started) * 1000)
            trace.append(step)
            log.info("agent_step", step=iteration, tools=[], tokens=step_usage.total_tokens)
            return _finish("done", trace, iteration, total_usage, estimate)

        step.actions = list(await asyncio.gather(*(_run_action(registry, call) for call in calls)))
        step.duration_ms = round((time.perf_counter() - started) * 1000)
        trace.append(step)
        log.info(
            "agent_step",
            step=iteration,
            tools=[action.tool for action in step.actions],
            tokens=step_usage.total_tokens,
        )
        # One request with ALL the outputs, each tied to its own call_id, plus the turn status
        next_input = [
            {
                "type": "function_call_output",
                "call_id": action.call_id,
                "output": json.dumps(action.observation),
            }
            for action in step.actions
        ]
        if iteration < max_iterations:
            next_input.append(
                {
                    "role": "user",
                    "content": _turn_notice(iteration + 1, max_iterations, tool_schemas),
                }
            )
    # Reached only when every iteration was spent (every other path above returns): this is the
    # `for ... else` of the exercise, written without `else` because ruff PLW0120 forbids it
    return _finish(
        "max_steps_exceeded",
        trace,
        max_iterations,
        total_usage,
        None,
        f"Stopped after {max_iterations} iterations without a final answer.",
    )


def _finish(
    status: Literal["done", "max_steps_exceeded", "error"],
    trace: list[AgentStep],
    iterations: int,
    usage: TokenUsage,
    estimate: AgentEstimate | None,
    error: str | None = None,
) -> AgentResult:
    log.info(
        "agent_done",
        status=status,
        iterations=iterations,
        total_hours=estimate.total_hours if estimate else None,
        tokens=usage.total_tokens,
    )
    return AgentResult(
        status=status,
        estimate=estimate,
        trace=trace,
        iterations=iterations,
        usage=usage,
        error=error,
    )


# --- Human-readable trace ------------------------------------------------------------------

_LABEL_WIDTH = 13


def _field(label: str, text: str) -> str:
    """``  label:   text``; continuation lines are aligned under the text."""
    lines = (text or "").splitlines() or [""]
    prefix = f"  {label + ':':<{_LABEL_WIDTH}}"
    padding = " " * len(prefix)
    return "\n".join((prefix if index == 0 else padding) + line for index, line in enumerate(lines))


def _format_call(action: AgentAction) -> str:
    arguments = ", ".join(
        f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in action.arguments.items()
    )
    call = f"{action.tool}({arguments})"
    if len(call) > ARGUMENT_PREVIEW_MAX_CHARS:
        call = call[: ARGUMENT_PREVIEW_MAX_CHARS - 3] + "..."
    return call


def _format_observation(action: AgentAction) -> str:
    if action.error:
        return f"ERROR: {action.error}"
    return str(action.observation.get("summary") or json.dumps(action.observation)[:200])


def _format_step(step: AgentStep) -> str:
    lines = [f"STEP {step.step}", _field("reasoning", step.reasoning)]
    for action in step.actions:
        lines.append(_field("action", _format_call(action)))
        lines.append(_field("observation", _format_observation(action)))
    if not step.actions:
        lines.append(_field("action", "(none: final answer)"))
        outcome = (
            f"ERROR: final answer rejected ({step.validation_error})"
            if step.validation_error
            else "final answer produced"
        )
        lines.append(_field("observation", outcome))
    return "\n".join(lines)


def format_trace(result: AgentResult) -> str:
    """Readable STEP / reasoning / action / observation trace followed by the outcome."""
    blocks = [_format_step(step) for step in result.trace]
    usage = result.usage
    summary = [
        "RESULT",
        _field("status", result.status),
        _field("iterations", str(result.iterations)),
        _field(
            "tokens",
            f"input={usage.input_tokens} output={usage.output_tokens} "
            f"reasoning={usage.reasoning_tokens} total={usage.total_tokens}",
        ),
    ]
    if result.estimate:
        summary.append(_field("total hours", f"{result.estimate.total_hours:g}"))
        for component in result.estimate.components:
            refs = ", ".join(component.reference_budget_ids) or "none"
            summary.append(
                _field("component", f"{component.name}: {component.hours:g} h (refs: {refs})")
            )
        summary.append(_field("notes", result.estimate.notes))
    if result.error:
        summary.append(_field("error", result.error))
    blocks.append("\n".join(summary))
    return "\n\n".join(blocks) + "\n"
