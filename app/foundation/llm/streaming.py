"""Streaming helpers over LiteLLM tool calls (generic: the caller supplies the response model).

Instructor's ``create_partial`` consumes the whole stream before returning anything, so the stream is
opened with LiteLLM directly and the JSON fragments of the tool call feed ``Partial[...]``.
"""

import time
from collections.abc import Iterator
from typing import Any

import litellm
import structlog

from app.foundation.llm.wrapper import normalise_model_name
from app.logging_config import TRACE

logger = structlog.get_logger(__name__)


def simulate_stream_chunks(text: str, chunk_size: int = 20) -> Iterator[str]:
    # Trocea el texto ya cacheado para reproducir visualmente el mismo efecto que un streaming real
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]
        time.sleep(0.01)


def tool_json_fragments(stream: Iterator[Any], provider: str, model: str, stats: dict) -> Iterator[str]:
    for chunk in stream:
        # Comprobado con logs reales: chunk.model siempre es el nombre pedido, no un snapshot fechado
        # distinto (a diferencia de lo que asumíamos antes sin verificarlo). LiteLLM no expone aquí
        # el snapshot exacto resuelto por el proveedor.
        stats["model_requested"] = normalise_model_name(model)
        stats["model_used"] = normalise_model_name(getattr(chunk, "model", model))

        if chunk.choices:
            # Con tool calling el JSON viaja en tool_calls[0].function.arguments, no en delta.content
            tool_calls = chunk.choices[0].delta.tool_calls
            if tool_calls and tool_calls[0].function.arguments:
                yield tool_calls[0].function.arguments
            finish_reason = chunk.choices[0].finish_reason
            if finish_reason:
                stats["finish_reason"] = finish_reason

        # A diferencia del SDK de OpenAI, el chunk de LiteLLM puede no tener "usage" en absoluto
        usage = getattr(chunk, "usage", None)
        if usage:
            stats["tokens_input"] = usage.prompt_tokens
            stats["tokens_output"] = usage.completion_tokens
        # Nivel TRACE (por debajo de DEBUG): oculto salvo que bajes LOG_LEVEL por debajo de DEBUG
        logger.log(TRACE, "raw_chunk", provider=provider, chunk=repr(chunk))


def drain_on_error(fragments: Iterator[str], states: Iterator[Any]) -> Iterator[Any]:
    try:
        yield from states
    except Exception:
        # El chunk con usage llega después del último fragmento JSON: se consume para no perder los tokens
        for _ in fragments:
            pass
        raise


def open_partial_stream(
    *,
    partial_model: Any,
    tool_schema: dict[str, Any],
    provider: str,
    model: str,
    api_key: str | None,
    messages: list[dict[str, Any]],
    stats: dict,
) -> tuple[Any, Iterator[Any]]:
    """Open a streamed tool call and return ``(first_state, remaining_states)``."""
    response = litellm.completion(
        model=model,
        api_key=api_key,
        messages=messages,
        tools=[{"type": "function", "function": tool_schema}],
        tool_choice={"type": "function", "function": {"name": tool_schema["name"]}},
        stream=True,
        stream_options={"include_usage": True},
    )
    fragments = tool_json_fragments(response, provider, model, stats)
    states = drain_on_error(fragments, partial_model.model_from_chunks(fragments))
    # Pedir el primer estado fuerza la conexión: los fallos de proveedor saltan aquí, antes de emitir nada
    return next(states), states
