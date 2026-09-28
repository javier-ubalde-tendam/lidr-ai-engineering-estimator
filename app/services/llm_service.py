import time

import structlog
from openai import APITimeoutError, AuthenticationError, OpenAI, RateLimitError
from openai import APIError as OpenAIAPIError

from app.config import estimate_cost_usd, get_settings
from app.context.examples import ESTIMATION_EXAMPLES, format_examples_for_prompt
from collections.abc import Iterator

settings = get_settings()
client = OpenAI(api_key=settings.OPENAI_API_KEY)
logger = structlog.get_logger(__name__)


def build_system_prompt() -> str:
    examples_block = format_examples_for_prompt(ESTIMATION_EXAMPLES)
    return (
        "You are a senior technical project estimator with 10 years of experience "
        "scoping software projects for a development agency. Given a meeting "
        "summary describing a client's requirements, you produce a detailed, "
        "realistic project estimation.\n\n"
        "Your response must always follow this exact structure, in Markdown:\n"
        "- A title with the project name\n"
        "- A '### Task Breakdown' section with a table (Task | Hours | Cost (EUR))\n"
        "- A '### Totals' section with total hours and total cost\n"
        "- A '### Recommended Team' section\n"
        "- A '### Estimated Duration' section\n\n"
        "Use the following past estimations as reference for tone, granularity, "
        "and pricing (assume a blended rate of ~62.5 EUR/hour unless the "
        "complexity clearly justifies otherwise):\n\n"
        f"{examples_block}"
    )

# Endpoint de FastAPI sin streaming
def estimate_project(meeting_summary: str) -> tuple[str, int, int]:
    # Estimación gruesa (heurística ~4 caracteres/token); el valor real llega en la respuesta
    tokens_input_estimated = len(meeting_summary) // 4
    logger.info(
        "llm_call_started",
        model_requested=settings.LLM_MODEL,
        provider=settings.LLM_PROVIDER,
        tokens_input_estimated=tokens_input_estimated,
    )

    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": meeting_summary},
            ],
        )
    except AuthenticationError as exc:
        logger.error("llm_call_failed", error_type="auth", error=str(exc))
        raise
    except RateLimitError as exc:
        logger.error("llm_call_failed", error_type="rate_limit", error=str(exc))
        raise
    except APITimeoutError as exc:
        logger.error("llm_call_failed", error_type="timeout", error=str(exc))
        raise
    except OpenAIAPIError as exc:
        logger.error("llm_call_failed", error_type="server_error", error=str(exc))
        raise

    latency_ms = (time.perf_counter() - start) * 1000
    usage = response.usage
    tokens_output = usage.completion_tokens if usage else 0
    tokens_input = usage.prompt_tokens if usage else 0

    logger.info(
        "llm_call_completed",
        tokens_output=tokens_output,
        latency_ms=round(latency_ms, 2),
        cost_usd=estimate_cost_usd(settings.LLM_MODEL, tokens_input, tokens_output),
        finish_reason=response.choices[0].finish_reason,
    )

    return (response.choices[0].message.content, tokens_input, tokens_output)

# Versión con streaming, para la UI de Streamlit. Es un generador (yield en vez de return): la función no ejecuta nada hasta que alguien empieza a iterarla
def estimate_project_stream(meeting_summary: str, stats: dict) -> Iterator[str]:
    tokens_input_estimated = len(meeting_summary) // 4
    logger.info(
        "llm_call_started",
        model_requested=settings.LLM_MODEL,
        provider=settings.LLM_PROVIDER,
        tokens_input_estimated=tokens_input_estimated,
    )

    start = time.perf_counter()
    try:
        stream = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": meeting_summary},
            ],
            stream=True,
            stream_options={"include_usage": True},
        )

        for chunk in stream:
            # Los chunks "normales" traen texto; el chunk final de uso no trae choices
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
                finish_reason = chunk.choices[0].finish_reason
                if finish_reason:
                    stats["finish_reason"] = finish_reason
            # Solo el último chunk trae usage (gracias a stream_options include_usage)
            if chunk.usage:
                stats["tokens_input"] = chunk.usage.prompt_tokens
                stats["tokens_output"] = chunk.usage.completion_tokens
                stats["model_requested"] = settings.LLM_MODEL
                # chunk.model es el snapshot real servido por OpenAI, no el alias pedido
                stats["model_used"] = chunk.model
    except AuthenticationError as exc:
        logger.error("llm_call_failed", error_type="auth", error=str(exc))
        raise
    except RateLimitError as exc:
        logger.error("llm_call_failed", error_type="rate_limit", error=str(exc))
        raise
    except APITimeoutError as exc:
        logger.error("llm_call_failed", error_type="timeout", error=str(exc))
        raise
    except OpenAIAPIError as exc:
        logger.error("llm_call_failed", error_type="server_error", error=str(exc))
        raise

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "llm_call_completed",
        tokens_input=stats.get("tokens_input", 0),
        tokens_output=stats.get("tokens_output", 0),
        latency_ms=round(latency_ms, 2),
        cost_usd=estimate_cost_usd(
            stats.get("model_requested", settings.LLM_MODEL),
            stats.get("tokens_input", 0),
            stats.get("tokens_output", 0),
        ),
        finish_reason=stats.get("finish_reason"),
    )