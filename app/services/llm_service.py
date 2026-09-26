from openai import OpenAI
from app.config import get_settings
from app.context.examples import ESTIMATION_EXAMPLES, format_examples_for_prompt

settings = get_settings()
client = OpenAI(api_key=settings.OPENAI_API_KEY)


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


def estimate_project(meeting_summary: str) -> tuple[str, int, int]:
    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": meeting_summary},
        ],
    )
    usage = response.usage
    return (
        response.choices[0].message.content,
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
    )