from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.config import get_settings
from app.schemas.estimation import EstimationRequest

PROMPTS_DIR = Path(__file__).parent

# Sin autoescape (por defecto): el prompt es texto plano para un LLM, no HTML
_env = Environment(
    loader=FileSystemLoader(PROMPTS_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
    undefined=StrictUndefined,
)

def render_estimation_prompt(
    request: EstimationRequest,
    version: str | None = None,
) -> tuple[str, str]:
    # None = usar PROMPT_VERSION de la config (.env); un valor explícito lo sobrescribe
    version = version or get_settings().PROMPT_VERSION
    system = _env.get_template(f"estimation/{version}/system.j2")
    user = _env.get_template(f"estimation/{version}/user.j2")

    context = {
        "prompt_version": version,  # system.j2 lo usa para incluir el examples.j2 de la misma versión
        "project_type": request.project_type.value,
        "detail_level": request.detail_level.value,
        "output_format": request.output_format.value,
        "description": request.description,
    }

    return system.render(**context), user.render(**context)