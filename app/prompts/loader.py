from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.config import get_settings
from app.schemas.critic import CriticFeedback
from app.schemas.estimation import EstimationRequest, EstimationResult
from app.sessions.models import Message, ProjectMetadata

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


def render_conversational_prompt(
    *,
    transcript: str,
    project_type: str,
    detail_level: str,
    metadata: ProjectMetadata,
    tier: str | None,
    critic_feedback: CriticFeedback | None,
    version: str | None = None,
) -> tuple[str, str]:
    """Igual que render_estimation_prompt, pero con el bloque <project_metadata> inyectado.

    No recibe un EstimationRequest (su description tiene max_length=10000, insuficiente con
    adjuntos): el transcript ya enriquecido con adjuntos viaja como texto libre.
    `tier` y `critic_feedback` son obligatorios aunque valgan None: el entorno usa StrictUndefined
    y los prompts v4+ los leen siempre.
    """
    version = version or get_settings().CONVERSATIONAL_PROMPT_VERSION
    system = _env.get_template(f"estimation/{version}/system.j2")

    context = {
        "prompt_version": version,
        "project_type": project_type,
        "detail_level": detail_level,
        "description": transcript,
        # El entorno usa StrictUndefined: hay que pasar siempre metadata y metadata_is_empty
        "metadata": metadata,
        "metadata_is_empty": metadata.is_empty(),
        "tier": tier,
        "critic_feedback": critic_feedback,
    }

    user_prompt = render_conversational_user(
        transcript=transcript,
        project_type=project_type,
        tier=tier,
        critic_feedback=critic_feedback,
        version=version,
    )
    return system.render(**context), user_prompt


def render_conversational_user(
    *,
    transcript: str,
    project_type: str,
    tier: str | None = None,
    critic_feedback: CriticFeedback | None = None,
    version: str | None = None,
) -> str:
    # Separado de render_conversational_prompt para poder re-renderizar SOLO el user prompt con un
    # transcript distinto (p. ej. una referencia compacta a los adjuntos) sin re-renderizar el system
    version = version or get_settings().CONVERSATIONAL_PROMPT_VERSION
    user = _env.get_template(f"estimation/{version}/user.j2")
    return user.render(
        prompt_version=version,
        project_type=project_type,
        description=transcript,
        tier=tier,
        critic_feedback=critic_feedback,
    )


def render_metadata_extraction_prompt(
    *,
    previous_metadata: ProjectMetadata,
    transcript: str,
    estimation_summary: str,
    version: str = "v1",
) -> tuple[str, str]:
    system = _env.get_template(f"metadata_extraction/{version}/system.j2")
    user = _env.get_template(f"metadata_extraction/{version}/user.j2")

    context = {
        "previous_metadata": previous_metadata,
        "transcript": transcript,
        "estimation_summary": estimation_summary,
    }

    return system.render(**context), user.render(**context)


def render_conversation_summary_prompt(
    *,
    previous_summary: str | None,
    evicted: list[Message],
    version: str = "v1",
) -> tuple[str, str]:
    system = _env.get_template(f"conversation_summary/{version}/system.j2")
    user = _env.get_template(f"conversation_summary/{version}/user.j2")
    context = {"previous_summary": previous_summary, "evicted": evicted}
    return system.render(**context), user.render(**context)


def render_critic_prompt(
    *,
    transcript: str,
    metadata: ProjectMetadata,
    tier: str,
    result: EstimationResult,
    version: str = "v1",
) -> tuple[str, str]:
    system = _env.get_template(f"critic/{version}/system.j2")
    user = _env.get_template(f"critic/{version}/user.j2")
    context = {"transcript": transcript, "metadata": metadata, "tier": tier, "result": result}
    return system.render(**context), user.render(**context)