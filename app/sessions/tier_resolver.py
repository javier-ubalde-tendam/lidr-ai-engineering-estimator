"""Tier de audiencia dinámico: cadena ordenada de reglas puras y explicables.

Gana el override explícito; si no, la primera regla que coincide; si ninguna, `default`.
Devolver también el nombre de la regla permite mostrar "executive (nda_detected)" en vez
de un tier sin explicación.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import structlog

from app.sessions.models import ProjectMetadata

logger = structlog.get_logger(__name__)


class Tier(str, Enum):
    EXECUTIVE = "executive"
    PM = "pm"
    DEVELOPER = "developer"
    DEFAULT = "default"


@dataclass
class ResolutionContext:
    transcript: str
    metadata: ProjectMetadata


@dataclass
class TierRule:
    name: str
    tier: Tier
    predicate: Callable[[ResolutionContext], bool]


_NDA_PATTERN = re.compile(
    r"\b(nda|non[- ]?disclosure|confidential|confidencial(es|idad)?|under embargo|legal hold)\b", re.IGNORECASE
)
_REGULATORY_PATTERN = re.compile(
    r"\b(hipaa|gdpr|rgpd|lopd|sox|pci[- ]?dss|fda|iso[- ]?27001|ccpa)\b", re.IGNORECASE
)
# Cargos de dirección; no incluye CTO a propósito (es un perfil técnico)
_EXECUTIVE_ROLE_PATTERN = re.compile(
    r"\b(director(a)?\s+(general|ejecutiv[oa])|gerente\s+general|consejer[oa]\s+delegad[oa]"
    r"|ceo|cfo|coo|c-level|managing\s+director|chief\s+executive)\b",
    re.IGNORECASE,
)
_DEV_KEYWORDS_PATTERN = re.compile(
    r"\b(docker|kubernetes|k8s|microservices?|terraform|iac|helm|grpc|graphql|kafka|airflow|spark|rabbitmq)\b",
    re.IGNORECASE,
)


def _has_nda(ctx: ResolutionContext) -> bool:
    return bool(
        _NDA_PATTERN.search(ctx.transcript)
        or (ctx.metadata.agreed_scope and _NDA_PATTERN.search(ctx.metadata.agreed_scope))
    )


def _has_regulatory_context(ctx: ResolutionContext) -> bool:
    return bool(
        _REGULATORY_PATTERN.search(ctx.transcript)
        or (ctx.metadata.agreed_scope and _REGULATORY_PATTERN.search(ctx.metadata.agreed_scope))
        or any(_REGULATORY_PATTERN.search(tech) for tech in ctx.metadata.mentioned_technologies)
    )


def _is_executive_role(ctx: ResolutionContext) -> bool:
    # Solo mira el turno actual: el rol no se guarda en el metadata
    return bool(_EXECUTIVE_ROLE_PATTERN.search(ctx.transcript))


def _technical_audience(ctx: ResolutionContext) -> bool:
    # Con una sola mención ("corremos en docker") el ruido es demasiado alto: se exigen dos distintas
    return len({hit.lower() for hit in _DEV_KEYWORDS_PATTERN.findall(ctx.transcript)}) >= 2


def _is_small_team(ctx: ResolutionContext) -> bool:
    size = ctx.metadata.assumed_team_size
    return size is not None and size <= 2


# El orden es la precedencia: el contexto legal pesa más que el perfil técnico o el del equipo
_RULES: tuple[TierRule, ...] = (
    TierRule("nda_detected", Tier.EXECUTIVE, _has_nda),
    TierRule("regulatory_context", Tier.EXECUTIVE, _has_regulatory_context),
    TierRule("executive_role", Tier.EXECUTIVE, _is_executive_role),
    TierRule("technical_audience", Tier.DEVELOPER, _technical_audience),
    TierRule("low_budget_pm", Tier.PM, _is_small_team),
)


def resolve_tier(
    *,
    transcript: str,
    metadata: ProjectMetadata,
    override: Tier | None = None,
) -> tuple[Tier, str]:
    # En producción el tier debería venir de un canal autenticado, no del cliente (según los
    # apuntes); aquí el override se acepta tal cual para poder hacer demos y evals
    if override is not None:
        logger.info("tier_resolved", tier=override.value, rule="explicit_override")
        return override, "explicit_override"

    ctx = ResolutionContext(transcript=transcript, metadata=metadata)
    for rule in _RULES:
        try:
            matched = rule.predicate(ctx)
        except Exception as exc:  # noqa: BLE001 - una regla rota no debe impedir evaluar las demás
            logger.warning("tier_rule_failed", rule=rule.name, error_type=type(exc).__name__, error=str(exc))
            continue
        if matched:
            logger.info("tier_resolved", tier=rule.tier.value, rule=rule.name)
            return rule.tier, rule.name

    logger.info("tier_resolved", tier=Tier.DEFAULT.value, rule="no_rule_matched")
    return Tier.DEFAULT, "no_rule_matched"
