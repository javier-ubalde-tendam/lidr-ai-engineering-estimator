"""Boss: orquesta Actor y Critic sin hacer llamadas LLM propias.

    actor -> critic -> decidir
                        |- accept                          -> devolver
                        |- needs_iteration + iteraciones   -> iterate: actor de nuevo con el feedback
                        |- reject / presupuesto agotado    -> synthesize: último borrador anotado

Al no llamar al LLM, cada decisión es reproducible desde sus entradas y queda en `BossTrace`.
El actor y el critic llegan como callables para poder re-invocar al actor con el feedback previo.
"""

from collections.abc import Callable

import structlog
from annotated_types import MaxLen

from app.schemas.acb import ACBIteration, BossDecision, BossTrace
from app.schemas.critic import CriticFeedback, CriticIssue
from app.schemas.estimation import LOW_CONFIDENCE_THRESHOLD, OUT_OF_SCOPE_PREFIX, EstimationResult

logger = structlog.get_logger(__name__)

ActorCallable = Callable[[CriticFeedback | None], EstimationResult]
CriticCallable = Callable[[EstimationResult], CriticFeedback]

# Se lee del schema para que el recorte siga el límite real aunque alguien lo cambie
SUMMARY_MAX_LENGTH: int = next(
    meta.max_length for meta in EstimationResult.model_fields["summary"].metadata if isinstance(meta, MaxLen)
)
# Tope del bloque de avisos: con 12 issues ocuparía todo el summary y desaparecería la estimación
_CAVEATS_MAX_CHARS = 600
_SEVERITY_ORDER = {"critical": 0, "major": 1, "minor": 2}


class Boss:
    def __init__(self, max_iterations: int) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        self.max_iterations = max_iterations

    def run(self, actor: ActorCallable, critic: CriticCallable) -> tuple[EstimationResult, BossTrace]:
        trace = BossTrace(final_decision="accept", iterations_run=0)
        feedback: CriticFeedback | None = None

        for iteration in range(self.max_iterations):
            draft = actor(feedback)
            review = critic(draft)
            iterations_left = self.max_iterations - iteration - 1
            decision = self._decide(review, iterations_left=iterations_left)

            trace.iterations.append(
                ACBIteration(
                    iteration=iteration,
                    decision_after=decision,
                    critic_verdict=review.verdict,
                    critic_confidence=review.confidence_in_review,
                    issue_summary=[f"[{i.severity}] {i.category} @ {i.field_path}" for i in review.issues],
                )
            )
            trace.iterations_run = iteration + 1
            logger.info(
                "boss_iteration",
                iteration=iteration,
                decision=decision,
                verdict=review.verdict,
                issues=len(review.issues),
            )

            if decision == "accept":
                trace.final_decision = "accept"
                return draft, trace
            if decision == "synthesize":
                if review.verdict == "needs_iteration":
                    logger.info("boss_iteration_budget_exhausted", iterations=self.max_iterations)
                trace.final_decision = "synthesize"
                return self._synthesize_fallback(draft, review), trace
            feedback = review

        # Inalcanzable: en la última vuelta _decide solo devuelve accept o synthesize
        raise AssertionError("Boss loop ended without a decision")

    @staticmethod
    def _decide(review: CriticFeedback, *, iterations_left: int) -> BossDecision:
        if review.verdict == "accept":
            return "accept"
        if review.verdict == "needs_iteration" and iterations_left > 0:
            return "iterate"
        return "synthesize"

    @staticmethod
    def _synthesize_fallback(draft: EstimationResult, review: CriticFeedback) -> EstimationResult:
        """Último borrador con los issues abiertos al inicio del summary y la confianza a la mitad.

        Un borrador con avisos informa más que un resultado vacío; que el bucle no convergió ya
        viaja en `BossTrace.final_decision`.
        """
        # Un "Out of scope:" ya es un resultado de rechazo con fase de relleno: anotarlo lo desfiguraría
        if not review.issues or draft.summary.startswith(OUT_OF_SCOPE_PREFIX):
            return draft

        caveats = "⚠ Open caveats from independent review:\n"
        for issue in sorted(review.issues, key=lambda i: _SEVERITY_ORDER[i.severity]):
            line = _caveat_line(issue)
            if len(caveats) + len(line) > _CAVEATS_MAX_CHARS:
                break
            caveats += line
        summary = f"{caveats}\n{draft.summary}"[:SUMMARY_MAX_LENGTH]

        # El suelo evita que el validator de baja confianza exija el prefijo "Out of scope:"
        confidence = max(LOW_CONFIDENCE_THRESHOLD, draft.confidence_pct // 2)
        return draft.model_copy(update={"summary": summary, "confidence_pct": confidence})


def _caveat_line(issue: CriticIssue) -> str:
    text = f"- [{issue.severity}] {issue.category} ({issue.field_path}): {issue.description}"
    return (text if len(text) <= 160 else f"{text[:157]}...") + "\n"
