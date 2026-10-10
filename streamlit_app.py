import json
import time

import httpx
import streamlit as st
from pydantic import ValidationError

from app.config import get_settings
from app.logging_config import configure_logging
from app.prompts.loader import render_estimation_prompt
from app.schemas.acb import ACBResponse
from app.schemas.estimation import (
    OUT_OF_SCOPE_PREFIX,
    ConversationalEstimationResponse,
    DetailLevel,
    EstimationRequest,
    EstimationResponse,
    OutputFormat,
    ProjectType,
)

configure_logging()

# "auto" = sin override: el servidor resuelve el tier con sus reglas
TIER_OPTIONS = ["auto", "executive", "pm", "developer", "default"]

def estimate_via_api(request: EstimationRequest) -> EstimationResponse:
    url = f"{get_settings().API_BASE_URL}/api/v1/estimate"
    # httpx tiene un timeout por defecto de 5 s; una llamada al LLM suele tardar más
    response = httpx.post(url, json=request.model_dump(mode="json"), timeout=60)
    response.raise_for_status()  # lanza HTTPStatusError si el status es 4xx/5xx
    # model_validate comprueba que el JSON recibido cumple el contrato de la response
    return EstimationResponse.model_validate(response.json())

def stream_estimation_via_api(request: EstimationRequest, stats: dict):
    url = f"{get_settings().API_BASE_URL}/api/v1/estimate/stream"
    current_event = None

    # mode="json" convierte los Enum a sus valores (str); sin él httpx no podría serializarlos
    with httpx.stream("POST", url, json=request.model_dump(mode="json"), timeout=60) as response:
        if response.is_error:
            # Con streaming el body no se descarga solo: hay que leerlo aquí, antes de que
            # raise_for_status() lance y el `with` cierre la conexión (si no, .json() falla luego)
            response.read()
        response.raise_for_status()  # sin esto, un 4xx/5xx acaba en una respuesta vacía sin error
        for line in response.iter_lines():
            if line.startswith("event:"):
                current_event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                payload = json.loads(line.split(":", 1)[1].strip())
                if current_event == "metrics":
                    stats.update(payload)
                else:
                    yield current_event, payload  # partial, complete o error
                current_event = None

def show_result(result: dict, output_format: OutputFormat) -> None:
    if result["summary"].startswith(OUT_OF_SCOPE_PREFIX):
        # La fase "Not estimated" es un relleno para cumplir el schema: no se muestra como estimación
        st.warning(result["summary"])
        return
    st.markdown(result["summary"])
    st.caption(f"Confianza: {result['confidence_pct']}%")
    phases = result["phases"]
    if output_format == OutputFormat.PHASES_TABLE:
        st.dataframe(phases, hide_index=True)
    elif output_format == OutputFormat.LINE_ITEMS:
        st.markdown(
            "\n".join(
                f"{i}. **{p['name']}**: {p['duration_weeks']} semanas, {p['cost_eur']} €. {p['summary']}"
                for i, p in enumerate(phases, start=1)
            )
        )
    else:
        for p in phases:
            st.markdown(f"**{p['name']}** ({p['duration_weeks']} semanas, {p['cost_eur']} €). {p['summary']}")
    st.write(f"**Total:** {result['total_cost_eur']} € en {result['total_duration_weeks']} semanas")


# --- Modo conversacional (sesión 05): historial + project_metadata + adjuntos ---

def create_session_via_api() -> str:
    url = f"{get_settings().API_BASE_URL}/sessions"
    response = httpx.post(url, timeout=30)
    response.raise_for_status()
    return response.json()["session_id"]

def get_session_info_via_api(session_id: str) -> dict:
    url = f"{get_settings().API_BASE_URL}/sessions/{session_id}"
    response = httpx.get(url, timeout=30)
    response.raise_for_status()
    return response.json()

def estimate_conversational_via_api(
    session_id: str,
    transcript: str,
    project_type: ProjectType,
    detail_level: DetailLevel,
    output_format: OutputFormat,
    uploaded_files: list,
    tier: str = "auto",
    use_acb: bool = False,
) -> ConversationalEstimationResponse | ACBResponse:
    # El modo ACB usa otro endpoint con el mismo contrato; su response añade la traza `acb`
    endpoint = "estimate-acb" if use_acb else "estimate"
    url = f"{get_settings().API_BASE_URL}/sessions/{session_id}/{endpoint}"
    data = {
        "transcript": transcript,
        "project_type": project_type.value,
        "detail_level": detail_level.value,
        "output_format": output_format.value,
    }
    if tier != "auto":
        data["tier"] = tier
    # Un único campo "attachments" repetido: así lo espera list[UploadFile] en el endpoint
    files = [("attachments", (f.name, f.getvalue(), f.type)) for f in uploaded_files]
    # El ACB encadena varias llamadas al LLM: necesita más margen que el turno normal
    response = httpx.post(url, data=data, files=files, timeout=300 if use_acb else 120)
    response.raise_for_status()
    response_model = ACBResponse if use_acb else ConversationalEstimationResponse
    return response_model.model_validate(response.json())

def show_call_metrics(turn: dict) -> None:
    # Los campos son opcionales en la response: solo se muestran los que vienen
    parts = []
    if turn.get("latency_ms") is not None:
        parts.append(f"{turn['latency_ms']} ms")
    if turn.get("tokens_in") is not None and turn.get("tokens_out") is not None:
        parts.append(f"{turn['tokens_in']} tokens in / {turn['tokens_out']} out")
    if turn.get("cost_usd") is not None:
        parts.append(f"${turn['cost_usd']:.4f}")
    if parts:
        st.caption(" · ".join(parts))

def show_acb_trace(acb: dict) -> None:
    with st.expander(f"Traza Actor-Critic-Boss ({acb['iterations_run']} iteraciones, decisión final: {acb['final_decision']})"):
        for iteration in acb["iterations"]:
            st.markdown(
                f"**Iteración {iteration['iteration']}** · veredicto del critic: `{iteration['critic_verdict']}` "
                f"(confianza {iteration['critic_confidence']}%) · decisión del boss: `{iteration['decision_after']}`"
            )
            for issue in iteration["issue_summary"]:
                st.caption(f"- {issue}")
        st.caption(f"**Decisión final:** {acb['final_decision']}")

def show_api_error(exc: httpx.HTTPStatusError) -> None:
    # Mismo formato que en el modo formulario: el 400 de los guardrails viaja como {"reason", "message"}
    try:
        detail = exc.response.json().get("detail")
    except ValueError:
        detail = exc.response.text
    if isinstance(detail, dict) and "reason" in detail:
        st.error(f"Entrada bloqueada ({detail['reason']}): {detail['message']}")
    else:
        st.error(f"Error del servicio ({exc.response.status_code}): {detail or exc}")


st.title("IA Estimator")

mode = st.radio(
    "Modo",
    options=["Formulario (stream)", "Conversación (memoria + adjuntos)"],
    horizontal=True,
)

if mode == "Formulario (stream)":
    with st.form("estimation_form"):
        description = st.text_area("Descripción del proyecto", height=200)
        project_type = st.selectbox(
            "Tipo de proyecto",
            options=list(ProjectType),
            format_func=lambda p: p.value,  # muestra "web_saas" en vez de "ProjectType.WEB_SAAS"
        )
        detail_level = st.selectbox(
            "Nivel de detalle",
            options=list(DetailLevel),
            format_func=lambda d: d.value,  # muestra "summary" en vez de "DetailLevel.SUMMARY"
        )
        output_format = st.selectbox(
            "Formato de salida",
            options=list(OutputFormat),
            format_func=lambda o: o.value,  # muestra "phases_table" en vez de "OutputFormat.PHASES_TABLE"
        )
        submitted = st.form_submit_button("Estimar")

    if submitted:
        try:
            request = EstimationRequest(
                description=description,
                project_type=project_type,
                detail_level=detail_level,
                output_format=output_format,
            )
        except ValidationError as exc:
            # exc.errors() es una lista de dicts: "loc" = campo, "msg" = motivo
            for err in exc.errors():
                st.error(f"{err['loc'][0]}: {err['msg']}")
        else:
            stats: dict = {}  # el generador lo rellena al recibir el evento "metrics"
            placeholder = st.empty()  # un único hueco que se reescribe con cada estado parcial
            result, error = None, None
            try:
                start = time.perf_counter()
                for event, payload in stream_estimation_via_api(request, stats):
                    if event == "partial":
                        placeholder.json(payload)  # provisional: campos a null y números a medias
                    elif event == "complete":
                        result = payload
                    elif event == "error":
                        error = payload["detail"]
                elapsed = time.perf_counter() - start   # incluye la latencia HTTP, no solo la del LLM
            except httpx.HTTPStatusError as exc:
                # El 400 de los guardrails de input viaja como {"reason", "message"}; el 502 del LLM, como string
                detail = exc.response.json().get("detail")
                if isinstance(detail, dict) and "reason" in detail:
                    st.error(f"Entrada bloqueada ({detail['reason']}): {detail['message']}")
                else:
                    st.error(f"Error al llamar al servicio IA: {detail or exc}")
            except httpx.HTTPError as exc:
                st.error(f"Error al llamar al servicio IA: {exc}")
            else:
                if result:
                    with placeholder.container():  # sustituye el JSON parcial por el resultado validado
                        show_result(result, request.output_format)
                else:
                    placeholder.error(error or "El servicio no devolvió ninguna estimación")
                # {**d, "k": v} copia el dict y añade una clave (como un putAll de Map en Java)
                st.session_state.last_metrics = {**stats, "elapsed_seconds": elapsed}
                st.session_state.last_prompts = render_estimation_prompt(request, stats["prompt_version"])

    # Panel lateral: se pinta después del formulario para ver los datos de esta misma ejecución
    with st.sidebar:
        # Validaciones blandas: el resultado se muestra igualmente, pero el usuario debe saber que falló una
        for warning in st.session_state.get("last_metrics", {}).get("validation_warnings", []):
            st.error(f"Validación no superada: {warning}")
        st.header("Contexto del LLM")
        prompts = st.session_state.get("last_prompts")
        if prompts:
            system_prompt, user_prompt = prompts
            with st.expander("System prompt"):
                st.text(system_prompt)
            with st.expander("User prompt"):
                st.text(user_prompt)
        else:
            st.caption("Se mostrará tras la primera estimación.")

        st.subheader("Última llamada")
        metrics = st.session_state.get("last_metrics")
        if metrics:
            st.caption(f"**Exact cache hit:** {metrics.get('exact_cache_hit', metrics['cache_hit'])}")
            st.caption(f"**Semantic cache hit:** {metrics.get('semantic_cache_hit', False)}")
            # En cache hit el servidor no llama al LLM: no hay proveedor, modelo ni tokens
            st.caption(f"**Proveedor:** {metrics.get('provider_used', '-')}")
            st.caption(f"**Modelo:** {metrics.get('model_used', '-')}")
            st.caption(f"**Versión del prompt:** {metrics['prompt_version']}")
            st.caption(f"**Tokens entrada:** {metrics.get('tokens_input', '-')}")
            st.caption(f"**Tokens salida:** {metrics.get('tokens_output', '-')}")
            st.caption(f"**Tiempo:** {metrics['elapsed_seconds']:.2f} s")
        else:
            st.caption("Todavía no se ha generado ninguna estimación.")

else:
    # Modo conversacional (sesión 05): historial + project_metadata + adjuntos.
    # NO reutiliza el panel de "Contexto del LLM" renderizando prompts localmente
    # (render_estimation_prompt) porque aquí el prompt real depende del estado del servidor
    # (historial + project_metadata de la sesión), no solo de los campos del formulario.
    if "session_id" not in st.session_state:
        st.session_state.session_id = create_session_via_api()
        st.session_state.conversation_turns = []

    if st.button("Nueva conversación"):
        st.session_state.session_id = create_session_via_api()
        st.session_state.conversation_turns = []
        st.rerun()

    for turn in st.session_state.get("conversation_turns", []):
        with st.chat_message(turn["role"]):
            if turn["role"] == "assistant":
                show_result(turn["result"], OutputFormat(turn["output_format"]))
                show_call_metrics(turn)
                if turn.get("acb"):
                    show_acb_trace(turn["acb"])
            else:
                st.markdown(turn["content"])

    with st.form("conversation_form", clear_on_submit=True):
        transcript = st.text_area("Transcripción / mensaje para este turno", height=200)
        uploaded_files = st.file_uploader(
            "Adjuntos (opcional)", type=["pdf", "docx"], accept_multiple_files=True
        )
        project_type = st.selectbox("Tipo de proyecto", options=list(ProjectType), format_func=lambda p: p.value)
        detail_level = st.selectbox("Nivel de detalle", options=list(DetailLevel), format_func=lambda d: d.value)
        output_format = st.selectbox("Formato de salida", options=list(OutputFormat), format_func=lambda o: o.value)
        tier = st.selectbox("Tier de audiencia", options=TIER_OPTIONS, help="auto = lo decide el servidor según el contexto")
        use_acb = st.toggle("Revisión Actor-Critic-Boss", help="Más lento y caro: un critic audita cada borrador")
        submitted = st.form_submit_button("Enviar turno")

    if submitted:
        if len(transcript) < 20:
            st.error("transcript: debe tener al menos 20 caracteres")
        else:
            try:
                with st.spinner("Estimando..."):
                    response = estimate_conversational_via_api(
                        st.session_state.session_id,
                        transcript,
                        project_type,
                        detail_level,
                        output_format,
                        uploaded_files,
                        tier=tier,
                        use_acb=use_acb,
                    )
            except httpx.HTTPStatusError as exc:
                show_api_error(exc)
            except httpx.HTTPError as exc:
                st.error(f"Error al llamar al servicio IA: {exc}")
            else:
                st.session_state.conversation_turns.append({"role": "user", "content": transcript})
                st.session_state.conversation_turns.append(
                    {
                        "role": "assistant",
                        "output_format": output_format.value,
                        # Todo lo que trae la response (resultado, métricas y, en ACB, la traza `acb`)
                        **response.model_dump(mode="json"),
                    }
                )
                st.rerun()

    with st.sidebar:
        st.header("Estado de la sesión")
        st.caption(f"**session_id:** {st.session_state.session_id}")
        try:
            info = get_session_info_via_api(st.session_state.session_id)
        except httpx.HTTPError as exc:
            st.error(f"No se pudo leer el estado de la sesión: {exc}")
        else:
            # La ventana admite max_turns pares: message_count cuenta mensajes sueltos (user + assistant)
            st.caption(f"**Mensajes en la ventana:** {info['message_count']} / {info['max_turns'] * 2}")
            st.caption(f"**Anclas (mensajes):** {info['anchors_count']}")
            st.caption(f"**Resumen acumulado:** {info['summary_chars']} caracteres")
            if info["last_resolved_tier"]:
                st.caption(f"**Tier:** {info['last_resolved_tier']} ({info['last_tier_rule']})")
            else:
                st.caption("**Tier:** aún sin resolver")
            st.subheader("Project metadata")
            metadata = info["metadata"]
            metadata_fields = [
                ("Nombre del proyecto", metadata.get("project_name")),
                ("Tamaño del equipo", metadata.get("assumed_team_size")),
                ("Tecnologías", ", ".join(metadata.get("mentioned_technologies") or [])),
                ("Alcance acordado", metadata.get("agreed_scope")),
            ]
            shown = False
            for label, value in metadata_fields:
                if value:  # solo se muestra un campo si tiene valor
                    st.caption(f"**{label}:** {value}")
                    shown = True
            if not shown:
                st.caption("Todavía no hay hechos conocidos del proyecto.")