import json
import time

import httpx
import streamlit as st

from pydantic import ValidationError

from app.config import get_settings
from app.logging_config import configure_logging
from app.prompts.loader import render_estimation_prompt

from app.schemas.estimation import (
    OUT_OF_SCOPE_PREFIX,
    DetailLevel,
    EstimationRequest,
    EstimationResponse,
    OutputFormat,
    ProjectType,
)

configure_logging()

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

st.title("IA Estimator - Formulario tipado")

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
        st.caption(f"**Cache hit:** {metrics['cache_hit']}")
        # En cache hit el servidor no llama al LLM: no hay proveedor, modelo ni tokens
        st.caption(f"**Proveedor:** {metrics.get('provider_used', '-')}")
        st.caption(f"**Modelo:** {metrics.get('model_used', '-')}")
        st.caption(f"**Versión del prompt:** {metrics['prompt_version']}")
        st.caption(f"**Tokens entrada:** {metrics.get('tokens_input', '-')}")
        st.caption(f"**Tokens salida:** {metrics.get('tokens_output', '-')}")
        st.caption(f"**Tiempo:** {metrics['elapsed_seconds']:.2f} s")
    else:
        st.caption("Todavía no se ha generado ninguna estimación.")