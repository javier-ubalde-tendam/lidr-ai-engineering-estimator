import json
import time

import httpx
import streamlit as st

from pydantic import ValidationError

from app.config import get_settings
from app.context.examples import ESTIMATION_EXAMPLES, format_examples_for_prompt
from app.logging_config import configure_logging
from app.services.llm_service import build_system_prompt

from app.schemas.estimation import (
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

# POR EL MOMENTO AL METER EL FORMULARIO TIPADO, EL STREAMING NO SE USA EN LA INTERFAZ DE STREAMLIT
def stream_estimation_via_api(request: EstimationRequest, stats: dict):
    url = f"{get_settings().API_BASE_URL}/api/v1/estimate/stream"
    current_event = None

    # mode="json" convierte los Enum a sus valores (str); sin él httpx no podría serializarlos
    with httpx.stream("POST", url, json=request.model_dump(mode="json"), timeout=60) as response:
        for line in response.iter_lines():
            if line.startswith("event:"):
                current_event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                payload = json.loads(line.split(":", 1)[1].strip())
                if current_event == "token":
                    yield payload["content"]
                elif current_event == "metrics":
                    stats.update(payload)
                current_event = None

st.title("IA Estimator - Formulario tipado")

# Panel lateral
with st.sidebar:
    st.header("Contexto del LLM")

    with st.expander("System prompt"):
        st.text(build_system_prompt())

    with st.expander("Ejemplos inyectados (CAG)"):
        st.text(format_examples_for_prompt(ESTIMATION_EXAMPLES))


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
        try:
            start = time.perf_counter()
            with st.spinner("Generando estimación..."):
                result = estimate_via_api(request)
            elapsed = time.perf_counter() - start   # incluye la latencia HTTP, no solo la del LLM
        except httpx.HTTPError as exc:
            st.error(f"Error al llamar al servicio IA: {exc}")
        else:
            # exclude evita guardar el texto largo de la estimación en session_state
            # {**d, "k": v} copia el dict y añade una clave (como un putAll de Map en Java)
            st.session_state.last_metrics = {
                **result.model_dump(exclude={"estimation"}),
                "elapsed_seconds": elapsed,
            }
            st.markdown(result.estimation)
            st.caption(f"{result.provider} / {result.model} · prompt {result.prompt_version}")

# Segundo bloque de sidebar: se ejecuta después de calcular las métricas de esta consulta
with st.sidebar:
    st.subheader("Última llamada")
    metrics = st.session_state.get("last_metrics")
    if metrics:
        st.caption(f"**Cache hit:** {metrics.get('cache_hit', False)}")
        st.caption(f"**Proveedor:** {metrics['provider']}")
        st.caption(f"**Modelo:** {metrics['model']}")
        st.caption(f"**Versión del prompt:** {metrics['prompt_version']}")
        st.caption(f"**Tokens entrada:** {metrics['tokens_input']}")
        st.caption(f"**Tokens salida:** {metrics['tokens_output']}")
        st.caption(f"**Tiempo:** {metrics['elapsed_seconds']:.2f} s")
    else:
        st.caption("Todavía no se ha generado ninguna estimación.")