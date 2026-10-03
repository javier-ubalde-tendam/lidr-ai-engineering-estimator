import json
import time

import httpx
import streamlit as st

from pydantic import ValidationError

from app.config import get_settings
from app.logging_config import configure_logging
from app.prompts.loader import render_estimation_prompt

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

def stream_estimation_via_api(request: EstimationRequest, stats: dict):
    url = f"{get_settings().API_BASE_URL}/api/v1/estimate/stream"
    current_event = None

    # mode="json" convierte los Enum a sus valores (str); sin él httpx no podría serializarlos
    with httpx.stream("POST", url, json=request.model_dump(mode="json"), timeout=60) as response:
        response.raise_for_status()  # sin esto, un 4xx/5xx acaba en una respuesta vacía sin error
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
        try:
            start = time.perf_counter()
            # write_stream consume el generador y pinta cada trozo según llega
            st.write_stream(stream_estimation_via_api(request, stats))
            elapsed = time.perf_counter() - start   # incluye la latencia HTTP, no solo la del LLM
        except httpx.HTTPError as exc:
            st.error(f"Error al llamar al servicio IA: {exc}")
        else:
            # {**d, "k": v} copia el dict y añade una clave (como un putAll de Map en Java)
            st.session_state.last_metrics = {**stats, "elapsed_seconds": elapsed}
            st.session_state.last_prompts = render_estimation_prompt(request, stats["prompt_version"])

# Panel lateral: se pinta después del formulario para ver los datos de esta misma ejecución
with st.sidebar:
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