import time

import streamlit as st

from app.context.examples import ESTIMATION_EXAMPLES, format_examples_for_prompt
from app.logging_config import configure_logging
from app.services.llm_service import build_system_prompt, estimate_project, estimate_project_stream

configure_logging()

st.title("Estimador de Proyectos - Chat")

# Panel lateral
with st.sidebar:
    st.header("Contexto del LLM")

    with st.expander("System prompt"):
        st.text(build_system_prompt())

    with st.expander("Ejemplos inyectados (CAG)"):
        st.text(format_examples_for_prompt(ESTIMATION_EXAMPLES))

# Streamlit reejecuta todo el script en cada interacción; session_state es lo único que persiste entre reruns
# st.session_state: es un dict-like que sobrevive entre reruns dentro de la misma sesión de navegador
if "messages" not in st.session_state:
    st.session_state.messages = []

# Repinta el historial acumulado en reruns anteriores (muestra los mensajes guardados en sesión: st.session_state)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Usamos markdown en lugar de st.text para que se renderice el contenido en formato Markdown
        st.markdown(message["content"])

# Devuelve texto solo en el rerun donde el usuario envía algo; el resto del tiempo es None
transcription = st.chat_input("Pega aquí la transcripción de la reunión...")

if transcription:
    st.session_state.messages.append({"role": "user", "content": transcription})
    with st.chat_message("user"):
        st.markdown(transcription)

    # stats se rellena como efecto lateral dentro del generador de streaming
    stats: dict = {}
    with st.chat_message("assistant"):
        start = time.perf_counter()
        estimation = st.write_stream(estimate_project_stream(transcription, stats))
        stats["elapsed_seconds"] = time.perf_counter() - start

    st.session_state.messages.append({"role": "assistant", "content": estimation})
    st.session_state.last_metrics = stats

# Segundo bloque de sidebar: se ejecuta después de calcular las métricas de esta consulta
with st.sidebar:
    st.subheader("Última llamada")
    metrics = st.session_state.get("last_metrics")
    if metrics:
        st.caption(f"**Modelo solicitado:** {metrics.get('model_requested', '-')}")
        st.caption(f"**Modelo usado:** {metrics.get('model_used', '-')}")
        st.caption(f"**Tokens entrada:** {metrics.get('tokens_input', '-')}")
        st.caption(f"**Tokens salida:** {metrics.get('tokens_output', '-')}")
        st.caption(f"**Tiempo:** {metrics.get('elapsed_seconds', 0):.2f} s")
    else:
        st.caption("Todavía no se ha generado ninguna estimación.")