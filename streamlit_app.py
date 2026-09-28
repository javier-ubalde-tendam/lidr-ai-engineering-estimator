import time

import streamlit as st

from app.context.examples import ESTIMATION_EXAMPLES, format_examples_for_prompt
from app.services.llm_service import build_system_prompt, estimate_project, estimate_project_stream

st.title("Estimador de Proyectos - Chat")

# Panel lateral
with st.sidebar:
    st.header("Contexto del LLM")

    with st.expander("System prompt"):
        st.text(build_system_prompt())

    with st.expander("Ejemplos inyectados (CAG)"):
        st.text(format_examples_for_prompt(ESTIMATION_EXAMPLES))

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

transcription = st.chat_input("Pega aquí la transcripción de la reunión...")

if transcription:
    st.session_state.messages.append({"role": "user", "content": transcription})
    with st.chat_message("user"):
        st.markdown(transcription)

    stats: dict = {}
    with st.chat_message("assistant"):
        start = time.perf_counter()
        estimation = st.write_stream(estimate_project_stream(transcription, stats))
        stats["elapsed_seconds"] = time.perf_counter() - start

    st.session_state.messages.append({"role": "assistant", "content": estimation})
    st.session_state.last_metrics = stats

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