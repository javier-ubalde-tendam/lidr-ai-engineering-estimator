# Sesión 13 — grafo de estimación (LangGraph)

Ejercicio: re-expresar el flujo de estimación como un grafo con estado, checkpointer Postgres y trazas Logfire.
Solución y decisiones: [docs/sessions/13-graph.md](../../docs/sessions/13-graph.md).

- `sample_transcript_complex.txt`: transcripción de entrada (copia de la de la sesión 12).
- `example_run_complex.txt`: informe de una ejecución real en Docker (Postgres + corpus ingerido). Las horas son de
  ingeniero. Esa ejecución no exportó traza a Logfire (no había `LOGFIRE_TOKEN` en `.env`).

Reproducir (dentro de Docker; la imagen incluye `exercises/`):

```powershell
docker compose exec ai_service python scripts/run_graph_s13.py exercises/session-13/sample_transcript_complex.txt --out exercises/session-13/example_run_complex.txt
```

Con `LOGFIRE_TOKEN` en `.env` (y el contenedor recreado) la misma ejecución exporta un span por nodo.
