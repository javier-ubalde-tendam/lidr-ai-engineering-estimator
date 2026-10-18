# Sanity check de embeddings

- **Modelo:** `text-embedding-3-small` (1536 dimensiones, sin `dimensions`)
- **Fecha de ejecución:** 2026-10-18
- **Comando:** `uv run python scripts/compare.py --text-a "..." --text-b "..."`

| Pareja | Texto A | Texto B | Coseno | Expectativa |
|--------|---------|---------|--------|-------------|
| A (cercanos) | OAuth 2.0 authentication backend with JWT tokens for fintech mobile app | Authorization service using JSON Web Tokens for a banking application | 0.5957 | > 0.6 (orientativo) |
| B (no relacionados) | OAuth 2.0 authentication backend with JWT tokens for fintech mobile app | Database migration from MySQL to PostgreSQL with zero downtime | 0.1920 | < 0.4 (orientativo) |
| C (genéricos) | Backend services | API development | 0.5407 | sin expectativa fija |

## Comentario

El orden encaja con la intuición: los textos cercanos puntúan claramente más que los no
relacionados (0.60 frente a 0.19). Lo que llama la atención es que la pareja A queda en 0.5957,
rozando por debajo del umbral orientativo de 0.6: dos frases casi sinónimas con vocabulario
distinto (OAuth/JWT frente a "Authorization"/"JSON Web Tokens") no superan ese listón, así que
conviene tomar los umbrales como orientación y no como regla. Lo más sorprendente es la pareja C:
dos textos genéricos y casi sin contenido alcanzan 0.5407, muy cerca de la pareja A. Con tan poco
texto el embedding refleja un "tema" difuso (software/backend), no una intención concreta. Por
eso el chunker antepone al componente el contexto del presupuesto (proyecto, sector, año,
tecnología): así un componente genérico como "Backend services" queda anclado a su proyecto y se
distingue de otro igual de genérico en un proyecto distinto.
