# Borrador de correo de entrega (NO ENVIADO)

**Para:** george@lidr.co
**Asunto:** Entrega ejercicio sesión 10: búsqueda híbrida + reranking

---

Hola George:

Te envío la entrega del ejercicio de la sesión 10 (búsqueda full-text con `tsvector` + GIN, fusión
RRF, reranking con cross-encoder en patrón recall-then-rerank, y medición de A/B/C/D sobre un golden
set de 5 consultas).

PR: <URL_DEL_PR>

Resultados (ejecución real; 17 presupuestos / 60 chunks, recall 50 → top 5, RRF k=60, rama léxica
en modo OR, CPU):

| Config | Búsqueda | Reranking | Precision@5 | Latencia media (ms) | p50 | p95 |
| --- | --- | --- | --- | --- | --- | --- |
| A | Vectorial | No | 0.88 | 6.8 | 5.2 | 20.1 |
| B | Híbrida | No | 0.80 | 9.3 | 8.6 | 13.2 |
| C | Vectorial | Sí | 0.84 | 5514.0 | 6170.5 | 6541.7 |
| D | Híbrida | Sí | 0.84 | 6216.1 | 6171.7 | 6695.0 |

**Conclusión.** En este proyecto usaría la configuración A (vectorial, sin reranking) por defecto y
dejaría los conmutadores para volver a medir cuando cambie el corpus. A obtiene la mejor
Precision@5 (0,88) con 6,8 ms. La híbrida baja a 0,80: con OR su rama léxica casa ~27 de 60 chunks
por consulta y, en un corpus pequeño y homogéneo, aporta más ruido que señal. El reranking (0,84)
no justifica su latencia aquí: cuesta ~5,5–6,2 s por consulta en CPU (≈800× la búsqueda) y, en
conjunto, pierde 0,04 frente a A. Gana en la consulta industrial, pero pierde en la de e-commerce y
la de logística. Limitaciones: con 60 chunks, un recall de 50 es casi todo el corpus; solo hay 5
consultas (una diferencia de 0,04 es un chunk de 25 posiciones); la anotación tiene revisión humana
limitada; y en CPU un único chunk de 1.768 tokens triplica el coste del cross-encoder.

Dos decisiones que quizá quieras revisar: uso la configuración FTS `english` porque el corpus real
está en inglés, y la rama léxica usa semántica OR en lugar del AND de `plainto_tsquery`, porque con
AND una consulta larga devolvía 0 filas.

El detalle está en `docs/sessions/10-hybrid-reranking.md`.

Un saludo,
Javier
