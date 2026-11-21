# Resultados sesión 10: búsqueda híbrida + reranking

Generado por `scripts/eval_retrieval_s10.py` el 2026-11-08T19:04:49+00:00. No editar a mano.

| Config | Búsqueda | Reranking | Precision@5 | Latencia media (ms) | p50 | p95 |
| --- | --- | --- | --- | --- | --- | --- |
| A | Vectorial | No | 0.88 | 6.8 | 5.2 | 20.1 |
| B | Híbrida | No | 0.80 | 9.3 | 8.6 | 13.2 |
| C | Vectorial | Sí | 0.84 | 5514.0 | 6170.5 | 6541.7 |
| D | Híbrida | Sí | 0.84 | 6216.1 | 6171.7 | 6695.0 |

Latencia media del embedding de la consulta (común a todas, no incluida arriba): 274.8 ms

## Precision@5 por consulta

| Consulta | A | B | C | D |
| --- | --- | --- | --- | --- |
| q1-ecommerce-checkout-trap | 1.00 | 0.80 | 0.80 | 0.80 |
| q2-finance-lending | 0.80 | 0.80 | 0.80 | 0.80 |
| q3-healthcare-telemedicine | 0.80 | 0.80 | 0.80 | 0.80 |
| q4-industrial-iot-maintenance | 0.80 | 0.60 | 1.00 | 1.00 |
| q5-logistics-warehouse | 1.00 | 1.00 | 0.80 | 0.80 |

## lexical_hits medios por consulta (configuraciones híbridas)

- B: 26.6
- D: 26.6

## Metadatos

- date: 2026-11-08T19:04:49+00:00
- documents: 17
- chunks: 60
- queries: 5
- k: 5
- recall_k: 50
- rrf_k: 60
- lexical_match_mode: or
- reranker_model: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
- distance_threshold: 2.0
- measured_runs_per_query: 5
- embedding_latency_mean_ms: 274.8482207998677
- device: cpu
- cpu: x86_64 x12 (Linux)
