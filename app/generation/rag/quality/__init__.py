"""Session 11 - generation quality: grounding guards around the RAG estimate.

Three techniques layered on top of the grounded generation, each switchable at runtime
(``RuntimeRetrievalConfig``):

* ``augmentation``: shapes the retrieved context BEFORE generation (extractive compression and
  edge-loading reorder against lost-in-the-middle).
* ``hallucination``: the semantic gate AFTER generation. A deterministic numeric anchor plus a
  strict judge grade each grounded line grounded / degraded / insufficient (a real citation does
  not prove the number is entailed by it).
* ``synthesis``: surfaces contradictory historical sources as an hour range with a reason instead
  of averaging the conflict into a misleading point.

The package only depends on ``foundation`` and on ``generation/rag`` modules (schemas). The LLM
wrapper and the runtime flags are injected by whoever composes it (``rag/estimator.py``, built in
``app/dependencies.py``), so nothing here reaches for the composition root.
"""
