"""Gate check: does the cross-encoder download, load and rank a control pair correctly?

Run it BEFORE relying on reranking::

    docker compose exec ai_service python -m app.generation.rag.retrieval.verify_reranker

Exit codes: 0 = ready, 1 = the model failed to load/score, 2 = it loaded but ranked wrongly.
"""

import sys

from app.generation.rag.retrieval.reranker import CrossEncoderReranker

QUERY = "E-commerce checkout with shopping cart, online payments and order management"
RELEVANT_DOCUMENT = (
    "Online store checkout flow: shopping cart, payment processing and order lifecycle."
)
IRRELEVANT_DOCUMENT = (
    "Hospital patient appointment scheduling and telemedicine video consultations."
)

EXIT_OK = 0
EXIT_LOAD_FAILED = 1
EXIT_WRONG_RANKING = 2


def main() -> int:
    reranker = CrossEncoderReranker.from_settings()
    print(f"Loading reranker model: {reranker.model_name}")
    try:
        relevant, irrelevant = reranker.score(QUERY, [RELEVANT_DOCUMENT, IRRELEVANT_DOCUMENT])
    except Exception as exc:  # noqa: BLE001 - any load/score failure must become a readable exit
        print(f"FAILED to load or run the reranker: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_LOAD_FAILED

    print(f"Query: {QUERY!r}")
    print(f"Relevant document score:   {relevant:.4f}")
    print(f"Irrelevant document score: {irrelevant:.4f}")
    if relevant <= irrelevant:
        print(
            "WRONG RANKING: the model loaded but did not score the relevant document higher. "
            "Check RERANKER_MODEL.",
            file=sys.stderr,
        )
        return EXIT_WRONG_RANKING

    print("OK: reranker loaded and ranked the relevant document first.")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
