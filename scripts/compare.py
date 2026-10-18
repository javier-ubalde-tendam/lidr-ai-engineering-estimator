import argparse
import math
import sys
from pathlib import Path

from pydantic import ValidationError

# Al ejecutar `python scripts/compare.py`, sys.path[0] es scripts/, no la raíz del repo, y
# `from app...` fallaría. Insertamos la raíz (en Java lo resolvería el classpath).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.dependencies import get_openai_client
from app.embedding_pipeline.embedder import OpenAIEmbedder

MISSING_KEY_MESSAGE = "Error: falta OPENAI_API_KEY (defínela en el .env de la raíz del repo)"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Producto escalar / (norma_a * norma_b), solo con la biblioteca estándar."""
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / (norm_a * norm_b)


def main() -> int:
    parser = argparse.ArgumentParser(description="Similitud coseno entre dos textos")
    parser.add_argument("--text-a", required=True)
    parser.add_argument("--text-b", required=True)
    args = parser.parse_args()

    try:
        settings = get_settings()
    except ValidationError:
        # Con LLM_PROVIDER=openai, Settings ya se niega a construirse sin la clave
        print(MISSING_KEY_MESSAGE, file=sys.stderr)
        return 1
    if not settings.OPENAI_API_KEY:
        print(MISSING_KEY_MESSAGE, file=sys.stderr)
        return 1

    embedder = OpenAIEmbedder(client=get_openai_client(), model=settings.EMBEDDING_MODEL)
    similarity = cosine_similarity(embedder.embed_one(args.text_a), embedder.embed_one(args.text_b))

    print(f"Text A: {args.text_a}")
    print(f"Text B: {args.text_b}")
    print(f"Cosine similarity: {similarity:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
