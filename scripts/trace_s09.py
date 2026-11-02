"""Trace manual de la Sesión 09: embedding de una transcripción y búsqueda top-k en /search.

Solo cliente: no añade lógica de negocio. Requiere el stack levantado y OPENAI_API_KEY en .env.
Uso: uv run python scripts/trace_s09.py examples/transcripts/02_ambiguous.txt
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import httpx

# Al ejecutar `python scripts/...`, sys.path[0] es scripts/, no la raíz del repo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.dependencies import get_embedder

K = 5
PREVIEW_CHARS = 120
SEARCH_TIMEOUT = httpx.Timeout(60.0, connect=10.0)


def preview(content: str) -> str:
    # Una línea por resultado: los saltos de línea del chunk romperían la tabla
    return " ".join(content.split())[:PREVIEW_CHARS]


def step1_embedding(text: str) -> None:
    vector = get_embedder().embed_one(text)
    norm = math.sqrt(sum(x * x for x in vector))
    print("== Paso 1: embedding de la transcripción completa ==")
    print(f"chars: {len(text)}")
    print(f"dimensionality: {len(vector)}")
    print(f"l2_norm: {norm:.6f}")
    print(f"first_component: {vector[0]!r}")
    print(f"last_component: {vector[-1]!r}")


def step2_search(text: str) -> int:
    base_url = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
    print(f"\n== Paso 2: POST {base_url}/search (k={K}) ==")
    response = httpx.post(f"{base_url}/search", json={"query": text, "k": K}, timeout=SEARCH_TIMEOUT)
    if response.status_code != 200:
        # El error es un hallazgo del trace: se muestra tal cual, sin intentar arreglarlo
        print(f"HTTP {response.status_code}")
        print(response.text)
        return 1

    body = response.json()
    print(json.dumps(body, indent=2, ensure_ascii=False))

    print("\n== Resumen ==")
    print("rank | chunk_id | distance | client_sector | budget_id | component_id | preview")
    for rank, result in enumerate(body["results"], start=1):
        meta = result["metadata"]
        print(
            f"{rank} | {result['chunk_id']} | {result['distance']:.4f} | "
            f"{meta.get('client_sector')} | {meta.get('budget_id')} | "
            f"{meta.get('component_id')} | {preview(result['content'])}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace S09: embedding + /search de una transcripción")
    parser.add_argument("transcript", type=Path, help="Fichero de texto con la transcripción")
    args = parser.parse_args()

    # Consola de Windows (cp1252) no siempre admite los caracteres de las transcripciones
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    text = args.transcript.read_text(encoding="utf-8")
    step1_embedding(text)
    return step2_search(text)


if __name__ == "__main__":
    sys.exit(main())
