"""Ingesta data/budgets_sample.json vía HTTP y ejecuta cinco búsquedas de ejemplo (sesión 08).

Requiere el stack levantado (`docker compose up --build -d`). Uso:
    docker compose run --rm ai_service python query_examples.py
    docker compose exec -T ai_service python query_examples.py --output /app/output_examples.txt
"""

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
BUDGETS_FILE = "data/budgets_sample.json"
DOCUMENT_TYPE = "historical_budget"
K = 5
PREVIEW_CHARS = 120

# Cada consulta ejemplifica una categoría; el resultado real se contrasta con el corpus
QUERIES: list[tuple[str, str]] = [
    ("Match directo", "REST API development with JWT authentication for financial sector"),
    (
        "Reformulación",
        "secure backend service with token-based access control for banking applications",
    ),
    ("Fuera de dominio", "mobile application for restaurant reservations"),
    ("Ambigua", "integration with external system"),
    (
        "Muy específica",
        "migration from monolith to microservices architecture using Kubernetes",
    ),
]

INGEST_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
SEARCH_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


class Report:
    """Acumula las líneas del informe: lo mismo se imprime y (opcional) se guarda en un fichero."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, line: str = "") -> None:
        self.lines.append(line)
        print(line)


def preview(content: str) -> str:
    # Los saltos de línea del chunk romperían el formato de una línea por resultado
    return " ".join(content.split())[:PREVIEW_CHARS]


def ingest_budgets(client: httpx.Client, base_url: str, report: Report) -> bool:
    budgets = json.loads((ROOT / BUDGETS_FILE).read_text(encoding="utf-8"))
    ingested = already_present = 0
    failures: list[str] = []
    for budget in budgets:
        source_path = f"{BUDGETS_FILE}::{budget['budget_id']}"
        response = client.post(
            f"{base_url}/embeddings/ingest",
            json={"source_path": source_path, "document_type": DOCUMENT_TYPE, "content": budget},
            timeout=INGEST_TIMEOUT,
        )
        if response.status_code == 200:
            ingested += 1
        elif response.status_code == 409:
            already_present += 1  # Ya ingestado: la ejecución es idempotente
        else:
            failures.append(f"{source_path}: HTTP {response.status_code} {response.text[:200]}")

    report.add(f"Ingesta: {len(budgets)} presupuestos en {BUDGETS_FILE}")
    report.add(f"  nuevos: {ingested} | ya existentes (409): {already_present} | fallos: {len(failures)}")
    for failure in failures:
        print(f"ERROR: {failure}", file=sys.stderr)
    return not failures


def run_searches(client: httpx.Client, base_url: str, report: Report) -> bool:
    for number, (purpose, query) in enumerate(QUERIES, start=1):
        response = client.post(
            f"{base_url}/search", json={"query": query, "k": K}, timeout=SEARCH_TIMEOUT
        )
        if response.status_code != 200:
            print(f"ERROR: /search HTTP {response.status_code}: {response.text[:200]}", file=sys.stderr)
            return False
        body = response.json()
        report.add()
        report.add(f"Consulta {number}/{len(QUERIES)} - {purpose}")
        report.add(f"  query: {query}")
        report.add(f"  tiempo: {body['search_time_ms']} ms | k={body['k']}")
        for rank, result in enumerate(body["results"], start=1):
            report.add(
                f"  {rank}. chunk_id={result['chunk_id']} distance={result['distance']:.4f} "
                f"type={result['chunk_type']} | {preview(result['content'])}"
            )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingesta del corpus y cinco búsquedas de ejemplo")
    parser.add_argument("--output", help="Fichero UTF-8 donde guardar el mismo informe")
    parser.add_argument(
        "--api-base-url",
        default=os.environ.get("API_BASE_URL", "http://localhost:8000"),
        help="Por defecto API_BASE_URL (Docker: http://ai_service:8000; host: http://localhost:8000)",
    )
    args = parser.parse_args()
    base_url = args.api_base_url.rstrip("/")

    # Consola de Windows (cp1252) no siempre admite los caracteres del informe
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    report = Report()
    report.add(f"Backend: {base_url}")
    try:
        with httpx.Client() as client:
            if not ingest_budgets(client, base_url, report):
                return 1
            if not run_searches(client, base_url, report):
                return 1
    except httpx.HTTPError as exc:
        print(f"ERROR: no se pudo contactar con {base_url}: {type(exc).__name__}", file=sys.stderr)
        return 1

    if args.output:
        Path(args.output).write_text("\n".join(report.lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
