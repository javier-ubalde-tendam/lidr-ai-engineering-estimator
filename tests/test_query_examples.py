import importlib.util
import json
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent


def load_module():
    # query_examples.py vive en la raíz y no es un paquete: se carga por ruta
    spec = importlib.util.spec_from_file_location("query_examples", ROOT / "query_examples.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


qe = load_module()


def test_preview_normalizes_newlines_and_truncates() -> None:
    text = "line1\nline2\n\n" + "x" * 500

    preview = qe.preview(text)

    assert "\n" not in preview
    assert len(preview) == 120
    assert preview.startswith("line1 line2 xxx")


def test_ingest_posts_each_budget_and_treats_409_as_existing() -> None:
    budgets = json.loads((ROOT / "data" / "budgets_sample.json").read_text(encoding="utf-8"))
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(body)
        if len(sent) % 2:
            return httpx.Response(200, json={})
        return httpx.Response(409, json={"detail": "Document already ingested", "document_id": 1})

    report = qe.Report()
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        ok = qe.ingest_budgets(client, "http://test", report)

    assert ok is True
    assert len(sent) == len(budgets) == 15
    assert sent[0]["source_path"] == f"data/budgets_sample.json::{budgets[0]['budget_id']}"
    assert sent[0]["document_type"] == "historical_budget"
    assert sent[0]["content"] == budgets[0]


def test_ingest_reports_failure_on_other_statuses() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "Database error"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert qe.ingest_budgets(client, "http://test", qe.Report()) is False


def test_run_searches_prints_the_five_queries_with_k5() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "k": 5,
                "search_time_ms": 12,
                "results": [
                    {
                        "chunk_id": 7,
                        "distance": 0.23119,
                        "chunk_type": "budget_component",
                        "content": "a\nb",
                    }
                ],
            },
        )

    report = qe.Report()
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert qe.run_searches(client, "http://test", report) is True

    assert [r["k"] for r in requests] == [5] * 5
    assert len(requests) == 5
    assert any("chunk_id=7 distance=0.2312 type=budget_component | a b" in line for line in report.lines)
