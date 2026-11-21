"""API-key auth tests: the 401/200 boundary, key independence and the disabled-router rule."""

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.api.security import require_estimate_key, require_retrieval_key
from app.main import app

SEARCH_BODY = {"query_text": "ecommerce storefront with card checkout"}
ESTIMATE_KEY = "test-estimate-key"  # same fake value as tests/conftest.py


@pytest.fixture
def client(retrieval_stubs: None) -> TestClient:
    return TestClient(app)


def test_retrieval_requires_a_key(client: TestClient) -> None:
    response = client.post("/v1/retrieval/search", json=SEARCH_BODY)

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "X-API-Key"


def test_retrieval_accepts_its_own_key(client: TestClient, retrieval_api_key: str) -> None:
    response = client.post(
        "/v1/retrieval/search", json=SEARCH_BODY, headers={"X-API-Key": retrieval_api_key}
    )

    assert response.status_code == 200


@pytest.mark.parametrize("wrong_key", ["nope", "test-retrieval-ke", "clave-ñandú"])
def test_wrong_key_is_rejected(client: TestClient, wrong_key: str) -> None:
    # Non-ASCII keys must be a clean 401, not a 500 from compare_digest
    response = client.post(
        "/v1/retrieval/search", json=SEARCH_BODY, headers={"X-API-Key": wrong_key.encode()}
    )

    assert response.status_code == 401


def test_unconfigured_key_disables_the_router(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An empty configured key must never match, not even an empty or arbitrary header
    monkeypatch.setenv("RETRIEVAL_API_KEY", "")

    for headers in ({}, {"X-API-Key": ""}, {"X-API-Key": "anything"}):
        response = client.post("/v1/retrieval/search", json=SEARCH_BODY, headers=headers)
        assert response.status_code == 401


def test_the_two_keys_are_independent(
    monkeypatch: pytest.MonkeyPatch, retrieval_api_key: str
) -> None:
    secured = FastAPI()

    @secured.get("/retrieval", dependencies=[Depends(require_retrieval_key)])
    def retrieval() -> dict[str, str]:
        return {"ok": "retrieval"}

    @secured.get("/estimate", dependencies=[Depends(require_estimate_key)])
    def estimate() -> dict[str, str]:
        return {"ok": "estimate"}

    client = TestClient(secured)
    retrieval_headers = {"X-API-Key": retrieval_api_key}
    estimate_headers = {"X-API-Key": ESTIMATE_KEY}

    assert client.get("/retrieval", headers=retrieval_headers).status_code == 200
    assert client.get("/estimate", headers=estimate_headers).status_code == 200
    assert client.get("/estimate", headers=retrieval_headers).status_code == 401
    assert client.get("/retrieval", headers=estimate_headers).status_code == 401

    monkeypatch.setenv("ESTIMATE_API_KEY", "")
    assert client.get("/estimate", headers=estimate_headers).status_code == 401
