"""End-to-end integration tests over the real HTTP app (Step 12).

These use the real local services (PaddleOCR + Qwen via Ollama + Qdrant), so they
are marked `integration` and auto-skip unless Ollama is reachable and
`-m integration` is passed. They mirror the full journey: upload -> Q&A -> RAG -> export.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from app.main import app
from conftest import requires_ollama

pytestmark = [requires_ollama, pytest.mark.integration]


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_e2e_upload_classify_extract(client, samples):
    with open(samples["awb_jpg"], "rb") as fh:
        r = client.post("/api/documents", files={"file": ("Avia.jpg", fh, "image/jpeg")})
    assert r.status_code == 200
    doc = r.json()
    assert doc["status"] == "completed"
    assert doc["doc_type"] == "air_waybill"
    assert doc["extracted"]["awb_number"]
    pytest.doc_id = doc["id"]


def test_e2e_document_qa(client):
    doc_id = getattr(pytest, "doc_id", None)
    assert doc_id, "upload test must run first"
    r = client.post(
        "/api/chat",
        json={"message": "What is the destination airport?", "thread_id": "e2e", "document_id": doc_id},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["route"] == "doc_qa"
    assert "TASHKENT" in body["answer"].upper()


def test_e2e_export_excel(client):
    doc_id = getattr(pytest, "doc_id", None)
    r = client.get(f"/api/documents/{doc_id}/export", params={"format": "xlsx"})
    assert r.status_code == 200
    assert r.content[:2] == b"PK"


def test_e2e_rag_after_seed(client):
    seed = client.post("/api/kb/seed", json={"languages": ["en"], "limit": 30, "reset": True})
    assert seed.status_code == 200 and seed.json()["count"] > 0
    r = client.post(
        "/api/chat",
        json={"message": "What is the customs territory of Uzbekistan?", "thread_id": "e2e-rag"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["route"] == "rag"
    assert body["sources"]
