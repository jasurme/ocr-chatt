"""Tests for the FastAPI HTTP layer (Step 11). Offline via dependency overrides."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from app.api.store import DocumentStore
from app.chat import ChatService, CustomsRAG, DocumentQA
from app.chat.router import Route
from app.classification import Classification, DocumentClassifier
from app.extraction import StructuredExtractor
from app.graph import build_pipeline
from app.main import app
from app.ocr import OCRService
from app.rag import get_vectorstore
from app.schemas import InvoiceData
from app.schemas.types import DocumentType
from app.services import get_chat_service, get_document_store, get_pipeline
from tests._fakes import FakeChatModel, FakeOCRProvider, HashEmbeddings


@pytest.fixture
def client(tmp_path):
    # Fake document-processing pipeline (no network).
    classifier = DocumentClassifier(
        model=FakeChatModel(
            structured_response=Classification(
                doc_type=DocumentType.INVOICE, language="en", confidence=0.9, reasoning="x"
            )
        )
    )
    extractor = StructuredExtractor(
        model=FakeChatModel(structured_response=InvoiceData(invoice_number="INV-TEST", currency="EUR")),
        vision_model=FakeChatModel(structured_response=InvoiceData(invoice_number="INV-TEST", currency="EUR")),
    )
    pipeline = build_pipeline(
        ocr_service=OCRService(provider=FakeOCRProvider(), threshold=100),
        classifier=classifier,
        extractor=extractor,
    )

    # Fake chatbot (router always doc_qa; downgrades to general without a doc).
    vs = get_vectorstore(embeddings=HashEmbeddings(64), in_memory=True, vector_size=64, collection="api_kb")
    chat_service = ChatService(
        qa=DocumentQA(model=FakeChatModel(content="DOC ANSWER")),
        rag=CustomsRAG(vectorstore=vs, model=FakeChatModel(content="RAG ANSWER [Source 1]")),
        router_model=FakeChatModel(structured_response=Route(route="doc_qa", reasoning="x")),
        general_model=FakeChatModel(content="GENERAL ANSWER"),
        checkpointer=InMemorySaver(),
    )
    store = DocumentStore(tmp_path / "processed")

    app.dependency_overrides[get_pipeline] = lambda: pipeline
    app.dependency_overrides[get_chat_service] = lambda: chat_service
    app.dependency_overrides[get_document_store] = lambda: store
    yield TestClient(app)
    app.dependency_overrides.clear()


def _upload(client, samples, key="invoice_pdf2"):
    path = samples[key]
    with open(path, "rb") as fh:
        return client.post("/api/documents", files={"file": (path.name, fh, "application/octet-stream")})


# ------------------------------ basics ------------------------------------- #
def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "models" in body and body["models"]["vision"]


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


# ---------------------------- documents ------------------------------------ #
def test_upload_and_get_document(client, samples):
    r = _upload(client, samples)
    assert r.status_code == 200
    doc = r.json()
    assert doc["status"] == "completed"
    assert doc["doc_type"] == "invoice"
    assert doc["extracted"]["invoice_number"] == "INV-TEST"
    doc_id = doc["id"]

    r2 = client.get(f"/api/documents/{doc_id}")
    assert r2.status_code == 200 and r2.json()["id"] == doc_id

    r3 = client.get("/api/documents")
    assert any(d["id"] == doc_id for d in r3.json()["documents"])


def test_get_missing_document_404(client):
    assert client.get("/api/documents/nope").status_code == 404


def test_delete_document(client, samples):
    doc_id = _upload(client, samples).json()["id"]
    assert client.delete(f"/api/documents/{doc_id}").status_code == 200
    assert client.get(f"/api/documents/{doc_id}").status_code == 404  # gone
    assert client.delete(f"/api/documents/{doc_id}").status_code == 404  # already gone


def test_export_csv_and_xlsx(client, samples):
    doc_id = _upload(client, samples).json()["id"]

    rc = client.get(f"/api/documents/{doc_id}/export", params={"format": "csv"})
    assert rc.status_code == 200
    assert "text/csv" in rc.headers["content-type"]
    assert "INV-TEST" in rc.content.decode("utf-8-sig")

    rx = client.get(f"/api/documents/{doc_id}/export", params={"format": "xlsx"})
    assert rx.status_code == 200
    assert rx.content[:2] == b"PK"


def test_export_bad_format_422(client, samples):
    doc_id = _upload(client, samples).json()["id"]
    assert client.get(f"/api/documents/{doc_id}/export", params={"format": "pdf"}).status_code == 422


# ------------------------------- chat -------------------------------------- #
def test_chat_general_without_document(client):
    r = client.post("/api/chat", json={"message": "hello", "thread_id": "t1"})
    assert r.status_code == 200
    body = r.json()
    assert body["route"] == "general"  # downgraded (no doc loaded)
    assert body["answer"] == "GENERAL ANSWER"


def test_chat_doc_qa_with_document(client, samples):
    doc_id = _upload(client, samples).json()["id"]
    r = client.post(
        "/api/chat",
        json={"message": "What is the invoice number?", "thread_id": "t2", "document_id": doc_id},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["route"] == "doc_qa"
    assert body["answer"] == "DOC ANSWER"


def test_chat_with_missing_document_404(client):
    r = client.post("/api/chat", json={"message": "hi", "document_id": "nope"})
    assert r.status_code == 404


def test_chat_multi_document(client, samples):
    a = _upload(client, samples).json()["id"]
    b = _upload(client, samples).json()["id"]
    r = client.post(
        "/api/chat",
        json={"message": "compare these", "thread_id": "multi", "document_ids": [a, b]},
    )
    assert r.status_code == 200
    assert r.json()["route"] == "doc_qa"  # multiple docs still routes to doc_qa


def test_chat_multi_document_missing_404(client, samples):
    a = _upload(client, samples).json()["id"]
    r = client.post("/api/chat", json={"message": "x", "document_ids": [a, "nope"]})
    assert r.status_code == 404


def test_chat_history(client):
    client.post("/api/chat", json={"message": "hello", "thread_id": "hist"})
    r = client.get("/api/chat/hist/history")
    assert r.status_code == 200
    assert len(r.json()["messages"]) == 2
