"""Tests for the local LLM / embeddings factories (Qwen + bge-m3 via Ollama)."""

from __future__ import annotations

import pytest
from langchain_ollama import ChatOllama, OllamaEmbeddings

import app.llm.client as client
from app.config import Settings
from conftest import requires_ollama


def _use(monkeypatch, **overrides):
    """Point the client factories at a Settings built with the given overrides."""
    monkeypatch.setattr(client, "get_settings", lambda: Settings(**overrides))


# ----------------------------- factories ----------------------------------- #
def test_chat_model_is_local_qwen(monkeypatch):
    _use(monkeypatch, ollama_chat_model="qwen2.5vl:7b")
    m = client.get_chat_model()
    assert isinstance(m, ChatOllama)
    assert m.model == "qwen2.5vl:7b"


def test_vision_model_is_local_qwen(monkeypatch):
    _use(monkeypatch, ollama_vision_model="qwen2.5vl:7b")
    m = client.get_vision_model()
    assert isinstance(m, ChatOllama)
    assert m.model == "qwen2.5vl:7b"


def test_embeddings_are_local(monkeypatch):
    _use(monkeypatch, ollama_embed_model="bge-m3")
    e = client.get_embeddings()
    assert isinstance(e, OllamaEmbeddings)
    assert e.model == "bge-m3"


def test_structured_output_available(monkeypatch):
    # The classifier/extractor rely on with_structured_output + bind_tools.
    _use(monkeypatch)
    m = client.get_chat_model()
    assert hasattr(m, "with_structured_output") and hasattr(m, "bind_tools")


def test_embedding_dim_default():
    assert Settings().embedding_dim == 1024  # bge-m3


# --------------------------- live integration ------------------------------ #
@requires_ollama
@pytest.mark.integration
def test_ollama_live_roundtrip():
    """Real local Qwen round-trip — skips automatically if the model isn't pulled."""
    s = Settings()
    model = ChatOllama(model=s.ollama_chat_model, base_url=s.ollama_base_url, temperature=0)
    try:
        resp = model.invoke("Reply with the single word: OK")
    except Exception as exc:
        pytest.skip(f"Ollama model '{s.ollama_chat_model}' unavailable: {exc}")
    assert isinstance(resp.content, str) and resp.content.strip()
