"""Cached application service singletons (pipeline, chatbot, store, knowledge base).

Building these is cheap — the models are created lazily on first use — so
importing/serving the app makes no network calls. The vector store is a
singleton because the embedded (on-disk) Qdrant allows only one client at a time.
"""

from __future__ import annotations

from functools import lru_cache

from app.api.store import DocumentStore
from app.chat import ChatService, CustomsRAG
from app.config import get_settings
from app.graph import build_pipeline
from app.llm import get_chat_model


@lru_cache(maxsize=1)
def get_pipeline():
    return build_pipeline()


@lru_cache(maxsize=1)
def get_knowledge_base():
    """Shared Qdrant vector store (single client for the embedded on-disk DB)."""
    from app.rag.vectorstore import get_vectorstore

    return get_vectorstore()


@lru_cache(maxsize=1)
def get_checkpointer():
    """Durable LangGraph checkpointer (SQLite) — chat history and the active document
    context persist across server restarts, keyed by ``thread_id``."""
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    settings = get_settings()
    settings.ensure_dirs()
    # check_same_thread=False: FastAPI serves sync routes from a thread pool.
    conn = sqlite3.connect(str(settings.chat_db_path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    # Wire the chatbot's RAG to the shared knowledge base, resolved lazily.
    # Lazy wiring means a knowledge-base problem (e.g. embedded-DB lock) never
    # breaks Document Q&A or general chat — only RAG degrades gracefully.
    return ChatService(
        rag=CustomsRAG(
            vectorstore_provider=get_knowledge_base, model=get_chat_model(streaming=True)
        ),
        checkpointer=get_checkpointer(),
    )


@lru_cache(maxsize=1)
def get_document_store() -> DocumentStore:
    settings = get_settings()
    settings.ensure_dirs()
    return DocumentStore(settings.processed_dir)
