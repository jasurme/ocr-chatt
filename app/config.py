"""Central application configuration.

All tunables live here and are overridable via environment variables / `.env`.
The stack is fully local per the assignment: PaddleOCR (OCR), Qwen via Ollama
(LLM + vision), bge-m3 via Ollama (embeddings), and Qdrant (vector store).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root (…./chat-ocr), independent of the current working directory.
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Typed, validated settings loaded from environment / `.env`."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- App ----
    app_name: str = "Customs Document Intelligence"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    debug: bool = False

    # ---- LLM (local Qwen via Ollama) ----
    ollama_base_url: str = "http://localhost:11434"
    # Text model for chat / classification / extraction / RAG. qwen2.5:7b is fast
    # and supports tool-calling; the vision model handles image-only documents.
    ollama_chat_model: str = "qwen2.5:7b"
    ollama_vision_model: str = "qwen2.5vl:7b"
    ollama_request_timeout: float = 300.0
    # Master switch for using the multimodal model to read page images. Documents
    # are read with PaddleOCR first; vision is a fallback for image-only / low-OCR
    # pages (classification + extraction). Set false for a pure text + OCR pipeline.
    enable_vision: bool = True

    # ---- Embeddings (local, RAG) ----
    ollama_embed_model: str = "bge-m3"  # multilingual (uz/ru/en)
    embedding_dim: int = 1024  # bge-m3 vector size (Qdrant collection size)

    # ---- OCR ----
    ocr_provider: str = "paddle"  # paddle (mandated)
    paddle_lang: str = "en"  # PaddleOCR recognition language (en | ru | latin | cyrillic ...)
    paddle_use_textline_orientation: bool = False

    # Extract structured data from page images with the vision model (Qwen-VL)
    # when available; falls back to the OCR text otherwise.
    prefer_vision_extraction: bool = True
    # Max page images sent to the vision model. Kept modest so the request stays
    # within the local model's context window; vision falls back to text-only
    # extraction if it still overflows (see StructuredExtractor.extract).
    extraction_max_images: int = 4
    extraction_vision_text_chars: int = 6000  # OCR-text budget WHEN images are also sent
    max_prompt_chars: int = 24000  # max OCR chars embedded into a (text-only) prompt
    # Staged text extraction splits a big schema into smaller focused calls (scalars,
    # parties, then each list/table) — a small local model returns empty lists for a
    # large nested schema, but fills the focused ones. This caps rows per list field.
    extraction_max_items: int = 200

    # ---- Vector store (RAG) ----
    vector_store: str = "qdrant"
    qdrant_url: str = ""  # e.g. http://localhost:6333; empty => embedded/local path
    qdrant_path: str = str(ROOT_DIR / "data" / "qdrant")  # embedded fallback
    qdrant_collection: str = "customs_law"

    # ---- Paths ----
    data_dir: Path = ROOT_DIR / "data"
    upload_dir: Path = ROOT_DIR / "data" / "uploads"
    processed_dir: Path = ROOT_DIR / "data" / "processed"
    # Durable chat history (LangGraph SQLite checkpointer): conversations + the
    # active document context survive a server restart.
    chat_db_path: Path = ROOT_DIR / "data" / "chat.sqlite"

    # ---- Ingestion / OCR tuning ----
    pdf_render_dpi: int = 200  # DPI when rasterizing PDF pages for OCR
    # If a PDF page yields at least this many extractable chars, treat it as
    # "searchable" and skip image OCR for that page.
    searchable_text_threshold: int = 100
    max_pages: int = 25  # safety cap on pages processed per document

    # ---- RAG tuning ----
    lexuz_language: str = "uz"  # which lex.uz Customs Code to scrape: uz | ru | en
    rag_chunk_size: int = 1200
    rag_chunk_overlap: int = 150
    rag_top_k: int = 5
    # Hybrid retrieval: semantic (dense) + keyword (full-text) results, merged.
    rag_semantic_k: int = 7
    rag_keyword_k: int = 10

    def ensure_dirs(self) -> None:
        """Create runtime data directories if missing."""
        for d in (self.data_dir, self.upload_dir, self.processed_dir):
            Path(d).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()
