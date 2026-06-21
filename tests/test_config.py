"""Tests for application configuration (Step 1: scaffolding)."""

from __future__ import annotations

from pathlib import Path

from app.config import ROOT_DIR, Settings, get_settings


def test_settings_load():
    s = get_settings()
    assert s.app_name
    assert s.ollama_chat_model
    assert s.ollama_vision_model
    assert s.ollama_embed_model == "bge-m3"


def test_settings_is_cached_singleton():
    assert get_settings() is get_settings()


def test_root_dir_points_at_repo():
    assert (ROOT_DIR / "requirements.txt").exists()
    assert (ROOT_DIR / "sample_files").is_dir()


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "qwen2.5:14b")
    monkeypatch.setenv("OCR_PROVIDER", "paddle")
    s = Settings()  # fresh instance reads current env
    assert s.ollama_chat_model == "qwen2.5:14b"
    assert s.ocr_provider == "paddle"


def test_ensure_dirs_creates_runtime_dirs(tmp_path, monkeypatch):
    s = Settings()
    s.data_dir = tmp_path / "data"
    s.upload_dir = tmp_path / "data" / "uploads"
    s.processed_dir = tmp_path / "data" / "processed"
    s.ensure_dirs()
    assert Path(s.upload_dir).is_dir()
    assert Path(s.processed_dir).is_dir()


def test_local_defaults():
    s = Settings()
    assert s.ocr_provider == "paddle"
    assert s.vector_store == "qdrant"
    assert s.ollama_base_url.startswith("http")
    assert s.embedding_dim == 1024  # bge-m3
