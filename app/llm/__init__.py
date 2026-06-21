"""LLM access layer.

A thin factory around the local model integrations (Qwen via Ollama for chat /
vision, bge-m3 for embeddings) so the rest of the app never instantiates models
directly. Everything runs locally — no external API.
"""

from app.llm.client import get_chat_model, get_embeddings, get_vision_model, image_data_url

__all__ = ["get_chat_model", "get_vision_model", "get_embeddings", "image_data_url"]
