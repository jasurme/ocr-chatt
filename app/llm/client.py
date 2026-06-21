"""Factory functions for chat / vision / embedding models.

Fully local: chat, classification and vision extraction are served by **Qwen**
via **Ollama**; embeddings by a local model (bge-m3). The rest of the app calls
these factories and never references a specific model or vendor.
"""

from __future__ import annotations

import base64

from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.config import get_settings


def get_chat_model(model: str | None = None, temperature: float = 0.0, **kwargs):
    """Return the local chat model (Qwen via Ollama)."""
    s = get_settings()
    return ChatOllama(
        model=model or s.ollama_chat_model,
        base_url=s.ollama_base_url,
        temperature=temperature,
        client_kwargs={"timeout": s.ollama_request_timeout},
        **kwargs,
    )


def get_vision_model(temperature: float = 0.0, **kwargs):
    """Return the local vision-capable model (Qwen-VL via Ollama)."""
    s = get_settings()
    return ChatOllama(
        model=s.ollama_vision_model,
        base_url=s.ollama_base_url,
        temperature=temperature,
        client_kwargs={"timeout": s.ollama_request_timeout},
        **kwargs,
    )


def get_embeddings():
    """Return the local embeddings model (bge-m3 via Ollama)."""
    s = get_settings()
    return OllamaEmbeddings(model=s.ollama_embed_model, base_url=s.ollama_base_url)


def image_data_url(image_png: bytes, mime: str = "image/png") -> str:
    """Encode raw image bytes as a base64 data URL for multimodal messages.

    ChatOllama accepts these ``image_url`` content blocks and strips the prefix.
    """
    b64 = base64.b64encode(image_png).decode("ascii")
    return f"data:{mime};base64,{b64}"
