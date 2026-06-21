"""Qdrant vector store wiring (local embedded path by default, server via URL).

Embedded mode (default) stores vectors under ``data/qdrant`` — zero infra, great
for a demo. Set ``QDRANT_URL`` (e.g. http://localhost:6333, via docker compose)
to use a server instance instead.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import get_settings
from app.llm import get_embeddings


# One client per (path|url) per process. Embedded Qdrant takes an exclusive
# file lock, so re-using a single client avoids "already accessed by another
# instance" errors when several call sites open the store within one process.
_CLIENT_CACHE: dict[str, QdrantClient] = {}


def build_client(in_memory: bool = False) -> QdrantClient:
    s = get_settings()
    if in_memory:
        return QdrantClient(location=":memory:")  # isolated; never cached
    key = s.qdrant_url or f"path::{s.qdrant_path}"
    client = _CLIENT_CACHE.get(key)
    if client is None:
        if s.qdrant_url:
            client = QdrantClient(url=s.qdrant_url)
        else:
            Path(s.qdrant_path).mkdir(parents=True, exist_ok=True)
            client = QdrantClient(path=s.qdrant_path)
        _CLIENT_CACHE[key] = client
    return client


def reset_client_cache() -> None:
    """Close and forget cached clients (used by tests / the seeding CLI)."""
    for client in _CLIENT_CACHE.values():
        try:
            client.close()
        except Exception:
            pass
    _CLIENT_CACHE.clear()


def get_vectorstore(
    embeddings=None,
    client: QdrantClient | None = None,
    collection: str | None = None,
    vector_size: int | None = None,
    in_memory: bool = False,
) -> QdrantVectorStore:
    """Return a QdrantVectorStore, creating the collection if needed."""
    s = get_settings()
    client = client or build_client(in_memory=in_memory)
    collection = collection or s.qdrant_collection
    embeddings = embeddings if embeddings is not None else get_embeddings()
    # Vector size must match the active embeddings model (bge-m3 = 1024).
    size = vector_size or s.embedding_dim

    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=size, distance=Distance.COSINE),
        )
    return QdrantVectorStore(client=client, collection_name=collection, embedding=embeddings)


def collection_count(vectorstore: QdrantVectorStore) -> int:
    """Number of points currently stored in the collection."""
    return vectorstore.client.count(vectorstore.collection_name).count


def _content_key(vs: QdrantVectorStore) -> str:
    return getattr(vs, "content_payload_key", "page_content")


def _metadata_key(vs: QdrantVectorStore) -> str:
    return getattr(vs, "metadata_payload_key", "metadata")


def _is_local(client: QdrantClient) -> bool:
    try:
        from qdrant_client.local.qdrant_local import QdrantLocal

        return isinstance(getattr(client, "_client", None), QdrantLocal)
    except Exception:
        return False


def ensure_keyword_index(vs: QdrantVectorStore) -> None:
    """Create a full-text payload index on the content field (server Qdrant only).

    Local/embedded Qdrant ignores payload indexes, so there the keyword search
    falls back to an in-process token scan (`keyword_search`).
    """
    if _is_local(vs.client):
        return
    from qdrant_client import models

    try:
        vs.client.create_payload_index(
            collection_name=vs.collection_name,
            field_name=_content_key(vs),
            field_schema=models.TextIndexParams(
                type="text",
                tokenizer=models.TokenizerType.WORD,
                min_token_len=2,
                lowercase=True,
            ),
        )
    except Exception:
        pass  # already exists


def keyword_search(vs: QdrantVectorStore, keywords: list[str], k: int) -> list[Document]:
    """Full-text keyword search over the article text (top-k). OR across keywords.

    Tries Qdrant's full-text ``MatchText`` filter; if unavailable (some local
    modes), falls back to an in-process token-count scan.
    """
    from qdrant_client import models

    keywords = [kw for kw in (keywords or []) if kw and kw.strip()]
    if not keywords:
        return []
    ckey, mkey = _content_key(vs), _metadata_key(vs)

    ensure_keyword_index(vs)
    flt = models.Filter(
        should=[models.FieldCondition(key=ckey, match=models.MatchText(text=kw)) for kw in keywords]
    )
    try:
        points, _ = vs.client.scroll(
            collection_name=vs.collection_name,
            scroll_filter=flt,
            limit=k,
            with_payload=True,
        )
        if points:
            return [
                Document(page_content=(p.payload or {}).get(ckey, ""),
                         metadata=(p.payload or {}).get(mkey, {}) or {})
                for p in points
            ]
    except Exception:
        pass
    return _keyword_scan(vs, keywords, k)


def _keyword_scan(vs: QdrantVectorStore, keywords: list[str], k: int) -> list[Document]:
    """Fallback keyword search: scroll points and rank by keyword token hits."""
    ckey, mkey = _content_key(vs), _metadata_key(vs)
    kws = [kw.lower() for kw in keywords]
    scored: list[tuple[int, Document]] = []
    offset = None
    while True:
        points, offset = vs.client.scroll(
            collection_name=vs.collection_name, limit=256, offset=offset, with_payload=True
        )
        for p in points:
            content = (p.payload or {}).get(ckey, "") or ""
            low = content.lower()
            hits = sum(low.count(kw) for kw in kws)
            if hits:
                scored.append((hits, Document(page_content=content,
                                              metadata=(p.payload or {}).get(mkey, {}) or {})))
        if offset is None or not points:
            break
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored[:k]]
