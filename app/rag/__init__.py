"""RAG knowledge base: lex.uz scraping, chunking, Qdrant vector store."""

from app.rag.fallback_corpus import fallback_documents
from app.rag.ingest import articles_to_documents, load_documents, seed_knowledge_base
from app.rag.lexuz import Article, parse_articles, scrape_customs_code
from app.rag.vectorstore import build_client, collection_count, get_vectorstore

__all__ = [
    "Article",
    "parse_articles",
    "scrape_customs_code",
    "fallback_documents",
    "articles_to_documents",
    "load_documents",
    "seed_knowledge_base",
    "get_vectorstore",
    "build_client",
    "collection_count",
]
