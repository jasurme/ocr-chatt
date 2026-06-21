"""Build the RAG knowledge base: load legislation -> chunk -> embed -> Qdrant.

Run as a script to (re)build the local knowledge base::

    python -m app.rag.ingest                  # scrape lex.uz (default language)
    python -m app.rag.ingest --lang uz,ru,en  # scrape all three languages
    python -m app.rag.ingest --limit 50       # first 50 articles (faster/cheaper)
"""

from __future__ import annotations

import argparse

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.rag.lexuz import Article, scrape_customs_code
from app.rag.vectorstore import collection_count, get_vectorstore


def articles_to_documents(
    articles: list[Article], chunk_size: int | None = None, overlap: int | None = None
) -> list[Document]:
    s = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or s.rag_chunk_size,
        chunk_overlap=overlap if overlap is not None else s.rag_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    cs = chunk_size or s.rag_chunk_size
    docs: list[Document] = []
    for art in articles:
        header = art.title  # "34-modda. Tovarni reeksport ..."
        base = {
            "source_url": art.source_url,
            "document_title": art.document_title,
            "article_number": art.number,
            "title": header,
            "language": art.language,
            "origin": "lexuz",
        }
        body = art.body or ""
        body_chunks = splitter.split_text(body) if len(body) > cs else [body]
        for i, chunk in enumerate(body_chunks or [""]):
            # Every chunk is prefixed with the article heading so it is
            # article-aware for both embedding and keyword search:
            #   "34-modda. <name>\n<article text>"
            content = f"{header}\n{chunk}".strip()
            docs.append(Document(page_content=content, metadata=dict(base, chunk=i)))
    return docs


def load_documents(
    languages: tuple[str, ...] | None = None,
    limit: int | None = None,
) -> list[Document]:
    """Scrape knowledge-base documents from lex.uz.

    Each language is scraped independently so a single failing source (network
    or parse error) never discards the languages that did succeed. Returns an
    empty list if every source fails.
    """
    langs = languages or (get_settings().lexuz_language,)
    articles: list[Article] = []
    for lang in langs:
        try:
            got = scrape_customs_code(lang, limit=limit)
            articles.extend(got)
            print(f"  scraped {len(got)} articles ({lang})")
        except Exception as exc:  # network/parse failure for this language
            print(f"  ! failed to scrape '{lang}': {type(exc).__name__}: {exc}")
    return articles_to_documents(articles) if articles else []


def seed_knowledge_base(
    vectorstore=None,
    languages: tuple[str, ...] | None = None,
    limit: int | None = None,
    reset: bool = False,
) -> int:
    from app.rag.vectorstore import build_client, ensure_keyword_index

    s = get_settings()
    if reset:
        # Drop any existing collection up front using the RAW client, BEFORE the
        # validating QdrantVectorStore wrapper is built. This lets --reset recover
        # even when the old collection used a different vector size (e.g. a 1536-dim
        # store left over from a previous embeddings model); otherwise construction
        # raises a dimension-mismatch error before the reset can run.
        client = vectorstore.client if vectorstore is not None else build_client()
        if client.collection_exists(s.qdrant_collection):
            client.delete_collection(s.qdrant_collection)
        vs = get_vectorstore(client=client, collection=s.qdrant_collection)
    else:
        vs = vectorstore or get_vectorstore()
    docs = load_documents(languages=languages, limit=limit)
    if docs:
        vs.add_documents(docs)
    ensure_keyword_index(vs)  # enable full-text keyword search
    return len(docs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the customs-law RAG knowledge base.")
    parser.add_argument(
        "--lang", default=None,
        help="Customs Code language(s), comma-separated: uz | ru | en "
        "(default: settings). Example: --lang uz,ru,en",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max articles to scrape per language.")
    parser.add_argument("--reset", action="store_true", help="Recreate the collection first.")
    args = parser.parse_args()

    langs = (
        tuple(x.strip() for x in args.lang.split(",") if x.strip()) if args.lang else None
    )
    try:
        n = seed_knowledge_base(
            languages=langs, limit=args.limit, reset=args.reset,
        )
        count = collection_count(get_vectorstore())
    except RuntimeError as exc:
        if "already accessed" in str(exc).lower():
            print(
                "✗ The embedded knowledge base is locked by another process "
                "(is the web server running?).\n"
                "  Stop the server and retry, or seed via the running app: "
                "POST /api/kb/seed (or click the KB pill in the UI).\n"
                "  For concurrent access, run Qdrant as a server "
                "(docker compose up -d) and set QDRANT_URL."
            )
            raise SystemExit(1) from exc
        raise
    print(f"Seeded {n} documents. Collection now holds {count} points.")


if __name__ == "__main__":
    main()
