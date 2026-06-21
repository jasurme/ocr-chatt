"""Tests for the RAG knowledge base — hybrid retrieval + citations (Step 8)."""

from __future__ import annotations

import pytest

from app.chat.rag import CustomsRAG, Keywords, RAGAnswer
from app.rag import (
    articles_to_documents,
    collection_count,
    get_vectorstore,
    parse_articles,
    scrape_customs_code,
    seed_knowledge_base,
)
from app.rag.lexuz import Article
from app.rag.vectorstore import keyword_search
from conftest import requires_ollama
from tests._fakes import FakeChatModel, HashEmbeddings

SYNTHETIC_HTML = """
<html><body>
  <div class="docNavbar"><div class="ACT_TEXT lx_elem">Статья 1. TOC entry noise</div></div>
  <div class="CLAUSE_DEFAULT lx_elem"><span onclick="lx_pa(1)">Аудиони тинглаш</span>Статья 1. Общие положения</div>
  <div class="ACT_TEXT lx_elem"><span onclick="lx_pa(2)">UIBTN</span>Настоящий Кодекс регулирует таможенные отношения.</div>
  <div class="CHANGES_ORIGINS lx_no_select">editorial change note</div>
  <div class="CLAUSE_DEFAULT lx_elem">Статья 2. Основные термины</div>
  <div class="ACT_TEXT lx_elem">В настоящем Кодексе применяются определения терминов.</div>
</body></html>
"""

_VS_COUNTER = {"n": 0}


def make_inmemory_vs():
    _VS_COUNTER["n"] += 1
    return get_vectorstore(
        embeddings=HashEmbeddings(64), in_memory=True, vector_size=64,
        collection=f"test_{_VS_COUNTER['n']}",
    )


def uz_articles():
    return [
        Article(34, "34-modda. Tovarni reeksport bojxona rejimiga joylashtirish hujjatlari",
                "Reeksport rejimiga joylashtirish uchun zarur hujjatlar roʻyxati.", "http://x", "uz", "Code"),
        Article(5, "5-modda. Bojxona hududi va bojxona chegarasi",
                "Bojxona hududini quruqlik hududi, suvlar va havo fazosi tashkil etadi.", "http://x", "uz", "Code"),
        Article(7, "7-modda. Boshqa qoidalar", "Boshqa umumiy qoidalar matni.", "http://x", "uz", "Code"),
    ]


# ----------------------------- parsing ------------------------------------- #
def test_parse_articles_header_and_body():
    arts = parse_articles(SYNTHETIC_HTML, language="ru", source_url="http://x")
    assert len(arts) == 2
    assert arts[0].number == 1
    assert arts[0].title == "Статья 1. Общие положения"  # full heading kept
    assert "регулирует таможенные отношения" in arts[0].body  # body excludes heading
    assert arts[0].text.startswith("Статья 1. Общие положения")  # text = heading + body


def test_parse_strips_ui_and_editorial():
    joined = " ".join(a.text for a in parse_articles(SYNTHETIC_HTML, language="ru"))
    for noise in ("Аудиони тинглаш", "UIBTN", "editorial change note", "TOC entry noise"):
        assert noise not in joined


# ----------------------------- chunk format -------------------------------- #
def test_chunk_is_header_then_body():
    art = uz_articles()[1]  # article 5, short
    docs = articles_to_documents([art])
    assert len(docs) == 1
    assert docs[0].page_content == f"{art.title}\n{art.body}"
    assert docs[0].metadata["article_number"] == 5
    assert docs[0].metadata["title"].startswith("5-modda.")


def test_long_article_every_chunk_keeps_header():
    art = Article(34, "34-modda. Reeksport hujjatlari", "soz " * 800, "http://x", "uz", "Code")
    docs = articles_to_documents([art], chunk_size=500, overlap=50)
    assert len(docs) > 1
    assert all(d.page_content.startswith("34-modda. Reeksport hujjatlari") for d in docs)
    assert all(d.metadata["article_number"] == 34 for d in docs)


# ----------------------------- keyword search ------------------------------ #
def test_keyword_search_finds_matching_article():
    vs = make_inmemory_vs()
    vs.add_documents(articles_to_documents(uz_articles()))
    hits = keyword_search(vs, ["reeksport"], k=5)
    assert hits
    assert any("reeksport" in h.page_content.lower() for h in hits)
    # an unrelated keyword should not surface the reeksport article
    assert not any("reeksport" in h.page_content.lower() for h in keyword_search(vs, ["havo"], k=5))


# ------------------------------- hybrid ------------------------------------ #
def test_hybrid_retrieve_merges_semantic_and_keyword():
    vs = make_inmemory_vs()
    vs.add_documents(articles_to_documents(uz_articles()))
    fake = FakeChatModel(structured_by_schema={Keywords: Keywords(keywords=["reeksport"])})
    rag = CustomsRAG(vectorstore=vs, model=fake, semantic_k=2, keyword_k=5)
    docs, kws, n_sem, n_kw = rag.retrieve("Reeksport bojxona rejimi nima")
    assert kws == ["reeksport"]
    assert n_kw >= 1 and n_sem >= 1
    assert any("reeksport" in d.page_content.lower() for d in docs)


# ------------------------------- citations --------------------------------- #
def test_answer_returns_only_cited_sources():
    vs = make_inmemory_vs()
    vs.add_documents(articles_to_documents(uz_articles()))
    fake = FakeChatModel(structured_by_schema={
        Keywords: Keywords(keywords=["reeksport"]),
        RAGAnswer: RAGAnswer(answer_text="34-moddaga koʻra hujjatlar talab qilinadi.", cited_sources=[1]),
    })
    rag = CustomsRAG(vectorstore=vs, model=fake, semantic_k=3, keyword_k=5)
    result = rag.answer("Reeksport uchun qanday hujjatlar kerak")
    assert result["used_context"] is True
    assert len(result["sources"]) == 1  # only the cited article, not all retrieved
    assert result["citings"] and "34-modda" in result["citings"][0]  # grounded citation string
    assert "34-moddaga" in result["answer_text"]
    assert result["keywords"] == ["reeksport"]
    assert result["retrieved"]["merged"] >= 1


def test_answer_empty_kb():
    vs = make_inmemory_vs()
    rag = CustomsRAG(vectorstore=vs, model=FakeChatModel(content="x"))
    result = rag.answer("anything")
    assert result["used_context"] is False
    assert result["sources"] == []


# ------------------------------- degradation ------------------------------- #
def test_retrieve_degrades_when_vectorstore_raises():
    class BoomVS:
        def similarity_search(self, *a, **k):
            raise RuntimeError("Storage folder is already accessed by another instance")

    rag = CustomsRAG(vectorstore=BoomVS(), model=FakeChatModel(content="x"))
    docs, *_ = rag.retrieve("q")
    assert docs == []
    assert rag.answer("q")["used_context"] is False


def test_retrieve_degrades_when_provider_raises():
    def boom():
        raise RuntimeError("already accessed by another instance of Qdrant client")

    rag = CustomsRAG(vectorstore_provider=boom, model=FakeChatModel(content="x"))
    docs, *_ = rag.retrieve("q")
    assert docs == []


# --------------------------- live integration ------------------------------ #
@pytest.mark.integration
def test_scrape_lexuz_uz_live():
    arts = scrape_customs_code("uz", limit=40)
    assert len(arts) == 40
    a34 = next((a for a in arts if a.number == 34), None)
    assert a34 is not None and "reeksport" in a34.title.lower()


@requires_ollama
@pytest.mark.integration
def test_rag_hybrid_real_models():
    vs = get_vectorstore(in_memory=True, collection="live_rag")  # real bge-m3 embeddings
    seed_knowledge_base(vectorstore=vs, languages=("en",), limit=30)  # scrape lex.uz
    rag = CustomsRAG(vectorstore=vs, semantic_k=5, keyword_k=5)
    result = rag.answer("What does the customs territory of Uzbekistan include?")
    assert result["used_context"] is True
    assert result["answer_text"].strip()
    assert result["keywords"]  # keywords were extracted
