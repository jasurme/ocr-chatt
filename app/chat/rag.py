"""RAG mode: answer general customs-law questions from the lex.uz knowledge base.

Hybrid retrieval: dense **semantic** search (top-K) + **keyword** full-text search
(top-K) over LLM-extracted keywords, merged into one context. The answer names
the specific article(s) it used and only those sources are returned (not every
retrieved excerpt — most are usually irrelevant).
"""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import get_settings
from app.llm import get_chat_model


class Keywords(BaseModel):
    keywords: list[str] = Field(
        default_factory=list,
        description="2-3 important, specific keywords from the question for a keyword "
        "search (domain terms like 'reeksport', 'tranzit'); never generic words.",
    )


class RAGAnswer(BaseModel):
    answer_text: str = Field(
        description="The answer, in the SAME language as the question. Name the "
        "article(s) you relied on inside the text (e.g. '282-moddaga koʻra ...')."
    )
    cited_sources: list[int] = Field(
        default_factory=list,
        description="ONLY the [Source N] numbers you actually used to answer "
        "(usually 1-2). Use [] if no excerpt answered the question.",
    )


_KW_SYSTEM = (
    "Extract the 2-3 MOST IMPORTANT, specific keywords from the user's customs-law "
    "question for a keyword search. Prefer domain terms (e.g. 'reeksport', "
    "'bojxona rejimi', 'tranzit', 'bojxona qiymati'). Do NOT include generic words "
    "(what/is/the/nima/qanday/qanaqa). Keep them in the question's language."
)

_ANSWER_SYSTEM = (
    "You are a customs-law assistant for the Republic of Uzbekistan. You are given "
    "numbered reference excerpts from the Customs Code. Each is shown as "
    "'[Source N] <citation>' followed by the article text.\n"
    "Answer the user's question using ONLY these excerpts.\n"
    "Rules:\n"
    "- Many excerpts may be irrelevant — use only the few that actually answer it.\n"
    "- Write answer_text in the SAME language as the question, and explicitly name the "
    "article(s) you relied on inside the text "
    "(e.g. '34-moddaga koʻra ...' / 'According to Article 34 ...').\n"
    "- Set cited_sources to ONLY the [Source N] numbers you actually used (usually "
    "1-2); do not list excerpts you didn't use.\n"
    "- If the excerpts do not contain the answer, say so and set cited_sources to []."
)


class CustomsRAG:
    def __init__(
        self,
        vectorstore=None,
        model=None,
        vectorstore_provider=None,
        semantic_k: int | None = None,
        keyword_k: int | None = None,
    ):
        self._vectorstore = vectorstore
        self._provider = vectorstore_provider
        self._model = model
        s = get_settings()
        self.semantic_k = semantic_k or s.rag_semantic_k
        self.keyword_k = keyword_k or s.rag_keyword_k

    @property
    def vectorstore(self):
        if self._vectorstore is None:
            if self._provider is not None:
                self._vectorstore = self._provider()
            else:
                from app.rag.vectorstore import get_vectorstore

                self._vectorstore = get_vectorstore()
        return self._vectorstore

    @property
    def model(self):
        if self._model is None:
            self._model = get_chat_model()
        return self._model

    # ---- retrieval ----
    def extract_keywords(self, query: str) -> list[str]:
        try:
            kw = self.model.with_structured_output(Keywords).invoke(
                [("system", _KW_SYSTEM), ("human", query)]
            )
            out = [k.strip() for k in (kw.keywords or []) if k and k.strip()]
            if out:
                return out[:3]
        except Exception:
            pass
        # Fallback: take the longest word-ish tokens (skip short/common words).
        return [w for w in re.findall(r"[^\W\d_]{5,}", query, re.UNICODE)][:3]

    def semantic_search(self, query: str) -> list[Document]:
        try:
            return self.vectorstore.similarity_search(query, k=self.semantic_k)
        except Exception:
            return []

    def keyword_search(self, keywords: list[str]) -> list[Document]:
        from app.rag.vectorstore import keyword_search

        try:
            return keyword_search(self.vectorstore, keywords, self.keyword_k)
        except Exception:
            return []

    @staticmethod
    def _dedup_key(d: Document) -> tuple:
        m = d.metadata or {}
        return (m.get("article_number"), m.get("chunk"), (d.page_content or "")[:60])

    def retrieve(self, query: str) -> tuple[list[Document], list[str], int, int]:
        """Hybrid: merge semantic (top-K) + keyword (top-K), de-duplicated."""
        semantic = self.semantic_search(query)
        keywords = self.extract_keywords(query)
        keyword_hits = self.keyword_search(keywords)
        merged: list[Document] = []
        seen: set = set()
        for d in [*semantic, *keyword_hits]:
            key = self._dedup_key(d)
            if key in seen:
                continue
            seen.add(key)
            merged.append(d)
        return merged, keywords, len(semantic), len(keyword_hits)

    # ---- formatting ----
    @staticmethod
    def _citation(doc: Document) -> str:
        """A grounded, human-readable citation built from the article's metadata
        (never invented), e.g.
        'Oʻzbekiston Respublikasining Bojxona kodeksi 282-modda. Bojxona brokerining majburiyatlari'.
        """
        m = doc.metadata or {}
        code = (m.get("document_title") or "").strip()
        heading = (m.get("title") or "").strip()
        return f"{code} {heading}".strip() if code else heading

    @classmethod
    def _context(cls, docs: list[Document]) -> str:
        return "\n\n".join(
            f"[Source {i}] {cls._citation(d)}\n{d.page_content}"
            for i, d in enumerate(docs, start=1)
        )

    @classmethod
    def format_sources(cls, docs: list[Document]) -> list[dict]:
        sources = []
        for d in docs:
            m = d.metadata or {}
            sources.append(
                {
                    "citation": cls._citation(d),
                    "title": m.get("title"),
                    "article_number": m.get("article_number"),
                    "source_url": m.get("source_url"),
                    "language": m.get("language"),
                    "snippet": (d.page_content or "")[:300],
                }
            )
        return sources

    # ---- answer ----
    def answer(self, query: str, history: list[tuple[str, str]] | None = None) -> dict:
        docs, keywords, n_sem, n_kw = self.retrieve(query)
        meta = {"keywords": keywords, "retrieved": {"semantic": n_sem, "keyword": n_kw, "merged": len(docs)}}
        if not docs:
            return {
                "answer_text": (
                    "The knowledge base has no relevant information yet. Please seed it "
                    "(python -m app.rag.ingest) or consult https://lex.uz."
                ),
                "citings": [],
                "sources": [],
                "used_context": False,
                **meta,
            }

        context = self._context(docs)
        messages: list = [SystemMessage(content=_ANSWER_SYSTEM)]
        for role, content in history or []:
            messages.append((role, content))
        messages.append(HumanMessage(content=f"Reference excerpts:\n{context}\n\nQuestion: {query}"))

        cited: list[Document] = []
        try:
            res = self.model.with_structured_output(RAGAnswer).invoke(messages)
            answer_text = res.answer_text
            for i in res.cited_sources or []:
                if isinstance(i, int) and 1 <= i <= len(docs):
                    cited.append(docs[i - 1])
        except Exception:
            resp = self.model.invoke(messages)
            answer_text = resp.content if isinstance(resp.content, str) else str(resp.content)

        # Keep only the article(s) actually cited, de-duplicated by (article, language).
        seen_art: set = set()
        cited_unique: list[Document] = []
        for d in cited:
            m = d.metadata or {}
            key = (m.get("article_number"), m.get("language"))
            if key in seen_art:
                continue
            seen_art.add(key)
            cited_unique.append(d)

        return {
            "answer_text": answer_text,
            # Grounded citations copied verbatim from the cited articles' metadata
            # (never invented), e.g.
            # "Oʻzbekiston Respublikasining Bojxona kodeksi 282-modda. ...".
            "citings": [self._citation(d) for d in cited_unique],
            "sources": self.format_sources(cited_unique),
            "used_context": True,
            **meta,
        }
