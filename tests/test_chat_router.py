"""Tests for the conversational router graph with memory (Step 9)."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from app.chat import ChatService, CustomsRAG, DocumentQA
from app.chat.rag import Keywords, RAGAnswer
from app.chat.router import Route
from app.rag import get_vectorstore, seed_knowledge_base
from conftest import requires_ollama
from tests._fakes import FakeChatModel, HashEmbeddings

_C = {"n": 0}

INVOICE_CTX = {
    "filename": "инв.PDF",
    "doc_type": "invoice",
    "extracted": {"invoice_number": "1090130561", "total_amount": "12345.67", "currency": "EUR"},
}


def make_service(route="general", qa_answer="DOC ANSWER", rag_answer="RAG ANSWER [Source 1]",
                 general_answer="HELLO, I can help with customs docs."):
    _C["n"] += 1
    router_model = FakeChatModel(structured_response=Route(route=route, reasoning="x"))
    qa = DocumentQA(model=FakeChatModel(content=qa_answer))
    vs = get_vectorstore(
        embeddings=HashEmbeddings(64), in_memory=True, vector_size=64, collection=f"chat_{_C['n']}"
    )
    seed_knowledge_base(vectorstore=vs, use_network=False)
    rag_model = FakeChatModel(
        content=rag_answer,
        structured_by_schema={
            Keywords: Keywords(keywords=["customs"]),
            RAGAnswer: RAGAnswer(answer_text=rag_answer, cited_sources=[1]),
        },
    )
    rag = CustomsRAG(vectorstore=vs, model=rag_model)
    return ChatService(
        qa=qa,
        rag=rag,
        router_model=router_model,
        general_model=FakeChatModel(content=general_answer),
        checkpointer=InMemorySaver(),
    )


# ----------------------------- routing ------------------------------------- #
def test_routes_to_doc_qa():
    svc = make_service(route="doc_qa")
    out = svc.chat("What is the total?", thread_id="a", document_context=INVOICE_CTX)
    assert out["route"] == "doc_qa"
    assert out["answer"] == "DOC ANSWER"


def test_routes_to_doc_qa_multiple_documents():
    svc = make_service(route="doc_qa")
    awb = {"filename": "awb.pdf", "doc_type": "air_waybill", "extracted": {"awb_number": "1"}}
    out = svc.chat("compare the two", thread_id="multi", documents=[INVOICE_CTX, awb])
    assert out["route"] == "doc_qa"
    assert out["answer"] == "DOC ANSWER"


def test_routes_to_rag():
    svc = make_service(route="rag")
    out = svc.chat("What is the customs duty on imports?", thread_id="b")
    assert out["route"] == "rag"
    assert "RAG ANSWER" in out["answer"]
    assert out["sources"]  # RAG attaches sources


def test_routes_to_general():
    svc = make_service(route="general")
    out = svc.chat("hi there", thread_id="c")
    assert out["route"] == "general"
    assert "HELLO" in out["answer"]


def test_doc_qa_downgraded_when_no_document():
    # Router says doc_qa, but no document is loaded -> must not run doc_qa.
    svc = make_service(route="doc_qa")
    out = svc.chat("what is the total?", thread_id="d")  # no document_context
    assert out["route"] == "general"


def test_router_guard_rescues_content_followup():
    # Router (fake) says 'general', but a document is loaded and the message is a
    # content question -> guard forces doc_qa so follow-ups aren't lost.
    from langchain_core.messages import HumanMessage

    from app.chat.router import _looks_like_greeting, make_router_node

    node = make_router_node(FakeChatModel(structured_response=Route(route="general", reasoning="x")))
    doc_state = {"messages": [HumanMessage(content="qayerdan uchadi")], "documents": [{"x": 1}]}
    assert node(doc_state)["route"] == "doc_qa"
    # a greeting stays general even with a document loaded
    greet = {"messages": [HumanMessage(content="salom")], "documents": [{"x": 1}]}
    assert node(greet)["route"] == "general"
    # no document -> stays general
    nodoc = {"messages": [HumanMessage(content="qayerdan uchadi")]}
    assert node(nodoc)["route"] == "general"

    assert _looks_like_greeting("salom") and _looks_like_greeting("what can you do?")
    assert not _looks_like_greeting("qayerdan uchadi")
    assert not _looks_like_greeting("what is the total amount on the invoice")


# ----------------------------- memory -------------------------------------- #
def test_document_context_persists_across_turns():
    svc = make_service(route="doc_qa")
    svc.chat("extract data", thread_id="mem1", document_context=INVOICE_CTX)
    # Second turn does NOT resend the document; it must persist in the thread.
    out = svc.chat("and the total?", thread_id="mem1")
    assert out["route"] == "doc_qa"
    assert out["answer"] == "DOC ANSWER"


def test_history_accumulates_per_thread():
    svc = make_service(route="general")
    svc.chat("hello", thread_id="h1")
    svc.chat("again", thread_id="h1")
    hist = svc.history("h1")
    assert len(hist) == 4  # 2 human + 2 ai
    assert hist[0]["role"] == "human" and hist[0]["content"] == "hello"


def test_threads_are_isolated():
    svc = make_service(route="general")
    svc.chat("thread one msg", thread_id="t1")
    assert svc.history("t2") == []  # different thread, empty


def test_chat_does_not_crash_when_kb_unavailable():
    # Simulate a locked/unavailable knowledge base: RAG must degrade, not 500,
    # and doc_qa / general must keep working since they don't need the KB.
    def boom():
        raise RuntimeError("Storage folder already accessed by another instance")

    rag = CustomsRAG(vectorstore_provider=boom, model=FakeChatModel(content="unused"))
    svc = ChatService(
        qa=DocumentQA(model=FakeChatModel(content="DOC ANSWER")),
        rag=rag,
        router_model=FakeChatModel(structured_response=Route(route="rag", reasoning="x")),
        general_model=FakeChatModel(content="GENERAL ANSWER"),
        checkpointer=InMemorySaver(),
    )
    # routed to rag, but KB is down -> graceful answer, no exception
    out = svc.chat("What is the customs duty?", thread_id="kbdown")
    assert out["route"] == "rag"
    assert out["sources"] == []
    assert out["answer"]  # a helpful fallback message, not a crash


def test_stream_chat_event_protocol():
    # With fake (non-streaming) models, stream_chat must still emit a clean
    # protocol: route -> token(s) -> sources -> done, delivering the full answer.
    svc = make_service(route="general")
    events = list(svc.stream_chat("hi", thread_id="s1"))
    types = [e["type"] for e in events]
    assert types[0] == "route"
    assert types[-1] == "done"
    assert "sources" in types
    route_ev = next(e for e in events if e["type"] == "route")
    assert route_ev["route"] == "general"
    answer = "".join(e["text"] for e in events if e["type"] == "token")
    assert "HELLO" in answer  # fallback delivers the full text when not streamed


def test_stream_chat_doc_qa_and_memory():
    svc = make_service(route="doc_qa")
    ev1 = list(svc.stream_chat("extract", thread_id="sm", document_context=INVOICE_CTX))
    assert any(e["type"] == "route" and e["route"] == "doc_qa" for e in ev1)
    assert "DOC ANSWER" in "".join(e["text"] for e in ev1 if e["type"] == "token")
    # streamed turns are persisted to thread memory too
    assert len(svc.history("sm")) == 2


def test_doc_qa_unaffected_by_broken_kb(samples):
    def boom():
        raise RuntimeError("locked")

    svc = ChatService(
        qa=DocumentQA(model=FakeChatModel(content="DOC ANSWER")),
        rag=CustomsRAG(vectorstore_provider=boom),
        router_model=FakeChatModel(structured_response=Route(route="doc_qa", reasoning="x")),
        general_model=FakeChatModel(content="GEN"),
        checkpointer=InMemorySaver(),
    )
    out = svc.chat("total?", thread_id="dq", document_context=INVOICE_CTX)
    assert out["route"] == "doc_qa"
    assert out["answer"] == "DOC ANSWER"


# --------------------------- live integration ------------------------------ #
@requires_ollama
@pytest.mark.integration
def test_live_routing_three_modes():
    vs = get_vectorstore(in_memory=True, collection="chat_live")
    seed_knowledge_base(vectorstore=vs, use_network=False)
    svc = ChatService(rag=CustomsRAG(vectorstore=vs))

    doc = svc.chat("What is the invoice total?", thread_id="L", document_context=INVOICE_CTX)
    assert doc["route"] == "doc_qa"
    assert "12345" in doc["answer"].replace(",", "").replace(" ", "")

    law = svc.chat("What is the customs territory of Uzbekistan?", thread_id="L2")
    assert law["route"] == "rag"

    hello = svc.chat("Salom! Sen nima qila olasan?", thread_id="L3")
    assert hello["route"] == "general"


@requires_ollama
@pytest.mark.integration
def test_live_streaming_emits_multiple_tokens():
    svc = ChatService()
    events = list(svc.stream_chat("Write one short friendly sentence.", thread_id="stream"))
    tokens = [e for e in events if e["type"] == "token"]
    # a genuinely streamed reply arrives in several chunks (not one blob)
    assert len(tokens) >= 2
    assert events[-1]["type"] == "done"
    assert "".join(t["text"] for t in tokens).strip()


@requires_ollama
@pytest.mark.integration
def test_live_streaming_no_duplication():
    # Regression: the streamed answer must appear exactly once (messages-mode
    # also surfaces the node's complete message, which must NOT be re-emitted).
    svc = ChatService()
    events = list(svc.stream_chat("Reply with exactly this one word: BANANA", thread_id="dedup"))
    joined = "".join(e["text"] for e in events if e["type"] == "token")
    assert joined.count("BANANA") == 1


@requires_ollama
@pytest.mark.integration
def test_live_streaming_followup_routes_to_doc_qa():
    # Short ambiguous follow-up must still route to doc_qa via memory + history.
    svc = ChatService()
    ctx = {"filename": "ticket", "doc_type": "other",
           "extracted": {"passenger": "A", "from": "Shanghai", "to": "Tashkent"}}
    list(svc.stream_chat("kim uchadi?", thread_id="fu", document_context=ctx))  # consume turn 1
    ev = list(svc.stream_chat("qayerdan uchadi", thread_id="fu"))  # no doc resent
    assert next(e["route"] for e in ev if e["type"] == "route") == "doc_qa"


@requires_ollama
@pytest.mark.integration
def test_live_memory_followup():
    svc = ChatService()
    svc.chat("The invoice total is what?", thread_id="M", document_context=INVOICE_CTX)
    # follow-up relies on remembered document + history
    out = svc.chat("And which currency is that in?", thread_id="M")
    assert out["route"] == "doc_qa"
    assert "EUR" in out["answer"]
