"""Conversational chatbot as a LangGraph ``StateGraph`` with memory.

A router node classifies each user turn into one of three modes and conditional
edges dispatch to the matching handler::

    START → router ─┬─(doc_qa)──→ doc_qa  → END   (answer from extracted JSON)
                    ├─(rag)─────→ rag     → END   (answer from lex.uz knowledge base)
                    └─(general)─→ general → END   (greetings / capabilities / smalltalk)

Conversation history and the active document persist across turns via a
checkpointer keyed by ``thread_id`` (LangGraph short-term memory).
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal, Optional

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from app.chat.qa import DocumentQA
from app.chat.rag import CustomsRAG
from app.llm import get_chat_model


class ChatState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    documents: Optional[list]  # active processed-document result dicts; persists in thread
    route: str
    sources: list


class Route(BaseModel):
    route: Literal["doc_qa", "rag", "general"] = Field(
        description="doc_qa = about the uploaded document's contents; "
        "rag = about customs law/regulations/procedures in general; "
        "general = greeting, capabilities, smalltalk, or anything else."
    )
    reasoning: str = Field(description="One short sentence.")


_ROUTER_SYSTEM = (
    "You route a user's message in a customs document assistant to ONE handler. "
    "Use the conversation so far to understand short follow-up questions.\n"
    "- 'doc_qa': the user asks about the SPECIFIC uploaded document(s) — a value "
    "printed on them: dates, places ('qayerdan', 'from where'), names, passengers, "
    "totals, amounts, goods, parties, numbers, or references like 'this', 'the ticket', "
    "'the invoice', 'bu', 'ushbu'. This ALSO includes asking what file(s) were "
    "uploaded, what TYPE each loaded document is, or to summarize/compare the loaded "
    "document(s) — e.g. 'what did I upload', 'qanday hujjat yuklandi', 'har birining "
    "turi', 'compare these two', 'summarize this'.\n"
    "- 'rag': the user asks about customs LAW, the Customs Code, legal definitions, "
    "rights or OBLIGATIONS, duties, tariffs, regimes, or procedures IN GENERAL — e.g. "
    "'what is the re-export regime', 'obligations of a customs broker', "
    "'bojxona brokerining majburiyatlari', 'reeksport rejimi nima'. These are about the "
    "law itself, NOT the uploaded file — choose 'rag' EVEN IF a document is loaded.\n"
    "- 'general': only greetings, thanks, or what-can-you-do.\n"
    "A document is currently {doc_state}. If NO document is loaded, never choose "
    "'doc_qa'. When torn between doc_qa and rag: choose 'rag' for anything about the "
    "law/regulations, and 'doc_qa' only for a fact that would be printed on the user's "
    "own document."
)

_GENERAL_SYSTEM = (
    "You are a friendly assistant for a customs document intelligence platform. "
    "You can: extract structured data from uploaded customs documents (invoices, "
    "air waybills, CMR, packing lists, customs declarations) and answer questions "
    "about Uzbekistan customs law. Greet the user, explain your capabilities when "
    "asked, and keep replies short. ALWAYS reply in the user's language "
    "(Uzbek, Russian, or English)."
)


# Greeting / meta detection (multilingual) — used to keep the router from
# misrouting short content follow-ups to 'general' when a document is loaded.
_GREET_WORDS = {
    "hello", "hi", "hey", "thanks", "thank", "thankyou", "help", "ok", "okay",
    "salom", "assalom", "assalomu", "rahmat", "raxmat", "yordam", "qalaysan", "qalesan",
    "privet", "spasibo", "привет", "спасибо", "помощь", "рахмат", "здравствуйте",
}
_GREET_PHRASES = (
    "what can you", "who are you", "good morning", "good evening", "good afternoon",
    "nima qila", "kim sen", "sen kim", "sen nima", "yordam ber",
    "что ты", "кто ты", "что умеешь", "чем поможешь", "добрый",
)


def _looks_like_greeting(text: str) -> bool:
    t = (text or "").lower().strip()
    if not t:
        return True
    if any(p in t for p in _GREET_PHRASES):
        return True
    words = set(re.findall(r"\w+", t))
    return bool(words & _GREET_WORDS) and len(words) <= 4


def _last_human_text(messages: list) -> str:
    for m in reversed(messages):
        if getattr(m, "type", None) == "human":
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def _history_pairs(messages: list) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for m in messages:
        t = getattr(m, "type", None)
        if t in ("human", "ai") and isinstance(m.content, str):
            pairs.append((t, m.content))
    return pairs


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def make_router_node(model):
    def router_node(state: ChatState) -> dict:
        has_doc = bool(state.get("documents"))
        question = _last_human_text(state["messages"])
        system = _ROUTER_SYSTEM.format(doc_state="loaded" if has_doc else "NOT loaded")
        # Include recent turns so short follow-ups ("and from where?") are understood.
        history = _history_pairs(state["messages"][:-1])[-6:]
        try:
            decision = model.with_structured_output(Route).invoke(
                [("system", system), *history, ("human", question)]
            )
            route = decision.route
        except Exception:
            route = "doc_qa" if has_doc else "general"
        if route == "doc_qa" and not has_doc:
            route = "general"
        elif has_doc and route == "general" and not _looks_like_greeting(question):
            # A document is loaded and this isn't a greeting -> it's almost
            # certainly a question about the document. Don't drop to small-talk.
            route = "doc_qa"
        return {"route": route}

    return router_node


def route_selector(state: ChatState) -> str:
    return state.get("route", "general")


# Nodes that produce the user-facing answer (used in the updates stream for
# routing/sources/final text).
_ANSWER_NODES = ("doc_qa", "rag", "general")

# Of those, only these stream plain prose token-by-token. 'rag' is excluded: it uses
# structured output, so its raw token stream is JSON (keywords + the answer object),
# not user-facing text. We emit RAG's parsed answer once, after the node completes
# (via the final-text path in stream_chat) — otherwise the JSON leaks into the chat.
_STREAMING_NODES = ("doc_qa", "general")


def make_doc_qa_node(qa: DocumentQA):
    def doc_qa_node(state: ChatState) -> dict:
        question = _last_human_text(state["messages"])
        history = _history_pairs(state["messages"][:-1])
        answer = qa.answer(question, state.get("documents") or [], history=history)
        return {"messages": [AIMessage(content=answer)], "sources": []}

    return doc_qa_node


def make_rag_node(rag: CustomsRAG):
    def rag_node(state: ChatState) -> dict:
        question = _last_human_text(state["messages"])
        history = _history_pairs(state["messages"][:-1])
        result = rag.answer(question, history=history)
        return {"messages": [AIMessage(content=result["answer_text"])], "sources": result["sources"]}

    return rag_node


def make_general_node(model):
    def general_node(state: ChatState) -> dict:
        history = _history_pairs(state["messages"])
        messages: list = [("system", _GENERAL_SYSTEM), *history]
        resp = model.invoke(messages)
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        return {"messages": [AIMessage(content=content)], "sources": []}

    return general_node


# --------------------------------------------------------------------------- #
# Graph + service
# --------------------------------------------------------------------------- #
def build_chat_graph(qa, rag, router_model, general_model, checkpointer):
    g = StateGraph(ChatState)
    g.add_node("router", make_router_node(router_model))
    g.add_node("doc_qa", make_doc_qa_node(qa))
    g.add_node("rag", make_rag_node(rag))
    g.add_node("general", make_general_node(general_model))

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router",
        route_selector,
        {"doc_qa": "doc_qa", "rag": "rag", "general": "general"},
    )
    g.add_edge("doc_qa", END)
    g.add_edge("rag", END)
    g.add_edge("general", END)
    return g.compile(checkpointer=checkpointer)


class ChatService:
    """High-level multilingual chatbot with per-thread memory."""

    def __init__(
        self,
        qa: DocumentQA | None = None,
        rag: CustomsRAG | None = None,
        router_model=None,
        general_model=None,
        checkpointer=None,
    ):
        # Answer models stream tokens; the router stays non-streaming (structured).
        self.qa = qa or DocumentQA(model=get_chat_model(streaming=True))
        self.rag = rag or CustomsRAG(model=get_chat_model(streaming=True))
        self.router_model = router_model or get_chat_model()
        self.general_model = general_model or get_chat_model(streaming=True)
        self.checkpointer = checkpointer or InMemorySaver()
        self.graph = build_chat_graph(
            self.qa, self.rag, self.router_model, self.general_model, self.checkpointer
        )

    @staticmethod
    def _normalize_docs(documents, document_context):
        """Build the documents list for the graph payload.

        Returns None when nothing was passed — the thread then KEEPS its previously
        active documents (persistence across turns). ``documents`` (a list) takes
        precedence; ``document_context`` (a single dict) is accepted for back-compat.
        Pass ``documents=[]`` to explicitly clear the active set.
        """
        if documents is not None:
            return list(documents)
        if document_context is not None:
            return [document_context]
        return None

    def chat(
        self,
        message: str,
        thread_id: str = "default",
        documents: list | None = None,
        document_context: dict | None = None,
    ) -> dict:
        payload: dict[str, Any] = {"messages": [("user", message)]}
        docs = self._normalize_docs(documents, document_context)
        if docs is not None:
            payload["documents"] = docs
        out = self.graph.invoke(payload, config={"configurable": {"thread_id": thread_id}})
        last = out["messages"][-1]
        answer = last.content if isinstance(last.content, str) else str(last.content)
        return {"answer": answer, "route": out.get("route"), "sources": out.get("sources", [])}

    def stream_chat(
        self,
        message: str,
        thread_id: str = "default",
        documents: list | None = None,
        document_context: dict | None = None,
    ):
        """Stream the reply as events while keeping full graph orchestration + memory.

        Yields dict events: ``{"type": "route", ...}``, ``{"type": "token", "text": ...}``
        (many), ``{"type": "sources", ...}``, ``{"type": "done", ...}``.

        Uses LangGraph's combined ``updates`` + ``messages`` stream so routing and
        sources come from node outputs while answer tokens stream live. If the LLM
        didn't stream tokens (e.g. a non-streaming/fake model), the full answer is
        emitted as a single token so the client always receives the reply.
        """
        payload: dict[str, Any] = {"messages": [("user", message)]}
        docs = self._normalize_docs(documents, document_context)
        if docs is not None:
            payload["documents"] = docs
        config = {"configurable": {"thread_id": thread_id}}

        streamed = False
        final_text = ""
        sources: list = []
        route: str | None = None

        for mode, data in self.graph.stream(
            payload, config, stream_mode=["updates", "messages"]
        ):
            if mode == "messages":
                chunk, meta = data
                # Only emit genuine streamed token chunks (AIMessageChunk). The
                # messages stream ALSO surfaces the complete AIMessage a node adds
                # to state — emitting that too would duplicate the whole answer.
                if (
                    meta.get("langgraph_node") in _STREAMING_NODES
                    and isinstance(chunk, AIMessageChunk)
                ):
                    text = chunk.content
                    if isinstance(text, str) and text:
                        streamed = True
                        yield {"type": "token", "text": text}
            else:  # "updates"
                for node, update in (data or {}).items():
                    if not isinstance(update, dict):
                        continue
                    if node == "router" and update.get("route"):
                        route = update["route"]
                        yield {"type": "route", "route": route}
                    elif node in _ANSWER_NODES:
                        sources = update.get("sources") or []
                        msgs = update.get("messages") or []
                        if msgs:
                            content = getattr(msgs[-1], "content", "")
                            final_text = content if isinstance(content, str) else str(content)

        if not streamed and final_text:
            yield {"type": "token", "text": final_text}
        yield {"type": "sources", "sources": sources}
        yield {"type": "done", "route": route}

    def history(self, thread_id: str) -> list[dict]:
        """Return the stored conversation for a thread (for UI restore)."""
        state = self.graph.get_state({"configurable": {"thread_id": thread_id}})
        msgs = (state.values or {}).get("messages", []) if state else []
        return [
            {"role": m.type, "content": m.content}
            for m in msgs
            if getattr(m, "type", None) in ("human", "ai")
        ]
