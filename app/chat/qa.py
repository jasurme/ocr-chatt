"""Document Q&A: answer questions about a processed document using the
structured JSON that was already extracted (fast + accurate; no re-OCR)."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings
from app.llm import get_chat_model

_SYSTEM = (
    "You are a helpful assistant for a customs brokerage. The user has uploaded "
    "one or more documents that have already been processed into structured data. "
    "Answer the user's question using ONLY the provided structured data (and the raw "
    "OCR text if given as a fallback).\n"
    "Rules:\n"
    "- Be concise and precise; quote exact values, numbers and codes.\n"
    "- Answer naturally, like a normal conversation. Do NOT mention internal field "
    "names or JSON keys (e.g. 'departure_location', 'passenger_name', 'extra_fields'), "
    "and do NOT explain where in the data you found the value — just state the answer.\n"
    "- If MULTIPLE documents are provided, each is labelled '### Document N: <file>'. "
    "Use the relevant one(s); when the user asks to compare or relate documents, draw "
    "on all of them and refer to each by its file name or type (never by field name).\n"
    "- If the answer is not present in the data, clearly say you don't have that "
    "information — do NOT invent values.\n"
    "- ALWAYS reply in the SAME language as the user's question, written fluently and "
    "naturally. If the question is in Uzbek, answer in clean, natural Uzbek and do NOT "
    "mix in Turkish or other languages.\n"
    "- When listing items (goods, line items), present them clearly."
)


def _prune(value: Any) -> Any:
    """Drop null/empty fields so the JSON context is compact and unambiguous."""
    if isinstance(value, dict):
        out = {k: _prune(v) for k, v in value.items()}
        return {k: v for k, v in out.items() if v not in (None, "", [], {})}
    if isinstance(value, list):
        items = [_prune(v) for v in value]
        return [v for v in items if v not in (None, "", [], {})]
    return value


class DocumentQA:
    def __init__(self, model=None):
        self._model = model

    @property
    def model(self):
        if self._model is None:
            self._model = get_chat_model()
        return self._model

    def build_context(self, documents) -> str:
        """Render one or more processed-document results into a compact prompt context.

        Accepts either a single result dict or a list of them (multi-document Q&A).
        Each document becomes a JSON block; when several are loaded each is prefixed
        with a '### Document N: <file>' header so the model can attribute facts and
        compare across files.
        """
        docs = documents if isinstance(documents, list) else [documents]
        docs = [d for d in docs if d]
        if not docs:
            return "(no document is currently loaded)"

        settings = get_settings()
        # Per-document OCR-fallback budget shrinks as more documents are loaded so the
        # combined prompt stays within the model's context window.
        ocr_budget = max(2000, settings.max_prompt_chars // len(docs))
        multi = len(docs) > 1

        sections: list[str] = []
        for i, dc in enumerate(docs, start=1):
            extracted = _prune(dc.get("extracted") or {})
            block = {
                "filename": dc.get("filename"),
                "doc_type": dc.get("doc_type"),
                "language": dc.get("language"),
                "extracted_data": extracted,
            }
            text = json.dumps(_prune(block), ensure_ascii=False, indent=2)
            # Fall back to raw OCR text when structured data is sparse/empty.
            if not extracted:
                ocr = (dc.get("ocr_text") or "")[:ocr_budget]
                if ocr:
                    text += f"\n\nRAW OCR TEXT (fallback):\n{ocr}"
            if multi:
                text = f"### Document {i}: {dc.get('filename') or 'document'}\n{text}"
            sections.append(text)
        return "\n\n".join(sections)

    def answer(
        self,
        question: str,
        documents,
        history: list[tuple[str, str]] | None = None,
    ) -> str:
        context = self.build_context(documents)
        messages: list = [SystemMessage(content=_SYSTEM)]
        for role, content in history or []:
            messages.append((role, content))
        messages.append(
            HumanMessage(
                content=(
                    f"Here is the processed document data:\n{context}\n\n"
                    f"Question: {question}"
                )
            )
        )
        response = self.model.invoke(messages)
        return response.content if isinstance(response.content, str) else str(response.content)
