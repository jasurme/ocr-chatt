"""Tests for Document Q&A over extracted JSON (Step 7)."""

from __future__ import annotations

import pytest

from app.chat.qa import DocumentQA, _prune
from conftest import requires_ollama
from tests._fakes import FakeChatModel

CONTEXT = {
    "filename": "инв.PDF",
    "doc_type": "invoice",
    "language": "en",
    "ocr_text": "Invoice 1090130561 ORAFOL ...",
    "extracted": {
        "invoice_number": "1090130561",
        "invoice_date": "2025-12-08",
        "currency": "EUR",
        "total_amount": "12345.67",
        "seller": {"name": "ORAFOL Europe GmbH", "country": "Germany", "tax_id": None},
        "buyer": {"name": "PE MAT-MAX", "country": "Uzbekistan"},
        "line_items": [
            {"description": "641G White", "quantity": "300", "unit": "M2", "hs_code": None},
            {"description": "641M Deep sea", "quantity": "50", "unit": "M2"},
        ],
        "notes": None,
    },
}


# ----------------------------- pruning ------------------------------------- #
def test_prune_removes_nulls_and_empties():
    pruned = _prune({"a": 1, "b": None, "c": "", "d": [], "e": {"x": None}, "f": [1, None]})
    assert pruned == {"a": 1, "f": [1]}


def test_build_context_excludes_nulls_includes_values():
    ctx = DocumentQA(model=FakeChatModel()).build_context(CONTEXT)
    assert "1090130561" in ctx
    assert "ORAFOL" in ctx
    assert "tax_id" not in ctx  # null pruned
    assert "notes" not in ctx


def test_build_context_falls_back_to_ocr_when_empty():
    qa = DocumentQA(model=FakeChatModel())
    ctx = qa.build_context({"extracted": {}, "ocr_text": "RAW textual content here"})
    assert "RAW OCR TEXT" in ctx
    assert "RAW textual content" in ctx


def test_build_context_single_doc_in_list_has_no_header():
    # A one-element list behaves exactly like a single dict (no "Document N" header).
    qa = DocumentQA(model=FakeChatModel())
    ctx = qa.build_context([CONTEXT])
    assert "### Document" not in ctx
    assert "1090130561" in ctx


def test_build_context_multiple_documents():
    qa = DocumentQA(model=FakeChatModel())
    awb = {"filename": "awb.pdf", "doc_type": "air_waybill",
           "extracted": {"awb_number": "180-12345678"}}
    ctx = qa.build_context([CONTEXT, awb])
    assert "### Document 1" in ctx and "### Document 2" in ctx
    assert "инв.PDF" in ctx and "awb.pdf" in ctx
    assert "1090130561" in ctx and "180-12345678" in ctx  # data from both files


# --------------------------- unit (fake LLM) ------------------------------- #
def test_answer_uses_context_and_returns_content():
    fake = FakeChatModel(content="The invoice number is 1090130561.")
    answer = DocumentQA(model=fake).answer("What is the invoice number?", CONTEXT)
    assert "1090130561" in answer
    # the structured data was actually placed in the prompt
    sent = str(fake.invoked[0])
    assert "1090130561" in sent and "What is the invoice number?" in sent


def test_answer_includes_history():
    fake = FakeChatModel(content="ok")
    DocumentQA(model=fake).answer(
        "and the total?", CONTEXT, history=[("human", "hi"), ("ai", "hello")]
    )
    sent = str(fake.invoked[0])
    assert "hello" in sent


# --------------------------- live integration ------------------------------ #
@requires_ollama
@pytest.mark.integration
def test_answer_english_invoice_number():
    ans = DocumentQA().answer("What is the invoice number?", CONTEXT)
    assert "1090130561" in ans


@requires_ollama
@pytest.mark.integration
def test_answer_russian_question_returns_value():
    ans = DocumentQA().answer("Какой номер инвойса?", CONTEXT)
    assert "1090130561" in ans
    # answered in Russian (contains Cyrillic)
    assert any("Ѐ" <= ch <= "ӿ" for ch in ans)


@requires_ollama
@pytest.mark.integration
def test_answer_uzbek_question_returns_value():
    ans = DocumentQA().answer("Invoys raqami nechta?", CONTEXT)
    assert "1090130561" in ans


@requires_ollama
@pytest.mark.integration
def test_answer_unknown_info_is_honest():
    ans = DocumentQA().answer("What is the captain's phone number?", CONTEXT).lower()
    assert any(k in ans for k in ["don't", "do not", "not", "no ", "unavailable", "n/a"])


@requires_ollama
@pytest.mark.integration
def test_answer_lists_goods():
    ans = DocumentQA().answer("List all the goods.", CONTEXT)
    assert "641G" in ans or "White" in ans
