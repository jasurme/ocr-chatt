"""Tests for document classification (Step 4)."""

from __future__ import annotations

import pytest

from app.classification import Classification, DocumentClassifier
from app.ingestion import ingest_file, render_page_png
from app.ocr import OCRService
from app.schemas.types import DocumentType, classifier_type_menu
from conftest import requires_ollama
from tests._fakes import FakeChatModel


# ----------------------------- type menu ----------------------------------- #
def test_type_menu_lists_all_types():
    menu = classifier_type_menu()
    for dt in DocumentType:
        assert dt.value in menu


# --------------------------- unit (fake LLM) ------------------------------- #
def test_classify_text_uses_text_model():
    fake = FakeChatModel(
        structured_response=Classification(
            doc_type=DocumentType.INVOICE, language="en", confidence=0.96, reasoning="has invoice no"
        )
    )
    clf = DocumentClassifier(model=fake)
    result = clf.classify(text="INVOICE No. HQPL00073841 Total amount EUR 6350")
    assert result.doc_type is DocumentType.INVOICE
    assert result.confidence == pytest.approx(0.96)
    # the document text was actually included in the prompt
    assert "INVOICE" in str(fake.captured[0])


def test_classify_uses_vision_when_text_is_empty():
    text_model = FakeChatModel(
        structured_response=Classification(
            doc_type=DocumentType.OTHER, language="en", confidence=0.1, reasoning="x"
        )
    )
    vision_model = FakeChatModel(
        structured_response=Classification(
            doc_type=DocumentType.AIR_WAYBILL, language="en", confidence=0.9, reasoning="awb no"
        )
    )
    clf = DocumentClassifier(model=text_model, vision_model=vision_model)
    result = clf.classify(text="", image_png=b"\x89PNG fake bytes")
    assert result.doc_type is DocumentType.AIR_WAYBILL
    assert vision_model.captured and not text_model.captured  # only vision used


def test_classify_prefers_text_even_when_image_present():
    text_model = FakeChatModel(
        structured_response=Classification(
            doc_type=DocumentType.LETTER, language="ru", confidence=0.8, reasoning="letter"
        )
    )
    vision_model = FakeChatModel(
        structured_response=Classification(
            doc_type=DocumentType.OTHER, language="en", confidence=0.2, reasoning="x"
        )
    )
    clf = DocumentClassifier(model=text_model, vision_model=vision_model)
    result = clf.classify(text="A long enough business letter body here.", image_png=b"img")
    assert result.doc_type is DocumentType.LETTER
    assert text_model.captured and not vision_model.captured


def test_classify_error_returns_other():
    fake = FakeChatModel(raise_on_invoke=RuntimeError("API down"))
    clf = DocumentClassifier(model=fake)
    result = clf.classify(text="some text long enough to classify")
    assert result.doc_type is DocumentType.OTHER
    assert result.confidence == 0.0


# --------------------------- live integration ------------------------------ #
@requires_ollama
@pytest.mark.integration
@pytest.mark.parametrize(
    "key,expected",
    [
        ("letter_pdf", DocumentType.LETTER),
        ("invoice_pdf2", DocumentType.INVOICE),
        ("junk_docx", DocumentType.OTHER),
    ],
)
def test_classify_real_text_documents(samples, key, expected):
    doc = ingest_file(samples[key])
    result = DocumentClassifier().classify(text=doc.full_text, filename=doc.filename)
    assert result.doc_type is expected


@requires_ollama
@pytest.mark.integration
@pytest.mark.parametrize(
    "key,expected",
    [
        ("awb_jpg", DocumentType.AIR_WAYBILL),
        ("gtd_jpg", DocumentType.CUSTOMS_DECLARATION),
    ],
)
def test_classify_real_image_documents(samples, key, expected):
    doc = ingest_file(samples[key])
    img = render_page_png(doc, 1)
    result = DocumentClassifier().classify(text="", image_png=img, filename=doc.filename)
    assert result.doc_type is expected
