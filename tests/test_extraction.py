"""Tests for structured extraction (Step 5)."""

from __future__ import annotations

import pytest

from app.extraction import ExtractionResult, StructuredExtractor
from app.extraction.prompts import build_extraction_system_prompt
from app.ingestion import ingest_file, render_page_png
from app.ocr import OCRService
from app.schemas import (
    AirWaybillData,
    CustomsDeclarationData,
    InvoiceData,
    LetterData,
    get_schema,
)
from app.schemas.types import DocumentType
from conftest import requires_ollama
from tests._fakes import FakeChatModel


def _images(path, pages):
    doc = ingest_file(path)
    return [render_page_png(doc, p) for p in pages]


# ----------------------------- registry ------------------------------------ #
def test_schema_registry_maps_each_type():
    assert get_schema(DocumentType.INVOICE) is InvoiceData
    assert get_schema(DocumentType.AIR_WAYBILL) is AirWaybillData
    assert get_schema(DocumentType.CUSTOMS_DECLARATION) is CustomsDeclarationData


def test_every_schema_instantiates_empty():
    # All fields optional => empty instance must be valid (used as error fallback).
    for dt in DocumentType:
        model = get_schema(dt)
        inst = model()
        assert isinstance(inst.model_dump(), dict)


def test_prompt_is_type_specialized():
    inv = build_extraction_system_prompt(DocumentType.INVOICE)
    awb = build_extraction_system_prompt(DocumentType.AIR_WAYBILL)
    assert "invoice" in inv.lower()
    assert "awb" in awb.lower() or "waybill" in awb.lower()
    assert inv != awb


# --------------------------- unit (fake LLM) ------------------------------- #
def test_extract_text_path():
    expected = InvoiceData(invoice_number="HQPL00073841", currency="EUR")
    fake = FakeChatModel(structured_response=expected)
    extractor = StructuredExtractor(model=fake)
    res = extractor.extract(DocumentType.INVOICE, text="INVOICE No. HQPL00073841 ... EUR")
    assert isinstance(res, ExtractionResult)
    assert res.method == "text"
    assert res.data.invoice_number == "HQPL00073841"
    assert "HQPL00073841" in str(fake.captured[0])


def test_extract_vision_path_uses_images_and_vision_model():
    text_model = FakeChatModel(structured_response=InvoiceData())
    vision_model = FakeChatModel(structured_response=InvoiceData(invoice_number="V"))
    extractor = StructuredExtractor(model=text_model, vision_model=vision_model)
    # vision is opt-in now (plan default is text-based); request it explicitly
    res = extractor.extract(DocumentType.INVOICE, text="t", images=[b"\x89PNGfake"], use_vision=True)
    assert res.method == "vision"
    assert res.data.invoice_number == "V"
    assert vision_model.captured and not text_model.captured
    # the human message carried an image block
    human = vision_model.captured[0][1]
    blocks = human["content"]
    assert any(b.get("type") == "image_url" for b in blocks)


def test_extract_falls_back_to_text_without_images():
    text_model = FakeChatModel(structured_response=InvoiceData(invoice_number="T"))
    vision_model = FakeChatModel(structured_response=InvoiceData(invoice_number="V"))
    extractor = StructuredExtractor(model=text_model, vision_model=vision_model)
    res = extractor.extract(DocumentType.INVOICE, text="text only", images=None)
    assert res.method == "text"
    assert res.data.invoice_number == "T"


def test_extract_error_returns_empty_instance():
    fake = FakeChatModel(raise_on_invoke=RuntimeError("boom"))
    extractor = StructuredExtractor(model=fake)
    res = extractor.extract(DocumentType.INVOICE, text="something long enough")
    assert res.error is not None and "boom" in res.error
    assert isinstance(res.data, InvoiceData)
    assert res.data.invoice_number is None


# --------------------------- live integration ------------------------------ #
@requires_ollama
@pytest.mark.integration
def test_extract_real_invoice():
    # Text-first (the production default): searchable PDF text -> staged extraction.
    doc = ingest_file("sample_files/инв.PDF")
    text = OCRService().process(doc).full_text
    res = StructuredExtractor().extract(DocumentType.INVOICE, text=text)
    assert res.error is None
    data: InvoiceData = res.data
    assert data.invoice_number and "1090130561" in data.invoice_number.replace(" ", "")
    assert len(data.line_items) >= 1
    assert any(li.description for li in data.line_items)


@requires_ollama
@pytest.mark.integration
def test_extract_real_awb():
    # Scanned image -> PaddleOCR text -> staged extraction.
    doc = ingest_file("sample_files/Avia.jpg")
    text = OCRService().process(doc).full_text
    res = StructuredExtractor().extract(DocumentType.AIR_WAYBILL, text=text)
    assert res.error is None
    awb: AirWaybillData = res.data
    assert awb.awb_number and ("4000" in awb.awb_number or "488" in awb.awb_number)


@requires_ollama
@pytest.mark.integration
def test_extract_real_customs_declaration():
    doc = ingest_file("sample_files/gtd.jpg")
    text = OCRService().process(doc).full_text
    res = StructuredExtractor().extract(DocumentType.CUSTOMS_DECLARATION, text=text)
    assert res.error is None
    gtd: CustomsDeclarationData = res.data
    assert (gtd.mrn or gtd.declaration_type)
    assert len(gtd.items) >= 1  # the GTD has 2 commodity items


@requires_ollama
@pytest.mark.integration
def test_extract_real_letter_text():
    doc = ingest_file("sample_files/BVS.pdf")
    res = StructuredExtractor().extract(
        DocumentType.LETTER, text=doc.full_text, images=None
    )
    assert res.error is None
    letter: LetterData = res.data
    assert letter.summary
