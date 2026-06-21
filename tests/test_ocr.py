"""Tests for the smart OCR service (Step 3).

Unit tests use a fake OCR provider (no network). The live integration test
(`-m integration`) exercises the real PaddleOCR engine.
"""

from __future__ import annotations

import pytest

from app.ingestion import ingest_file
from app.ocr import OCRMethod, OCRProvider, OCRResult, PageOCR
from app.ocr.service import OCRService, get_ocr_provider

_PNG_SIG = b"\x89PNG\r\n\x1a\n"


class FakeOCR(OCRProvider):
    name = "fake"

    def __init__(self, text: str = "FAKE OCR TEXT FROM IMAGE"):
        self.text = text
        self.calls = 0
        self.images: list[bytes] = []
        self.hints: list[str | None] = []

    def ocr_image(self, image_png: bytes, language_hint: str | None = None) -> str:
        self.calls += 1
        self.images.append(image_png)
        self.hints.append(language_hint)
        return self.text


# --------------------------- searchable-first ------------------------------ #
def test_searchable_pdf_uses_embedded_text_not_ocr(samples):
    fake = FakeOCR()
    svc = OCRService(provider=fake, threshold=100)
    doc = ingest_file(samples["letter_pdf"])
    result = svc.process(doc)
    assert fake.calls == 0  # never touched the OCR engine
    assert result.method_summary == "searchable"
    assert result.char_count > 200


def test_multipage_pdf_mixed_searchable_and_ocr(samples):
    # Real document: 7 pages have a good text layer, page 6 is sparse (95 chars)
    # and must fall back to image OCR -> demonstrates the smart per-page routing.
    fake = FakeOCR(text="OCR FALLBACK PAGE")
    svc = OCRService(provider=fake, threshold=100)
    doc = ingest_file(samples["invoice_pdf"])
    result = svc.process(doc)
    assert len(result.pages) == 8
    assert fake.calls == 1  # only the one sparse page
    searchable = [p for p in result.pages if p.method is OCRMethod.SEARCHABLE]
    ocred = [p for p in result.pages if p.method is OCRMethod.OCR]
    assert len(searchable) == 7
    assert len(ocred) == 1 and ocred[0].page_number == 6
    assert result.method_summary == "mixed"


def test_all_searchable_when_threshold_low(samples):
    fake = FakeOCR()
    svc = OCRService(provider=fake, threshold=50)  # page 6 (95 chars) now qualifies
    doc = ingest_file(samples["invoice_pdf"])
    result = svc.process(doc)
    assert fake.calls == 0
    assert all(p.method is OCRMethod.SEARCHABLE for p in result.pages)
    assert result.method_summary == "searchable"


# --------------------------- OCR fallback ---------------------------------- #
def test_image_triggers_ocr_engine(samples):
    fake = FakeOCR(text="TRANSCRIBED AIR WAYBILL")
    svc = OCRService(provider=fake, threshold=100)
    doc = ingest_file(samples["awb_jpg"])
    result = svc.process(doc)
    assert fake.calls == 1
    assert fake.images[0].startswith(_PNG_SIG)  # got a rendered PNG
    assert result.pages[0].method is OCRMethod.OCR
    assert result.full_text == "TRANSCRIBED AIR WAYBILL"
    assert result.method_summary == "ocr"


def test_high_threshold_forces_ocr_on_pdf(samples):
    # An absurd threshold makes even a searchable PDF fall back to image OCR.
    fake = FakeOCR(text="OCR PAGE")
    svc = OCRService(provider=fake, threshold=10_000_000)
    doc = ingest_file(samples["invoice_pdf2"])  # 9 pages
    result = svc.process(doc)
    assert fake.calls == doc.num_pages
    assert all(p.method is OCRMethod.OCR for p in result.pages)


def test_language_hint_forwarded(samples):
    fake = FakeOCR()
    svc = OCRService(provider=fake, threshold=100)
    doc = ingest_file(samples["gtd_jpg"])
    svc.process(doc, language_hint="Polish")
    assert fake.hints == ["Polish"]


# --------------------------- DOCX / text source ---------------------------- #
def test_docx_uses_text_method(samples):
    fake = FakeOCR()
    svc = OCRService(provider=fake, threshold=100)
    doc = ingest_file(samples["junk_docx"])  # short text, not renderable
    result = svc.process(doc)
    assert fake.calls == 0
    assert result.pages[0].method is OCRMethod.TEXT
    assert "summary" in result.full_text.lower()


# --------------------------- result helpers -------------------------------- #
def test_method_summary_mixed():
    res = OCRResult(
        pages=[
            PageOCR(1, "a", OCRMethod.SEARCHABLE),
            PageOCR(2, "b", OCRMethod.OCR),
        ]
    )
    assert res.method_summary == "mixed"


def test_method_summary_none_when_empty():
    res = OCRResult(pages=[PageOCR(1, "", OCRMethod.NONE)])
    assert res.method_summary == "none"


# --------------------------- provider factory ------------------------------ #
def test_factory_default_is_paddle():
    # PaddleOCR is the mandated default engine.
    assert get_ocr_provider().name == "paddle"


def test_factory_paddle_returns_provider():
    assert get_ocr_provider("paddle").name == "paddle"


def test_factory_unknown_raises():
    with pytest.raises(ValueError):
        get_ocr_provider("totally-not-a-provider")


# --------------------------- live integration ------------------------------ #
@pytest.mark.integration
@pytest.mark.slow
def test_paddle_ocr_reads_real_image(samples):
    # The mandated default engine, end-to-end (pure local PaddleOCR).
    svc = OCRService(provider=get_ocr_provider("paddle"), threshold=100)
    doc = ingest_file(samples["gtd_jpg"])
    result = svc.process(doc)
    assert result.provider == "paddle"
    assert result.pages[0].method is OCRMethod.OCR
    text = result.full_text.upper()
    assert len(text) > 200
    assert any(tok in text for tok in ["EXPORT", "UNIA", "EUROPEJSKA", "EX", "PL"])
