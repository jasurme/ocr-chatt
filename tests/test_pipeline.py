"""Tests for the LangGraph document-processing pipeline (Step 6)."""

from __future__ import annotations

import json

import pytest

from app.classification import Classification, DocumentClassifier
from app.extraction import StructuredExtractor
from app.graph import (
    build_pipeline,
    process_document,
    route_after_classify,
    route_after_ingest,
)
from app.ocr import OCRService
from app.schemas import AirWaybillData, InvoiceData, LetterData
from app.schemas.types import DocumentType
from conftest import requires_ollama
from tests._fakes import FakeChatModel, FakeOCRProvider


def make_fake_pipeline(doc_type=DocumentType.INVOICE, extracted=None, ocr_text=None):
    provider = FakeOCRProvider(text=ocr_text) if ocr_text else FakeOCRProvider()
    ocr = OCRService(provider=provider, threshold=100)
    clf_model = FakeChatModel(
        structured_response=Classification(
            doc_type=doc_type, language="en", confidence=0.9, reasoning="fake"
        )
    )
    classifier = DocumentClassifier(model=clf_model, vision_model=clf_model)
    ext_model = FakeChatModel(structured_response=extracted or InvoiceData(invoice_number="INV-1"))
    extractor = StructuredExtractor(model=ext_model, vision_model=ext_model)
    return build_pipeline(ocr_service=ocr, classifier=classifier, extractor=extractor)


class _Doc:
    def __init__(self, can_render):
        self.can_render_images = can_render


# ----------------------------- routing ------------------------------------- #
def test_route_after_ingest():
    assert route_after_ingest({"supported": True}) == "ocr"
    assert route_after_ingest({"supported": False}) == "finalize"
    assert route_after_ingest({}) == "finalize"


def test_route_after_classify():
    assert route_after_classify({"document": _Doc(False), "ocr_text": "hello"}) == "extract"
    assert route_after_classify({"document": _Doc(True), "ocr_text": ""}) == "extract"
    assert route_after_classify({"document": _Doc(False), "ocr_text": "  "}) == "finalize"


# ----------------------- graph structure ----------------------------------- #
def test_pipeline_compiles_with_expected_nodes():
    graph = make_fake_pipeline()
    nodes = set(graph.get_graph().nodes)
    for n in ("ingest", "ocr", "classify", "extract", "finalize"):
        assert n in nodes


# --------------------- end-to-end (fake services) -------------------------- #
def test_corrupt_pdf_handled_gracefully(tmp_path):
    # Valid-looking header but broken bytes must not crash the pipeline.
    f = tmp_path / "broken.pdf"
    f.write_bytes(b"%PDF-1.4 not a real pdf body \x00\x01 broken")
    result = process_document(f, pipeline=make_fake_pipeline())
    assert result["status"] in ("unsupported", "error")
    assert result["extracted"] is None
    assert any("ingest" in e for e in result["errors"])


def test_corrupt_image_handled_gracefully(tmp_path):
    f = tmp_path / "broken.png"
    f.write_bytes(b"\x89PNG\r\n\x1a\n garbage not a real image")
    result = process_document(f, pipeline=make_fake_pipeline())
    assert result["status"] in ("unsupported", "error")


def test_ocr_engine_failure_does_not_crash(samples):
    # OCR engine raising (e.g. API outage) -> pipeline still completes gracefully.
    provider = FakeOCRProvider(raise_on_ocr=RuntimeError("OCR API down"))
    ocr = OCRService(provider=provider, threshold=100)
    clf = DocumentClassifier(
        model=FakeChatModel(structured_response=Classification(
            doc_type=DocumentType.OTHER, language="en", confidence=0.1, reasoning="x")),
        vision_model=FakeChatModel(structured_response=Classification(
            doc_type=DocumentType.OTHER, language="en", confidence=0.1, reasoning="x")),
    )
    ext = StructuredExtractor(
        model=FakeChatModel(structured_response=AirWaybillData()),
        vision_model=FakeChatModel(structured_response=AirWaybillData()),
    )
    pipe = build_pipeline(ocr_service=ocr, classifier=clf, extractor=ext)
    result = process_document(samples["awb_jpg"], pipeline=pipe)  # image -> needs OCR
    assert result["status"].startswith("completed")  # no crash
    assert "ocr:failed" in result["steps"]
    assert any("ocr" in e for e in result["errors"])


def test_unsupported_file_routes_to_finalize(tmp_path):
    f = tmp_path / "thing.txt"
    f.write_text("plain text, unsupported")
    result = process_document(f, pipeline=make_fake_pipeline())
    assert result["status"] == "unsupported"
    assert "ingest:failed" in result["steps"]
    assert not any(s.startswith("ocr") for s in result["steps"])
    assert result["extracted"] is None


def test_image_document_full_path(samples):
    pipe = make_fake_pipeline(
        doc_type=DocumentType.AIR_WAYBILL, extracted=AirWaybillData(awb_number="488-4000")
    )
    result = process_document(samples["awb_jpg"], pipeline=pipe)
    assert result["status"] == "completed"
    assert result["file_type"] == "image"
    assert result["doc_type"] == "air_waybill"
    assert result["ocr_method"] == "ocr"  # image -> OCR engine
    steps = result["steps"]
    assert steps == ["ingest", "ocr:ocr", "classify:air_waybill", "extract:air_waybill", "finalize"]
    assert result["extracted"]["awb_number"] == "488-4000"


def test_searchable_pdf_uses_searchable_method(samples):
    pipe = make_fake_pipeline(doc_type=DocumentType.LETTER, extracted=LetterData(summary="hi"))
    result = process_document(samples["letter_pdf"], pipeline=pipe)
    assert result["status"] == "completed"
    assert result["ocr_method"] == "searchable"
    assert "classify:letter" in result["steps"]
    assert result["language"] == "en"


def test_docx_document_path(samples):
    pipe = make_fake_pipeline(doc_type=DocumentType.OTHER)
    result = process_document(samples["junk_docx"], pipeline=pipe)
    assert result["status"] == "completed"
    assert result["file_type"] == "docx"
    assert result["ocr_method"] == "text"


def test_result_is_json_serializable(samples):
    pipe = make_fake_pipeline(doc_type=DocumentType.AIR_WAYBILL, extracted=AirWaybillData())
    result = process_document(samples["awb_jpg"], pipeline=pipe)
    assert "document" not in result and "ocr_result" not in result
    json.dumps(result)  # must not raise


# --------------------------- live integration ------------------------------ #
@requires_ollama
@pytest.mark.integration
def test_end_to_end_real_awb(samples):
    result = process_document(samples["awb_jpg"])  # real services
    assert result["status"] == "completed"
    assert result["doc_type"] == "air_waybill"
    assert result["extracted"]["awb_number"]
    assert result["steps"][0] == "ingest"
    assert result["steps"][-1] == "finalize"
