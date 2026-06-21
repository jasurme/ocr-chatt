"""Tests for the ingestion layer (Step 2).

These run fully offline against the bundled sample files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion import (
    FileType,
    UnsupportedFileError,
    detect_file_type,
    ingest_file,
    iter_page_images,
    render_page_png,
)

_PNG_SIG = b"\x89PNG\r\n\x1a\n"


# ----------------------------- detection ----------------------------------- #
def test_detect_pdf(samples):
    assert detect_file_type(samples["invoice_pdf"]) is FileType.PDF
    assert detect_file_type(samples["letter_pdf"]) is FileType.PDF
    assert detect_file_type(samples["invoice_pdf2"]) is FileType.PDF  # .PDF uppercase


def test_detect_image(samples):
    assert detect_file_type(samples["awb_jpg"]) is FileType.IMAGE
    assert detect_file_type(samples["gtd_jpg"]) is FileType.IMAGE


def test_detect_docx(samples):
    assert detect_file_type(samples["junk_docx"]) is FileType.DOCX


def test_detect_by_magic_bytes_over_wrong_extension(tmp_path, samples):
    # A PDF renamed to .txt should still be detected as PDF via magic bytes.
    fake = tmp_path / "actually_a_pdf.txt"
    fake.write_bytes(samples["letter_pdf"].read_bytes())
    assert detect_file_type(fake) is FileType.PDF


def test_detect_unsupported(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("just some plain text, not a real doc")
    assert detect_file_type(f) is FileType.UNSUPPORTED


# ----------------------------- ingestion ----------------------------------- #
def test_ingest_searchable_pdf_has_text(samples):
    doc = ingest_file(samples["letter_pdf"])  # BVS.pdf is searchable (Russian)
    assert doc.file_type is FileType.PDF
    assert doc.num_pages == 1
    assert doc.embedded_char_count > 200
    assert "авианакладной" in doc.full_text or "MEDICAL" in doc.full_text.upper()
    assert doc.has_searchable_text(per_page_threshold=100)


def test_ingest_multipage_pdf(samples):
    doc = ingest_file(samples["invoice_pdf"])  # 8 pages
    assert doc.num_pages == 8
    assert doc.metadata["page_count_total"] == 8
    assert doc.full_text  # has some embedded text (even if garbled)


def test_ingest_image_no_embedded_text(samples):
    doc = ingest_file(samples["awb_jpg"])
    assert doc.file_type is FileType.IMAGE
    assert doc.num_pages == 1
    assert doc.pages[0].text == ""
    assert not doc.has_searchable_text(per_page_threshold=100)
    assert doc.pages[0].width > 0 and doc.pages[0].height > 0


def test_ingest_docx_text(samples):
    doc = ingest_file(samples["junk_docx"])
    assert doc.file_type is FileType.DOCX
    assert "title" in doc.full_text.lower()
    assert "summary" in doc.full_text.lower()
    assert not doc.can_render_images


def test_ingest_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        ingest_file("/no/such/file.pdf")


def test_ingest_unsupported_raises(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("plain text")
    with pytest.raises(UnsupportedFileError):
        ingest_file(f)


# ----------------------------- rendering ----------------------------------- #
def test_render_pdf_page_returns_png(samples):
    doc = ingest_file(samples["invoice_pdf2"])
    png = render_page_png(doc, 1, dpi=100)
    assert png is not None and png.startswith(_PNG_SIG)
    assert len(png) > 1000


def test_render_image_page_returns_png(samples):
    doc = ingest_file(samples["gtd_jpg"])
    png = render_page_png(doc, 1)
    assert png is not None and png.startswith(_PNG_SIG)


def test_render_docx_returns_none(samples):
    doc = ingest_file(samples["junk_docx"])
    assert render_page_png(doc, 1) is None


def test_render_out_of_range_page_returns_none(samples):
    doc = ingest_file(samples["awb_jpg"])
    assert render_page_png(doc, 99) is None


def test_iter_page_images_pdf(samples):
    doc = ingest_file(samples["invoice_pdf"])
    images = list(iter_page_images(doc, dpi=80))
    assert len(images) == 8
    for pn, png in images:
        assert 1 <= pn <= 8
        assert png.startswith(_PNG_SIG)


def test_iter_page_images_only_selected(samples):
    doc = ingest_file(samples["invoice_pdf"])
    images = list(iter_page_images(doc, dpi=80, only_pages=[1, 3]))
    assert [pn for pn, _ in images] == [1, 3]


def test_iter_page_images_docx_empty(samples):
    doc = ingest_file(samples["junk_docx"])
    assert list(iter_page_images(doc)) == []
