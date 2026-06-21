"""Document ingestion: turn an uploaded file (PDF/JPG/PNG/DOCX) into a
normalized in-memory representation (per-page text + on-demand page images).

This layer is intentionally OCR-agnostic: it extracts whatever the file already
contains (embedded PDF text, DOCX text) and exposes the ability to rasterize
pages to PNG. The smart-OCR layer decides per page whether the embedded text is
enough or whether to fall back to image OCR.
"""

from app.ingestion.models import FileType, IngestedDocument, Page, UnsupportedFileError
from app.ingestion.loader import detect_file_type, ingest_file, iter_page_images, render_page_png

__all__ = [
    "FileType",
    "IngestedDocument",
    "Page",
    "UnsupportedFileError",
    "detect_file_type",
    "ingest_file",
    "iter_page_images",
    "render_page_png",
]
