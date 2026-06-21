"""Smart OCR: searchable-text-first with image-OCR fallback (pluggable engine)."""

from app.ocr.base import OCRMethod, OCRProvider, OCRResult, PageOCR
from app.ocr.service import OCRService, get_ocr_provider

__all__ = [
    "OCRMethod",
    "OCRProvider",
    "OCRResult",
    "PageOCR",
    "OCRService",
    "get_ocr_provider",
]
