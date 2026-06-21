"""Smart OCR service: searchable-text first, image OCR fallback.

Implements the assignment's "smart" requirement: for each page, if the file
already carries enough embedded text we use it directly (fast + exact); only
otherwise do we render the page and run the OCR engine.
"""

from __future__ import annotations

from app.config import get_settings
from app.ingestion import IngestedDocument, render_page_png
from app.ocr.base import OCRMethod, OCRProvider, OCRResult, PageOCR


def get_ocr_provider(name: str | None = None) -> OCRProvider:
    """Build the configured OCR provider (PaddleOCR — the mandated engine)."""
    name = (name or get_settings().ocr_provider).lower()
    if name in ("paddle", "paddleocr"):
        from app.ocr.paddle import PaddleOCRProvider

        return PaddleOCRProvider()
    raise ValueError(f"Unknown OCR provider: {name!r} (only 'paddle' is supported)")


class OCRService:
    """Run smart OCR over an :class:`IngestedDocument`."""

    def __init__(self, provider: OCRProvider | None = None, threshold: int | None = None):
        self._provider = provider
        self.threshold = (
            threshold if threshold is not None else get_settings().searchable_text_threshold
        )

    @property
    def provider(self) -> OCRProvider:
        if self._provider is None:
            self._provider = get_ocr_provider()
        return self._provider

    def process(self, doc: IngestedDocument, language_hint: str | None = None) -> OCRResult:
        page_results: list[PageOCR] = []
        for page in doc.pages:
            embedded = page.text.strip()

            # 1) Searchable: enough embedded text -> use it directly.
            if len(embedded) >= self.threshold:
                page_results.append(PageOCR(page.page_number, embedded, OCRMethod.SEARCHABLE))
                continue

            # 2) Renderable (PDF/image) -> run the OCR engine on the page image.
            if doc.can_render_images:
                png = render_page_png(doc, page.page_number)
                if png is not None:
                    text = self.provider.ocr_image(png, language_hint=language_hint).strip()
                    method = OCRMethod.OCR if text else OCRMethod.NONE
                    page_results.append(PageOCR(page.page_number, text, method))
                    continue

            # 3) Non-renderable text source (DOCX) with short text — keep as-is.
            if embedded:
                page_results.append(PageOCR(page.page_number, embedded, OCRMethod.TEXT))
            else:
                page_results.append(PageOCR(page.page_number, "", OCRMethod.NONE))

        return OCRResult(pages=page_results, provider=self.provider.name)
