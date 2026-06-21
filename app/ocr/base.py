"""OCR provider interface + result types.

The smart-OCR service depends only on this interface; PaddleOCR is the mandated
engine, and any other engine can be added behind this interface with no impact
on the rest of the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class OCRMethod(str, Enum):
    SEARCHABLE = "searchable"  # used the file's embedded text layer (fast, exact)
    OCR = "ocr"  # rendered the page and ran the OCR engine
    TEXT = "text"  # non-renderable text source (e.g. DOCX)
    NONE = "none"  # nothing could be extracted


@dataclass
class PageOCR:
    page_number: int
    text: str
    method: OCRMethod


@dataclass
class OCRResult:
    pages: list[PageOCR] = field(default_factory=list)
    provider: str = ""

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text).strip()

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)

    @property
    def method_summary(self) -> str:
        """Single label describing how the document overall was read."""
        methods = {p.method for p in self.pages if p.method is not OCRMethod.NONE}
        if not methods:
            return OCRMethod.NONE.value
        if methods == {OCRMethod.SEARCHABLE}:
            return OCRMethod.SEARCHABLE.value
        if methods == {OCRMethod.OCR}:
            return OCRMethod.OCR.value
        if methods == {OCRMethod.TEXT}:
            return OCRMethod.TEXT.value
        return "mixed"


class OCRProvider(ABC):
    """Transcribe a single rendered page image to raw text."""

    name: str = "base"

    @abstractmethod
    def ocr_image(self, image_png: bytes, language_hint: str | None = None) -> str:
        """Return the text contained in a PNG-encoded page image."""
        raise NotImplementedError
