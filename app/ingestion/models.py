"""Data models for ingested documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class FileType(str, Enum):
    PDF = "pdf"
    IMAGE = "image"
    DOCX = "docx"
    UNSUPPORTED = "unsupported"


class UnsupportedFileError(ValueError):
    """Raised when a file's type cannot be handled by the pipeline."""


@dataclass
class Page:
    """A single logical page of a document.

    `text` holds whatever text could be extracted *without* OCR (embedded PDF
    text layer, or DOCX text). It may be empty for scanned PDFs / photos, in
    which case the OCR layer fills it in from the page image.
    """

    page_number: int  # 1-based
    text: str = ""
    width: int = 0  # pixels (for images) or points (for PDFs); 0 if unknown
    height: int = 0
    has_embedded_text: bool = False  # whether `text` came from the file itself


@dataclass
class IngestedDocument:
    """Normalized representation of an uploaded file."""

    filename: str
    file_type: FileType
    source_path: Path
    pages: list[Page] = field(default_factory=list)
    mime: str = ""
    encrypted: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    @property
    def full_text(self) -> str:
        """All embedded page text joined (page-break separated)."""
        return "\n\n".join(p.text for p in self.pages if p.text).strip()

    @property
    def embedded_char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)

    def has_searchable_text(self, per_page_threshold: int) -> bool:
        """True if *any* page carries enough embedded text to skip OCR."""
        return any(len(p.text.strip()) >= per_page_threshold for p in self.pages)

    @property
    def can_render_images(self) -> bool:
        """Whether page images can be produced (PDFs and image files; not DOCX)."""
        return self.file_type in (FileType.PDF, FileType.IMAGE)
