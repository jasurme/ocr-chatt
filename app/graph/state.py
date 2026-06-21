"""Shared state for the document-processing LangGraph."""

from __future__ import annotations

from operator import add
from typing import Annotated, Any

from typing_extensions import TypedDict


class PipelineState(TypedDict, total=False):
    """State threaded through the document-processing graph.

    `total=False` lets each node return only the keys it produces. The `steps`
    and `errors` channels use the `add` reducer so every node can append without
    clobbering earlier entries (handy for tracing/debugging the run).
    """

    # ---- inputs ----
    file_path: str
    filename: str

    # ---- ingest ----
    document: Any  # IngestedDocument (in-memory only)
    file_type: str
    num_pages: int
    supported: bool

    # ---- ocr ----
    ocr_result: Any  # OCRResult
    ocr_text: str
    ocr_method: str

    # ---- classify ----
    doc_type: str
    language: str
    confidence: float
    classification: dict

    # ---- extract ----
    extracted: dict
    extraction_method: str

    # ---- control / tracing ----
    status: str
    steps: Annotated[list[str], add]
    errors: Annotated[list[str], add]
