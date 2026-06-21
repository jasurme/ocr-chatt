"""Document-processing pipeline as a LangGraph ``StateGraph``.

Topology (matches the assignment's architectural flowchart)::

    START → ingest ─┬─(unsupported)─────────────→ finalize → END
                    └─(ok)→ ocr → classify ─┬─(extract)→ extract → finalize
                                            └─(skip)──────────────→ finalize

* ``ingest``   — load the file (PDF/JPG/PNG/DOCX) into a normalized document.
* ``ocr``      — smart OCR: searchable-text-first, image-OCR fallback (per page).
* ``classify`` — LLM classification into a :class:`DocumentType`.
* ``extract``  — type-specialized structured extraction into typed JSON.
* ``finalize`` — set the final status.

Routing is done with real conditional edges (``route_after_ingest`` /
``route_after_classify``) so unsupported files are handled gracefully and we
never burn an LLM call on a document with nothing to read. Services are injected
so the whole graph can be unit-tested without any network access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.classification import DocumentClassifier
from app.config import get_settings
from app.extraction import StructuredExtractor
from app.graph.state import PipelineState
from app.ingestion import UnsupportedFileError, ingest_file, render_page_png
from app.ocr import OCRService
from app.schemas.types import DocumentType

# Below this many chars we let the classifier look at the page image instead.
_MIN_TEXT_FOR_TEXT = 30


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def ingest_node(state: PipelineState) -> dict:
    path = state["file_path"]
    try:
        doc = ingest_file(path)
    except (UnsupportedFileError, FileNotFoundError) as exc:
        return {
            "filename": Path(path).name,
            "supported": False,
            "status": "unsupported",
            "errors": [f"ingest: {exc}"],
            "steps": ["ingest:failed"],
        }
    except Exception as exc:  # never crash the pipeline on a damaged file
        return {
            "filename": Path(path).name,
            "supported": False,
            "status": "error",
            "errors": [f"ingest: {type(exc).__name__}: {exc}"],
            "steps": ["ingest:failed"],
        }
    return {
        "document": doc,
        "filename": doc.filename,
        "file_type": doc.file_type.value,
        "num_pages": doc.num_pages,
        "supported": True,
        "steps": ["ingest"],
    }


def route_after_ingest(state: PipelineState) -> str:
    return "ocr" if state.get("supported") else "finalize"


def make_ocr_node(ocr_service: OCRService):
    def ocr_node(state: PipelineState) -> dict:
        # OCR failures (engine/API/render errors) must not crash the pipeline —
        # downstream classify/extract degrade gracefully on empty text.
        try:
            result = ocr_service.process(state["document"])
        except Exception as exc:
            return {
                "ocr_text": "",
                "ocr_method": "none",
                "errors": [f"ocr: {type(exc).__name__}: {exc}"],
                "steps": ["ocr:failed"],
            }
        return {
            "ocr_result": result,
            "ocr_text": result.full_text,
            "ocr_method": result.method_summary,
            "steps": [f"ocr:{result.method_summary}"],
        }

    return ocr_node


def make_classify_node(classifier: DocumentClassifier):
    def classify_node(state: PipelineState) -> dict:
        settings = get_settings()
        doc = state["document"]
        text = state.get("ocr_text", "")
        image = None
        # Vision fallback for image-only docs is used only when vision is enabled;
        # otherwise we classify from the (PaddleOCR) text alone.
        if (
            settings.enable_vision
            and len(text.strip()) < _MIN_TEXT_FOR_TEXT
            and doc.can_render_images
        ):
            image = render_page_png(doc, 1)
        c = classifier.classify(text=text, image_png=image, filename=state.get("filename"))
        return {
            "doc_type": c.doc_type.value,
            "language": c.language,
            "confidence": c.confidence,
            "classification": c.model_dump(),
            "steps": [f"classify:{c.doc_type.value}"],
        }

    return classify_node


def route_after_classify(state: PipelineState) -> str:
    doc = state.get("document")
    has_text = bool(state.get("ocr_text", "").strip())
    can_render = bool(doc and doc.can_render_images)
    # Nothing to read at all -> skip extraction (graceful).
    return "extract" if (has_text or can_render) else "finalize"


def make_extract_node(extractor: StructuredExtractor):
    def extract_node(state: PipelineState) -> dict:
        settings = get_settings()
        doc = state["document"]
        doc_type = DocumentType(state["doc_type"])
        text = state.get("ocr_text", "") or ""
        # Prefer staged TEXT extraction whenever we have real text (searchable PDFs or
        # successful OCR): it is far more reliable than vision for big nested schemas on
        # a small local model. Fall back to vision only when there is little/no text.
        has_text = len(text.strip()) >= settings.searchable_text_threshold

        images: list[bytes] | None = None
        use_vision = False
        if (
            settings.enable_vision
            and not has_text
            and settings.prefer_vision_extraction
            and doc.can_render_images
        ):
            n = min(doc.num_pages, settings.extraction_max_images)
            images = [png for png in (render_page_png(doc, p) for p in range(1, n + 1)) if png]
            use_vision = bool(images)

        res = extractor.extract(
            doc_type,
            text=text,
            images=images,
            filename=state.get("filename"),
            use_vision=use_vision,
        )
        out: dict[str, Any] = {
            "extracted": res.data_dict,
            "extraction_method": res.method,
            "steps": [f"extract:{doc_type.value}"],
        }
        if res.error:
            out["errors"] = [f"extract: {res.error}"]
        return out

    return extract_node


def finalize_node(state: PipelineState) -> dict:
    status = state.get("status")
    if not status:
        status = "completed" if state.get("extracted") else "completed_no_extraction"
    return {"status": status, "steps": ["finalize"]}


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #
def build_pipeline(
    ocr_service: OCRService | None = None,
    classifier: DocumentClassifier | None = None,
    extractor: StructuredExtractor | None = None,
):
    """Build and compile the document-processing graph.

    Pass services to inject fakes in tests; defaults build the real (lazy)
    local model-backed services without making any network call at construction time.
    """
    ocr_service = ocr_service or OCRService()
    classifier = classifier or DocumentClassifier()
    extractor = extractor or StructuredExtractor()

    g = StateGraph(PipelineState)
    g.add_node("ingest", ingest_node)
    g.add_node("ocr", make_ocr_node(ocr_service))
    g.add_node("classify", make_classify_node(classifier))
    g.add_node("extract", make_extract_node(extractor))
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "ingest")
    g.add_conditional_edges(
        "ingest", route_after_ingest, {"ocr": "ocr", "finalize": "finalize"}
    )
    g.add_edge("ocr", "classify")
    g.add_conditional_edges(
        "classify", route_after_classify, {"extract": "extract", "finalize": "finalize"}
    )
    g.add_edge("extract", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


# Result keys that are JSON-serializable and useful to callers (the UI/API).
_RESULT_KEYS = (
    "filename",
    "file_type",
    "num_pages",
    "doc_type",
    "language",
    "confidence",
    "classification",
    "ocr_method",
    "ocr_text",
    "extracted",
    "extraction_method",
    "status",
    "steps",
    "errors",
)


def clean_result(final_state: dict) -> dict:
    """Project the raw graph state down to a JSON-serializable result dict."""
    out = {k: final_state.get(k) for k in _RESULT_KEYS}
    out.setdefault("steps", [])
    out.setdefault("errors", [])
    return out


def process_document(file_path: str | Path, pipeline=None) -> dict:
    """Run a file through the pipeline and return a clean result dict."""
    pipeline = pipeline or build_pipeline()
    final = pipeline.invoke({"file_path": str(file_path)})
    return clean_result(final)
