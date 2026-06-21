"""Classify a document into one of the supported :class:`DocumentType` values.

Text-first (cheap, matches the assignment's "raw text -> classify" step); falls
back to the page image when there is little/no extractable text. Designed for
dependency injection so it can be unit-tested without network access.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.llm import get_chat_model, get_vision_model, image_data_url
from app.schemas.types import DocumentType, classifier_type_menu

# Below this many non-space characters we trust the image more than the text.
_MIN_TEXT_FOR_TEXT_CLASSIFY = 30
# Cap text sent to the model — the type is obvious from the first portion.
_MAX_TEXT_CHARS = 6000


class Classification(BaseModel):
    """Structured classifier output."""

    doc_type: DocumentType = Field(description="The single best-fitting document type.")
    language: str = Field(
        description="Primary language as an ISO-639 code: uz, ru, en, pl, etc."
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence 0..1.")
    reasoning: str = Field(description="One short sentence justifying the choice.")


_SYSTEM = (
    "You are an expert document classifier for a customs brokerage. "
    "Classify the document into EXACTLY ONE of these types:\n"
    "{menu}\n\n"
    "Also detect the primary language (ISO-639 code: uz, ru, en, pl, ...). "
    "Give a confidence between 0 and 1 and a one-sentence reason. "
    "If the document is unreadable or fits nothing, use 'other'."
)


class DocumentClassifier:
    def __init__(self, model=None, vision_model=None):
        self._model = model
        self._vision_model = vision_model

    def _text_runnable(self):
        model = self._model if self._model is not None else get_chat_model()
        return model.with_structured_output(Classification)

    def _vision_runnable(self):
        model = self._vision_model if self._vision_model is not None else get_vision_model()
        return model.with_structured_output(Classification)

    def classify(
        self,
        text: str = "",
        image_png: bytes | None = None,
        filename: str | None = None,
    ) -> Classification:
        system = _SYSTEM.format(menu=classifier_type_menu())
        clean = (text or "").strip()
        hint = f"\n\nFile name: {filename}" if filename else ""

        try:
            if len(clean) >= _MIN_TEXT_FOR_TEXT_CLASSIFY or image_png is None:
                user = (
                    "Classify this document.\n\nDOCUMENT TEXT:\n"
                    f"{clean[:_MAX_TEXT_CHARS]}{hint}"
                )
                messages = [("system", system), ("human", user)]
                return self._text_runnable().invoke(messages)

            # Little/no text -> classify from the page image.
            human = {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Classify this document from its image.{hint}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url(image_png), "detail": "high"},
                    },
                ],
            }
            return self._vision_runnable().invoke([("system", system), human])
        except Exception as exc:  # robustness: never crash the pipeline on classify
            return Classification(
                doc_type=DocumentType.OTHER,
                language="unknown",
                confidence=0.0,
                reasoning=f"Classification failed: {type(exc).__name__}",
            )
