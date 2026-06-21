"""Structured extraction: turn OCR text and/or page images into typed JSON."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field, create_model
from pydantic_core import PydanticUndefined

from app.config import get_settings
from app.extraction.prompts import build_extraction_system_prompt
from app.llm import get_chat_model, get_vision_model, image_data_url
from app.schemas.documents import get_schema
from app.schemas.types import DocumentType


@dataclass
class ExtractionResult:
    doc_type: DocumentType
    data: BaseModel
    method: str = "text"  # "vision" | "text"
    error: Optional[str] = None

    @property
    def data_dict(self) -> dict:
        return self.data.model_dump()


def _field_spec(f):
    """Reproduce a model field's (annotation, default) for create_model."""
    if f.default is not PydanticUndefined:
        return (f.annotation, f.default)
    if f.default_factory is not None:
        return (f.annotation, Field(default_factory=f.default_factory))
    return (f.annotation, None)


def _partition_fields(schema: type[BaseModel]):
    """Split a schema's fields into (plain scalars, nested objects, list-of-object).

    A small local model collapses a large nested schema (returns empty lists / nulls),
    but reliably fills small focused schemas — so we extract each group separately.
    `list[str]` counts as plain; `Optional[SubModel]` is nested; `list[SubModel]` is a
    table extracted on its own.
    """
    plain: dict = {}
    nested: dict = {}
    lists: dict = {}
    for name, f in schema.model_fields.items():
        ann = f.annotation
        if typing.get_origin(ann) is list:
            args = typing.get_args(ann)
            if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
                lists[name] = args[0]
            else:
                plain[name] = _field_spec(f)
            continue
        member_types = [a for a in typing.get_args(ann) if isinstance(a, type)]
        if any(issubclass(a, BaseModel) for a in member_types):
            nested[name] = _field_spec(f)
        else:
            plain[name] = _field_spec(f)
    return plain, nested, lists


def _dedupe_cap(items: list[dict], cap: int) -> list[dict]:
    """Drop exact-duplicate rows (small models repeat lines) and cap the count."""
    seen: set = set()
    out: list[dict] = []
    for it in items:
        key = tuple(sorted((k, str(v)) for k, v in it.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
        if len(out) >= cap:
            break
    return out


class StructuredExtractor:
    def __init__(self, model=None, vision_model=None):
        self._model = model
        self._vision_model = vision_model

    def _text_model(self):
        return self._model if self._model is not None else get_chat_model()

    def _vision(self):
        return self._vision_model if self._vision_model is not None else get_vision_model()

    def extract(
        self,
        doc_type: DocumentType,
        text: str = "",
        images: list[bytes] | None = None,
        filename: str | None = None,
        use_vision: bool | None = None,
    ) -> ExtractionResult:
        settings = get_settings()
        schema = get_schema(doc_type)
        text = (text or "").strip()[: settings.max_prompt_chars]
        prefer = settings.prefer_vision_extraction if use_vision is None else use_vision
        use_vision = bool(images) and prefer

        if use_vision:
            try:
                system = build_extraction_system_prompt(doc_type)
                data = self._extract_vision(schema, system, text, images, filename)
                return ExtractionResult(doc_type=doc_type, data=data, method="vision")
            except Exception as exc_vision:
                # Vision can exceed the local model's context window (several page
                # images + OCR text). Fall back to staged text extraction before
                # giving up, so the extracted-data panel is still populated.
                if text:
                    try:
                        data = self._extract_staged(schema, doc_type, text, filename)
                        return ExtractionResult(doc_type=doc_type, data=data, method="text")
                    except Exception:
                        pass
                return ExtractionResult(
                    doc_type=doc_type, data=schema(), method="vision",
                    error=f"{type(exc_vision).__name__}: {exc_vision}",
                )

        try:
            data = self._extract_staged(schema, doc_type, text, filename)
            return ExtractionResult(doc_type=doc_type, data=data, method="text")
        except Exception as exc:
            # Robustness: return an empty (all-null) instance rather than crash.
            return ExtractionResult(
                doc_type=doc_type, data=schema(), method="text",
                error=f"{type(exc).__name__}: {exc}",
            )

    def _extract_staged(self, schema, doc_type, text, filename):
        """Extract a large schema in focused stages, then merge into one instance.

        A small local model (e.g. qwen2.5vl:7b, which has no tool-calling and must use
        json_schema decoding) returns empty lists / nulls for a big nested schema, but
        fills small focused ones. So we run: (1) plain scalar fields, (2) nested party
        objects, (3) each list/table on its own — and merge. Raises only if EVERY stage
        failed (so the caller surfaces an error); a partial result is kept otherwise.
        """
        settings = get_settings()
        base = build_extraction_system_prompt(doc_type)
        plain, nested, lists = _partition_fields(schema)
        model = self._text_model()
        hint = f"\n\nSource file: {filename}" if filename else ""
        human = f"Extract structured data from this document.{hint}\n\nDOCUMENT TEXT:\n{text}"

        def run(sub_schema, focus):
            system = f"{base}\n\nFOCUS FOR THIS STEP: {focus}"
            return model.with_structured_output(sub_schema).invoke(
                [("system", system), ("human", human)]
            )

        collected: dict = {}
        last_exc: Exception | None = None
        ok = False

        if plain:
            try:
                Pm = create_model(f"{schema.__name__}Scalars", **plain)
                collected.update(run(Pm, "Extract ONLY the top-level scalar fields below.").model_dump())
                ok = True
            except Exception as e:
                last_exc = e
        if nested:
            try:
                Nm = create_model(f"{schema.__name__}Parties", **nested)
                collected.update(
                    run(Nm, "Extract ONLY the party/entity objects (name, address, country, ids).").model_dump()
                )
                ok = True
            except Exception as e:
                last_exc = e
        for lname, item_cls in lists.items():
            try:
                Lm = create_model(
                    f"{schema.__name__}_{lname}",
                    **{lname: (list[item_cls], Field(default_factory=list))},
                )
                r = run(
                    Lm,
                    f"Extract EVERY row of the '{lname}' list/table. One object per row; "
                    "map each column to the correct field; merge wrapped lines into one "
                    "row; never invent rows that are not in the text.",
                )
                rows = [it.model_dump() for it in (getattr(r, lname, None) or [])]
                collected[lname] = _dedupe_cap(rows, settings.extraction_max_items)
                ok = True
            except Exception as e:
                last_exc = e

        # If every stage failed, surface the underlying error to the caller; a
        # partial success (some stages worked) is kept and merged.
        if not ok:
            raise last_exc if last_exc is not None else RuntimeError("staged extraction failed")
        return schema(**collected)

    def _extract_vision(self, schema, system, text, images, filename):
        settings = get_settings()
        runnable = self._vision().with_structured_output(schema)
        hint = f" Source file: {filename}." if filename else ""
        intro = (
            "Extract all fields from this document. Read the page image(s) as the "
            "primary source." + hint
        )
        if text:
            # Images are the primary source here, so only attach a trimmed OCR
            # excerpt — sending the full text alongside the images is what pushes
            # the request past the model's context window.
            text = text[: settings.extraction_vision_text_chars]
            intro += (
                "\n\nFor reference, here is OCR text of the document (may contain "
                f"errors — trust the image when they differ):\n{text}"
            )
        content: list[dict] = [{"type": "text", "text": intro}]
        for img in (images or [])[: settings.extraction_max_images]:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(img), "detail": "high"},
                }
            )
        human = {"role": "user", "content": content}
        return runnable.invoke([("system", system), human])
