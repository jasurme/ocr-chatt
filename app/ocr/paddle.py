"""PaddleOCR provider — the project's mandated, default OCR engine.

Implemented against PaddleOCR 3.x (``predict`` → ``OCRResult["rec_texts"]``),
with a fallback parser for the 2.x ``ocr`` output shape. The engine is built
lazily (and downloads its models on first use) so importing the app stays cheap.

Recognition language is configurable (``PADDLE_LANG``): e.g. ``en`` (latin),
``ru`` / ``cyrillic`` for Russian, ``latin`` for Polish, etc. PaddleOCR uses one
language model per instance.
"""

from __future__ import annotations

import io

from app.config import get_settings
from app.ocr.base import OCRProvider


class PaddleOCRProvider(OCRProvider):
    name = "paddle"

    def __init__(self, lang: str | None = None, use_textline_orientation: bool | None = None):
        s = get_settings()
        self._lang = lang or s.paddle_lang
        self._use_orientation = (
            s.paddle_use_textline_orientation
            if use_textline_orientation is None
            else use_textline_orientation
        )
        self._engine = None

    def _ensure_engine(self):
        if self._engine is not None:
            return
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover - only when paddle absent
            raise RuntimeError(
                "PaddleOCR is not installed. Install it with "
                "`pip install paddlepaddle paddleocr` (it is the default OCR engine)."
            ) from exc
        try:
            # 3.x signature; disable the extra doc-orientation/unwarp models for speed.
            # enable_mkldnn=False: PaddlePaddle 3.x's oneDNN executor crashes on the
            # OCR models with "ConvertPirAttribute2RuntimeAttribute not support
            # [pir::ArrayAttribute<pir::DoubleAttribute>]"; the native CPU path works.
            self._engine = PaddleOCR(
                lang=self._lang,
                use_textline_orientation=self._use_orientation,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                enable_mkldnn=False,
            )
        except TypeError:
            self._engine = PaddleOCR(lang=self._lang, enable_mkldnn=False)

    def ocr_image(self, image_png: bytes, language_hint: str | None = None) -> str:
        self._ensure_engine()
        import numpy as np
        from PIL import Image

        image = np.array(Image.open(io.BytesIO(image_png)).convert("RGB"))
        try:
            results = self._engine.predict(image)  # PaddleOCR 3.x
        except AttributeError:  # pragma: no cover - older paddle
            results = self._engine.ocr(image)
        return _extract_text(results)


def _extract_text(results) -> str:
    """Normalize PaddleOCR output (3.x dict-like or 2.x nested list) to text."""
    lines: list[str] = []
    for res in results or []:
        texts = None
        try:
            texts = res["rec_texts"]  # 3.x OCRResult
        except Exception:
            texts = getattr(res, "rec_texts", None)
        if texts is not None:
            lines.extend(str(t) for t in texts)
            continue
        # 2.x: res is a list of [box, (text, score)]
        for line in res or []:
            if isinstance(line, (list, tuple)) and len(line) >= 2 and line[1]:
                lines.append(str(line[1][0]))
    return "\n".join(t for t in lines if t and t.strip()).strip()
