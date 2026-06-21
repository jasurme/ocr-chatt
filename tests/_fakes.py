"""Lightweight fake LangChain chat models for offline unit tests.

Not collected by pytest (filename has no ``test_`` prefix). Imported by tests
that need to drive classification / extraction / chat without hitting a live model.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Callable

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage


class _FakeStructured:
    def __init__(self, factory: Callable[[Any], Any], recorder: list):
        self._factory = factory
        self._recorder = recorder

    def invoke(self, messages, *args, **kwargs):
        self._recorder.append(messages)
        return self._factory(messages)


class FakeChatModel:
    """Mimics the slice of the chat model the app uses.

    Parameters
    ----------
    structured_response : a fixed object returned by the structured runnable.
    response_factory     : callable(messages) -> object, for dynamic responses.
    content              : str or callable(messages)->str for plain ``invoke``.
    raise_on_invoke      : exception instance to raise (tests error handling).
    """

    def __init__(
        self,
        structured_response: Any = None,
        response_factory: Callable[[Any], Any] | None = None,
        content: Any = None,
        raise_on_invoke: Exception | None = None,
        structured_by_schema: dict | None = None,
    ):
        self.structured_response = structured_response
        self.response_factory = response_factory
        self.content = content
        self.raise_on_invoke = raise_on_invoke
        # Map {SchemaClass: response} so one fake can answer multiple
        # with_structured_output(schema) calls (e.g. keyword extraction + answer).
        self.structured_by_schema = structured_by_schema or {}
        self.captured: list = []  # messages sent to structured runnable
        self.invoked: list = []  # messages sent to plain invoke
        self.bound_tools: Any = None
        self.schema: Any = None

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        if self.raise_on_invoke is not None:
            def boom(_messages):
                raise self.raise_on_invoke
            return _FakeStructured(boom, self.captured)
        if schema in self.structured_by_schema:
            resp = self.structured_by_schema[schema]
            return _FakeStructured(lambda _m: resp, self.captured)
        factory = self.response_factory or (lambda _m: self.structured_response)
        return _FakeStructured(factory, self.captured)

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = tools
        return self

    def invoke(self, messages, *args, **kwargs):
        if self.raise_on_invoke is not None:
            raise self.raise_on_invoke
        self.invoked.append(messages)
        c = self.content
        if callable(c):
            c = c(messages)
        return AIMessage(content=c if c is not None else "FAKE RESPONSE")


class FakeOCRProvider:
    """Fake OCR engine: returns canned text, records calls (no network)."""

    name = "fake"

    def __init__(self, text: str = "FAKE OCR TEXT FROM IMAGE LONG ENOUGH TO CLASSIFY",
                 raise_on_ocr: Exception | None = None):
        self.text = text
        self.calls = 0
        self.raise_on_ocr = raise_on_ocr

    def ocr_image(self, image_png: bytes, language_hint: str | None = None) -> str:
        self.calls += 1
        if self.raise_on_ocr is not None:
            raise self.raise_on_ocr
        return self.text


class HashEmbeddings(Embeddings):
    """Deterministic bag-of-words hashing embeddings (no network).

    Word-overlap drives cosine similarity, so retrieval is meaningful in tests
    without calling the embedding service. Stable across runs (hashlib, not builtin hash()).
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"\w+", text.lower()):
            idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim
            v[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)
