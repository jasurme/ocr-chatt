# Customs Document Intelligence Platform

An AI platform for customs/trade documents that performs **smart OCR →
document classification → structured field extraction → a multilingual chatbot**
(Document Q&A over the extracted JSON **and** RAG over Uzbek customs law).

**Fully local — no external APIs.** Built with **FastAPI**, **LangGraph**
(orchestration), **PaddleOCR** (OCR engine), **Qwen via Ollama** (LLM for
classification / extraction / chat, incl. vision), **bge-m3** (embeddings), and
**Qdrant** (vector store) — everything runs on your own machine.

---

## 1. Features (mapped to the assignment)

| Must-have | Where |
|-----------|-------|
| **Smart OCR (PaddleOCR)** — searchable PDF first, PaddleOCR image fallback, multi-format (PDF/JPG/PNG/DOCX) | [app/ocr/](app/ocr/), [app/ingestion/](app/ingestion/) |
| **Document classification** (invoice, AWB, CMR, packing list, customs declaration, letter, other) | [app/classification/](app/classification/) |
| **Structured extraction** into per-type JSON schemas (extracts *all* fields + an `extra_fields` catch-all) | [app/extraction/](app/extraction/), [app/schemas/](app/schemas/) |
| **Multilingual chatbot** (Uzbek / Russian / English) | [app/chat/](app/chat/) |
| **Document Q&A** answered directly from the structured JSON | [app/chat/qa.py](app/chat/qa.py) |
| **RAG** over Uzbekistan customs law scraped from **lex.uz** | [app/rag/](app/rag/), [app/chat/rag.py](app/chat/rag.py) |
| **Data export** to CSV / Excel | [app/export/](app/export/) |
| **Friendly drag-and-drop UI** (iOS-style) | [app/templates/](app/templates/), [app/static/](app/static/) |
| **LangGraph orchestration** of the whole pipeline + chat router | [app/graph/](app/graph/), [app/chat/router.py](app/chat/router.py) |

---

## 2. Architecture

### Document-processing graph ([app/graph/pipeline.py](app/graph/pipeline.py))

A LangGraph `StateGraph` with real conditional routing:

```
START → ingest ─┬─(unsupported)─────────────→ finalize → END
                └─(ok)→ ocr → classify ─┬─(extract)→ extract → finalize
                                        └─(skip)──────────────→ finalize
```

* **ingest** — load the file into a normalized document (per-page text + page images).
* **ocr** — *smart* OCR: per page, use the embedded text layer if present, else
  render the page and run the OCR engine.
* **classify** — LLM classifies the document type (text-first, vision fallback).
* **extract** — selects a *type-specialized prompt* + Pydantic schema and extracts
  structured JSON (vision-direct for robustness against bad PDF text layers).
* **finalize** — sets the final status; unsupported files are handled gracefully.

### Conversational graph ([app/chat/router.py](app/chat/router.py))

A second `StateGraph` with **per-thread memory** (checkpointer + `thread_id`):

```
START → router ─┬─(doc_qa)──→ doc_qa  → END   (answer from extracted JSON)
                ├─(rag)─────→ rag     → END   (answer from lex.uz knowledge base)
                └─(general)─→ general → END   (greetings / capabilities)
```

The router classifies each turn; conversation history **and** the active document
persist across turns so follow-up questions work in any language.

### Why LangGraph (orchestration)

* **Explicit state + nodes + conditional edges** make the pipeline auditable and
  testable — each node and each router is a plain function unit-tested in isolation.
* **Routing** models the assignment's flowchart directly (supported vs unsupported;
  Q&A vs RAG vs general).
* **Checkpointer memory** gives the chatbot multi-turn context for free.
* Services are **dependency-injected** into the graph, so the entire orchestration
  runs in tests with fake models (no network).

---

## 3. Tech stack & the "mandatory" stack

The assignment mandates **PaddleOCR + local Qwen + a local stack**. This project
is fully local:

| Concern | Engine | Notes |
|--------|--------|-------|
| OCR | **PaddleOCR** (`OCR_PROVIDER=paddle`) → [app/ocr/paddle.py](app/ocr/paddle.py) | the mandated engine; GPU via `paddlepaddle-gpu` |
| LLM (chat / classify / extract incl. vision) | **Qwen via Ollama** (`qwen2.5vl:7b`) → [app/llm/client.py](app/llm/client.py) | one multimodal model; bump chat to `qwen2.5:14b` if VRAM allows |
| Embeddings (RAG) | **bge-m3 via Ollama** | multilingual (uz/ru/en), 1024-dim |
| Vectors | **Qdrant** (embedded or Docker) | local / self-hosted |
| UI | **FastAPI** + custom web UI | drag-and-drop, multilingual chat |

The pipeline follows the spec: smart OCR (searchable text first, else PaddleOCR
on the page image) → classify → extract → answer. Extraction reads page images
with the Qwen vision model by default (`PREFER_VISION_EXTRACTION=true`) and falls
back to the OCR text otherwise.

> First run downloads PaddleOCR + pulls the Ollama models; subsequent runs are
> fast. OCR language is set with `PADDLE_LANG` (default `en`).

---

## 4. Project structure

```
app/
  config.py            # typed settings (.env)
  ingestion/           # load PDF/JPG/PNG/DOCX, render pages, extract text
  ocr/                 # smart OCR service + PaddleOCR provider
  classification/      # LLM document classifier
  schemas/             # DocumentType + per-type Pydantic extraction schemas
  extraction/          # structured extractor + specialized prompts
  graph/               # LangGraph document-processing pipeline
  chat/                # Document Q&A, RAG, and the chat router graph (memory)
  rag/                 # lex.uz scraper, chunking, Qdrant vector store, ingest CLI
  export/              # CSV / Excel export
  api/                 # FastAPI routes + document store
  llm/                 # local model factory (Qwen + bge-m3 via Ollama)
  main.py              # FastAPI app
  static/ templates/   # iOS-style web UI
tests/                 # ~118 tests (offline unit + live integration)
sample_files/          # the provided customs documents
```

---

## 5. Setup

Prerequisites: **Python 3.11+**, and **[Ollama](https://ollama.com)** running
locally. (PDF rendering uses PyMuPDF — no external poppler needed.)

```bash
make install                 # create .venv and install requirements
cp .env.example .env         # defaults already point at local Ollama

# pull the local models (text + vision Qwen + embeddings)
ollama pull qwen2.5:7b       # chat + classification + text extraction (default)
ollama pull qwen2.5vl:7b     # vision: image-only / low-OCR docs (ENABLE_VISION=true)
ollama pull bge-m3           # multilingual RAG embeddings

# (optional, GPU) faster OCR:  pip install paddlepaddle-gpu
```

`.env` keys (see [.env.example](.env.example)): `OLLAMA_BASE_URL`,
`OLLAMA_CHAT_MODEL`, `OLLAMA_EMBED_MODEL`, `OCR_PROVIDER`, `QDRANT_URL`, etc.
No external API keys are needed — everything runs locally.

**Vector store** — by default Qdrant runs *embedded* (on-disk under `data/qdrant`),
so nothing extra is required. To use a server instead:

```bash
docker compose up -d         # starts Qdrant on :6333
# then set QDRANT_URL=http://localhost:6333 in .env
```

---

## 6. Running

```bash
make run                     # http://127.0.0.1:8000
```

Then open the UI: drag-and-drop a document, watch it get classified and extracted,
download CSV/Excel, and chat about it (or about customs law) in Uzbek/Russian/English.

### Seed the RAG knowledge base

```bash
make seed                    # scrape the Customs Code of Uzbekistan from lex.uz
make seed-offline            # or use the bundled curated corpus (no network)
```

(You can also seed from the UI — click the `KB` pill when it shows `KB 0`.)

### GPU notes (e.g. RTX 4090, 24 GB)

One multimodal model (`qwen2.5vl:7b`, ~7 GB) serves chat, classification **and**
vision extraction, so it fits comfortably even alongside other GPU jobs. If you
have spare VRAM, set `OLLAMA_CHAT_MODEL=qwen2.5:14b` for stronger text quality
(keep `qwen2.5vl:7b` as `OLLAMA_VISION_MODEL`). Install `paddlepaddle-gpu` for
GPU-accelerated OCR.

---

## 7. API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/health` | status, configured models, KB size |
| `POST` | `/api/documents` | upload + process a file → result JSON |
| `GET`  | `/api/documents` | list processed documents |
| `GET`  | `/api/documents/{id}` | full result for a document |
| `GET`  | `/api/documents/{id}/export?format=csv\|xlsx` | download extracted data |
| `POST` | `/api/chat` | `{message, thread_id, document_id?}` → `{answer, route, sources}` |
| `POST` | `/api/chat/stream` | same input → **SSE** token stream (`route` → `token`… → `sources` → `done`) |
| `GET`  | `/api/chat/{thread_id}/history` | stored conversation |
| `GET`  | `/api/kb` · `POST /api/kb/seed` | knowledge-base status / seeding |

Interactive docs at `/docs` (Swagger).

---

## 8. Testing

Tests are split into fast **offline** unit tests (fully mocked LLM — no network)
and **integration** tests that exercise the real local stack (Ollama + PaddleOCR
+ lex.uz). Integration tests **auto-skip** when Ollama isn't reachable.

```bash
make test         # offline suite (100+ tests, no models needed, runs in seconds)
make test-int     # live integration tests (needs Ollama running + models pulled)
make test-all     # everything
```

There are tests for **every requirement**: ingestion, smart-OCR routing,
classification accuracy (per sample), structured extraction (per sample),
LangGraph routing, multilingual Q&A, RAG retrieval + scraping, CSV/Excel export,
and full HTTP end-to-end journeys.

---

## 9. Deliverables

* **Source code** — this repository (runnable).
* **README** — this file (architecture, setup, usage).
* **Demo** — run `make run`, upload a file from `sample_files/`, and use the chat.

## 10. Known limitations / future work

* A single PDF can contain multiple documents (e.g. `Final INVOICES .pdf` holds
  two invoices); the current extractor treats a file as one primary document.
* RAG currently indexes the Russian Customs Code by default; add `uz`/`en`
  versions via [app/rag/lexuz.py](app/rag/lexuz.py) `LEXUZ_SOURCES`.
* Swap in PaddleOCR + Qwen for a fully local deployment (see §3).
