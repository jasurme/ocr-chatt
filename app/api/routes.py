"""HTTP API routes."""

from __future__ import annotations

import io
import json
import shutil
import uuid
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import get_settings
from app.export import to_csv_bytes, to_excel_bytes
from app.graph import process_document
from app.services import (
    get_chat_service,
    get_document_store,
    get_knowledge_base,
    get_pipeline,
)

router = APIRouter(prefix="/api")


# ------------------------------- models ------------------------------------ #
class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"
    document_id: str | None = None  # single active document (back-compat)
    document_ids: list[str] | None = None  # multiple active documents (multi-doc Q&A)


class ChatResponse(BaseModel):
    answer: str
    route: str | None = None
    sources: list = []


class SeedRequest(BaseModel):
    offline: bool = False
    limit: int | None = None
    reset: bool = False


# ------------------------------- health ------------------------------------ #
@router.get("/health")
def health():
    s = get_settings()
    kb_count = None
    try:
        from app.rag.vectorstore import collection_count

        kb_count = collection_count(get_knowledge_base())
    except Exception:
        kb_count = None
    return {
        "status": "ok",
        "app_name": s.app_name,
        "ocr_provider": s.ocr_provider,
        "vector_store": s.vector_store,
        "models": {
            "chat": s.ollama_chat_model,
            "vision": s.ollama_vision_model,
            "embedding": s.ollama_embed_model,
        },
        "kb_count": kb_count,
    }


# ----------------------------- documents ----------------------------------- #
@router.post("/documents")
def upload_document(file: UploadFile = File(...), pipeline=Depends(get_pipeline),
                    store=Depends(get_document_store)):
    settings = get_settings()
    settings.ensure_dirs()
    original = file.filename or "upload"
    suffix = Path(original).suffix
    stored_name = f"{uuid.uuid4().hex[:12]}{suffix}"
    dest = settings.upload_dir / stored_name
    with open(dest, "wb") as fh:
        shutil.copyfileobj(file.file, fh)

    result = process_document(dest, pipeline=pipeline)
    result["filename"] = original  # show the user's name, not the stored uuid
    result["stored_filename"] = stored_name
    store.save(result)
    return result


@router.get("/documents")
def list_documents(store=Depends(get_document_store)):
    return {"documents": store.list()}


@router.get("/documents/{doc_id}")
def get_document(doc_id: str, store=Depends(get_document_store)):
    result = store.get(doc_id)
    if not result:
        raise HTTPException(status_code=404, detail="Document not found")
    return result


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str, store=Depends(get_document_store)):
    result = store.get(doc_id)
    if not result:
        raise HTTPException(status_code=404, detail="Document not found")
    store.delete(doc_id)
    # Best-effort: also remove the stored upload file.
    stored = result.get("stored_filename")
    if stored:
        try:
            (get_settings().upload_dir / stored).unlink(missing_ok=True)
        except OSError:
            pass
    return {"deleted": doc_id}


@router.get("/documents/{doc_id}/export")
def export_document(
    doc_id: str,
    format: str = Query("xlsx", pattern="^(csv|xlsx)$"),
    store=Depends(get_document_store),
):
    result = store.get(doc_id)
    if not result:
        raise HTTPException(status_code=404, detail="Document not found")

    stem = Path(result.get("filename") or "document").stem
    doc_type = result.get("doc_type") or "data"
    if format == "csv":
        data = to_csv_bytes(result)
        media = "text/csv"
        fname = f"{stem}_{doc_type}.csv"
    else:
        data = to_excel_bytes(result)
        media = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        fname = f"{stem}_{doc_type}.xlsx"
    # RFC 5987: ASCII fallback + UTF-8 name so non-Latin filenames work in headers.
    ascii_name = fname.encode("ascii", "ignore").decode() or f"export.{format}"
    disposition = (
        f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(fname)}"
    )
    return StreamingResponse(
        io.BytesIO(data),
        media_type=media,
        headers={"Content-Disposition": disposition},
    )


# ------------------------------- chat -------------------------------------- #
def _resolve_documents(req: ChatRequest, store) -> list | None:
    """Resolve the request's active document(s) to stored result dicts.

    Returns None when the request specifies no documents at all — the chat thread
    then keeps whatever was already active (memory across turns). An explicit
    (possibly empty) ``document_ids`` list overrides the thread's active set,
    so passing ``[]`` clears it.
    """
    if req.document_ids is not None:
        ids = req.document_ids
    elif req.document_id:
        ids = [req.document_id]
    else:
        return None
    docs = []
    for doc_id in ids:
        ctx = store.get(doc_id)
        if not ctx:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
        docs.append(ctx)
    return docs


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, chat_service=Depends(get_chat_service),
         store=Depends(get_document_store)):
    documents = _resolve_documents(req, store)
    out = chat_service.chat(req.message, thread_id=req.thread_id, documents=documents)
    return ChatResponse(**out)


@router.post("/chat/stream")
def chat_stream(req: ChatRequest, chat_service=Depends(get_chat_service),
                store=Depends(get_document_store)):
    """Server-Sent Events stream of the reply (token-by-token)."""
    documents = _resolve_documents(req, store)

    def event_stream():
        try:
            for event in chat_service.stream_chat(
                req.message, thread_id=req.thread_id, documents=documents
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # surface errors to the client without a 500 mid-stream
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/chat/{thread_id}/history")
def chat_history(thread_id: str, chat_service=Depends(get_chat_service)):
    return {"thread_id": thread_id, "messages": chat_service.history(thread_id)}


# --------------------------- knowledge base -------------------------------- #
@router.get("/kb")
def kb_status():
    try:
        from app.rag.vectorstore import collection_count

        return {"count": collection_count(get_knowledge_base())}
    except Exception as exc:
        return {"count": None, "error": str(exc)}


@router.post("/kb/seed")
def kb_seed(req: SeedRequest):
    from app.rag import seed_knowledge_base
    from app.rag.vectorstore import collection_count

    kb = get_knowledge_base()
    n = seed_knowledge_base(
        vectorstore=kb, use_network=not req.offline, limit=req.limit, reset=req.reset
    )
    return {"seeded": n, "count": collection_count(kb)}
