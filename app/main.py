"""FastAPI application entry point.

Run with:  uvicorn app.main:app --reload
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import ROOT_DIR, get_settings

_STATIC_DIR = ROOT_DIR / "app" / "static"
_TEMPLATES_DIR = ROOT_DIR / "app" / "templates"


def create_app() -> FastAPI:
    settings = get_settings()
    settings.ensure_dirs()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Customs Document Intelligence Platform — OCR, extraction, chatbot.",
    )
    app.include_router(router)

    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        index_file = _TEMPLATES_DIR / "index.html"
        if index_file.is_file():
            return HTMLResponse(index_file.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>Customs Document Intelligence</h1><p>UI not built yet.</p>")

    @app.get("/favicon.ico")
    def favicon():
        return JSONResponse({}, status_code=204)

    return app


app = create_app()
