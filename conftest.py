"""Shared pytest fixtures and path setup.

Lives at the repo root so `import app...` works regardless of where pytest is
invoked, and exposes the bundled sample documents as fixtures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLE_DIR = ROOT / "sample_files"


@pytest.fixture(scope="session")
def sample_dir() -> Path:
    return SAMPLE_DIR


# Individual sample documents, mapped to their real-world type for assertions.
SAMPLES = {
    "awb_jpg": SAMPLE_DIR / "Avia.jpg",
    "gtd_jpg": SAMPLE_DIR / "gtd.jpg",
    "invoice_pdf": SAMPLE_DIR / "Final INVOICES .pdf",
    "invoice_pdf2": SAMPLE_DIR / "инв.PDF",
    "letter_pdf": SAMPLE_DIR / "BVS.pdf",
    "junk_docx": SAMPLE_DIR / "bunyod.docx",
}


@pytest.fixture(scope="session")
def samples() -> dict[str, Path]:
    return dict(SAMPLES)


@pytest.fixture(params=list(SAMPLES.keys()))
def any_sample(request) -> Path:
    """Parametrized fixture yielding each sample file path in turn.

    Skips gracefully if the bundled sample documents aren't present (they are
    optional — the app never needs them; only sample-based tests do).
    """
    path = SAMPLES[request.param]
    if not path.exists():
        pytest.skip(f"sample file missing: {path.name} (place it in sample_files/)")
    return path


def ollama_reachable() -> bool:
    """True when a local Ollama server is reachable (gates live integration tests)."""
    import requests

    from app.config import get_settings

    try:
        return requests.get(f"{get_settings().ollama_base_url}/api/tags", timeout=2).ok
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not ollama_reachable(),
    reason="Ollama not reachable; skipping live integration test.",
)
