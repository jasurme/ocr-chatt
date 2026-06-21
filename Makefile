.PHONY: install run test test-all test-int seed seed-offline clean lint

VENV := .venv
PY := $(VENV)/bin/python

install:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

run:                       ## Start the web app at http://127.0.0.1:8000 (single process)
	$(PY) -m uvicorn app.main:app --host 127.0.0.1 --port 8000

dev:                       ## Dev server with autoreload (use Qdrant server, not embedded)
	$(PY) -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

test:                      ## Fast offline test suite (no API calls)
	$(PY) -m pytest -m "not integration"

test-int:                  ## Live integration tests (needs Ollama running + models pulled)
	$(PY) -m pytest -m integration

test-all:
	$(PY) -m pytest

seed:                      ## Build the RAG knowledge base from lex.uz (Customs Code)
	$(PY) -m app.rag.ingest

seed-offline:              ## Seed the curated fallback corpus (no network)
	$(PY) -m app.rag.ingest --offline

clean:
	rm -rf data/uploads/* data/processed/* data/qdrant .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
