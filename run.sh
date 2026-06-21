#!/usr/bin/env bash
#
# One-shot setup + run for the Customs Document Intelligence Platform.
#
#   ./run.sh                      # install, pull models, seed KB from lex.uz, run

set -euo pipefail
cd "$(dirname "$0")"

VENV=.venv
PY="$VENV/bin/python"

echo "==> 1/5  Python virtualenv + dependencies"
if ! command -v python3 >/dev/null 2>&1; then
  echo "    ERROR: python3 not found. Install Python 3.11+ and re-run." >&2
  exit 1
fi
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "    ERROR: Python 3.11+ required (found $(python3 -V 2>&1)). Install a newer Python and re-run." >&2
  exit 1
fi
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
"$PY" -m pip install --upgrade pip >/dev/null
"$PY" -m pip install -r requirements.txt

echo "==> 2/5  Config (.env)"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "    created .env from .env.example"
else
  echo "    .env already present (leaving it untouched)"
fi

# Pull whatever models .env actually asks for (so bumping the model 'just works').
read_env() { grep -E "^$1=" .env 2>/dev/null | head -1 | cut -d= -f2- | awk '{print $1}'; }
CHAT_MODEL="$(read_env OLLAMA_CHAT_MODEL)";   CHAT_MODEL="${CHAT_MODEL:-qwen2.5:7b}"
VISION_MODEL="$(read_env OLLAMA_VISION_MODEL)"; VISION_MODEL="${VISION_MODEL:-qwen2.5vl:7b}"
EMBED_MODEL="$(read_env OLLAMA_EMBED_MODEL)";  EMBED_MODEL="${EMBED_MODEL:-bge-m3}"

echo "==> 3/5  Ollama models"
if ! command -v ollama >/dev/null 2>&1; then
  echo "    ERROR: 'ollama' not found. Install it from https://ollama.com and re-run." >&2
  exit 1
fi
if ! ollama list >/dev/null 2>&1; then
  echo "    ERROR: Ollama is installed but not running." >&2
  echo "    Start it first: run 'ollama serve' (or open the Ollama app), then re-run ./run.sh" >&2
  exit 1
fi
for m in "$CHAT_MODEL" "$VISION_MODEL" "$EMBED_MODEL"; do
  echo "    pulling $m"
  ollama pull "$m"
done

# Scrape the full Customs Code from lex.uz (uz, ru, en) into the vector store.
seed_from_lexuz() {
  echo "    scraping lex.uz (uz, ru, en)"
  "$PY" -m app.rag.ingest --reset --lang uz,ru,en \
    || echo "    (seed failed — check your network, then seed later from the UI 'KB' pill)"
}

echo "==> 4/5  RAG knowledge base"
case "${SEED:-auto}" in
  skip)
    echo "    SEED=skip — leaving knowledge base as-is" ;;
  force)
    seed_from_lexuz ;;
  *)
    # Default: seed from lex.uz, but skip if a KB already exists (re-run friendly).
    if [ -d data/qdrant ]; then
      echo "    knowledge base already exists (use SEED=force to rebuild from lex.uz)"
    else
      seed_from_lexuz
    fi ;;
esac

echo "==> 5/5  Starting app at http://127.0.0.1:8000  (Ctrl-C to stop)"
exec "$PY" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
