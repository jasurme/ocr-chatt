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
for m in "$CHAT_MODEL" "$VISION_MODEL" "$EMBED_MODEL"; do
  echo "    pulling $m"
  ollama pull "$m"
done

# Scrape the full Customs Code from lex.uz; fall back to the offline corpus if
# the network is unavailable, so the app still has a knowledge base either way.
seed_from_lexuz() {
  echo "    scraping lex.uz (uz, ru, en)"
  if ! "$PY" -m app.rag.ingest --reset --lang uz,ru,en; then
    echo "    lex.uz scrape failed — falling back to the bundled offline corpus"
    "$PY" -m app.rag.ingest --offline \
      || echo "    (seed failed — you can seed later from the UI 'KB' pill)"
  fi
}

echo "==> 4/5  RAG knowledge base"
case "${SEED:-auto}" in
  skip)
    echo "    SEED=skip — leaving knowledge base as-is" ;;
  offline)
    echo "    seeding bundled offline corpus (no network)"
    "$PY" -m app.rag.ingest --offline \
      || echo "    (seed failed — you can seed later from the UI 'KB' pill)" ;;
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
