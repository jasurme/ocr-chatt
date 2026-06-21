"""In-memory + on-disk store for processed-document results."""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path


class DocumentStore:
    """Keeps processed results keyed by a short id.

    Results are held in memory for fast access and also persisted as JSON under
    ``processed_dir`` so they survive a restart.
    """

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._load_existing()

    def _load_existing(self) -> None:
        for fp in self.dir.glob("*.json"):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("id"):
                    self._mem[data["id"]] = data
            except (OSError, json.JSONDecodeError):
                continue

    def save(self, result: dict) -> str:
        doc_id = result.get("id") or uuid.uuid4().hex[:12]
        result["id"] = doc_id
        with self._lock:
            self._mem[doc_id] = result
            (self.dir / f"{doc_id}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return doc_id

    def get(self, doc_id: str) -> dict | None:
        return self._mem.get(doc_id)

    def delete(self, doc_id: str) -> bool:
        """Remove a processed document from memory and disk. Returns True if it existed."""
        with self._lock:
            existed = self._mem.pop(doc_id, None) is not None
            fp = self.dir / f"{doc_id}.json"
            try:
                fp.unlink(missing_ok=True)
            except OSError:
                pass
        return existed

    def list(self) -> list[dict]:
        """Lightweight summaries, newest-first by insertion order."""
        out = []
        for r in self._mem.values():
            out.append(
                {
                    "id": r.get("id"),
                    "filename": r.get("filename"),
                    "doc_type": r.get("doc_type"),
                    "status": r.get("status"),
                    "language": r.get("language"),
                }
            )
        return list(reversed(out))
