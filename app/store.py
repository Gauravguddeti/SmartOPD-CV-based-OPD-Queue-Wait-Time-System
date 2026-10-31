"""Tiny JSON persistence with atomic writes.

All runtime data lives in <project>/data/state.json - nothing ever leaves
this workspace.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Callable, Dict

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


class Store:
    """Loads/saves a single JSON document atomically (write temp + replace)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def load(self, default_factory: Callable[[], Dict]) -> Dict:
        with self._lock:
            if not self.path.exists():
                return default_factory()
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    return default_factory()
                return data
            except (json.JSONDecodeError, OSError):
                # Corrupt/partial save from a hard shutdown: start fresh
                # rather than crashing the clinic app.
                return default_factory()

    def save(self, state: Dict) -> None:
        with self._lock:
            fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(state, f, ensure_ascii=False, indent=1)
                os.replace(tmp, self.path)
            finally:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
