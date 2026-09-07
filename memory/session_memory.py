"""
Short-term working memory: a bounded rolling conversation buffer kept in
process memory for prompt context, plus a SQLite-backed session log so
conversations survive restarts and can be reviewed/audited later.
"""
from __future__ import annotations

import json
import sqlite3
import time
from collections import deque
from contextlib import closing
from pathlib import Path
from typing import Any, Deque, Dict, List

from core.config_loader import get_settings, resolve_path


class SessionMemory:
    def __init__(self):
        self.settings = get_settings()["memory"]
        self._buffer: Deque[Dict[str, Any]] = deque(maxlen=self.settings.get("session_buffer_max_turns", 20))
        self._db_path = resolve_path(self.settings["sqlite_path"])
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        # closing() matters here: sqlite3's own context manager commits or rolls back
        # the transaction but leaves the connection OPEN. add_turn runs twice per
        # conversation turn, so without this the process leaks a handle every turn.
        with closing(sqlite3.connect(self._db_path)) as conn, conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    session_id TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def add_turn(self, role: str, content: str, session_id: str = "default") -> None:
        entry = {"role": role, "content": content, "timestamp": time.time()}
        self._buffer.append(entry)
        with closing(sqlite3.connect(self._db_path)) as conn, conn:
            conn.execute(
                "INSERT INTO conversation_log (timestamp, role, content, session_id) VALUES (?, ?, ?, ?)",
                (entry["timestamp"], role, content, session_id),
            )
            conn.commit()

    def get_recent_messages(self) -> List[Dict[str, str]]:
        """Returns the rolling buffer formatted for the LLM's messages list."""
        return [{"role": e["role"], "content": e["content"]} for e in self._buffer]

    def get_session_history(self, session_id: str = "default", limit: int = 100) -> List[Dict[str, Any]]:
        with closing(sqlite3.connect(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM conversation_log WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
            return [dict(r) for r in reversed(rows)]

    def clear_buffer(self) -> None:
        self._buffer.clear()
