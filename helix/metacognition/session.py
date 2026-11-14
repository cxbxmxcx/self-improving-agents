"""Conversation sessions, shaped like the OpenAI Agents SDK's Session protocol.

A session stores a conversation's chat messages. The Runner reads them before
a run and inserts them after the system message on the first model call (a
PRE_MODEL hook, so the helix loop is unchanged), then stores the run's input
and the messages it added.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Session(Protocol):
    session_id: str

    async def get_items(self, limit: int | None = None) -> list[dict[str, Any]]: ...

    async def add_items(self, items: list[dict[str, Any]]) -> None: ...

    async def pop_item(self) -> dict[str, Any] | None: ...

    async def clear_session(self) -> None: ...


class SQLiteSession:
    """A session kept in SQLite: in memory by default, or in a file shared across processes."""

    def __init__(self, session_id: str, db_path: str | Path = ":memory:") -> None:
        self.session_id = session_id
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS session_items (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                           " session_id TEXT NOT NULL, item TEXT NOT NULL)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS session_items_by_session ON session_items (session_id, id)")
        self._conn.commit()

    async def get_items(self, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if limit is None:
                rows = self._conn.execute("SELECT item FROM session_items WHERE session_id = ? ORDER BY id",
                                          (self.session_id,)).fetchall()
            else:
                rows = self._conn.execute("SELECT item FROM session_items WHERE session_id = ? ORDER BY id DESC"
                                          " LIMIT ?", (self.session_id, limit)).fetchall()[::-1]
        return [json.loads(r[0]) for r in rows]

    async def add_items(self, items: list[dict[str, Any]]) -> None:
        with self._lock:
            self._conn.executemany("INSERT INTO session_items (session_id, item) VALUES (?, ?)",
                                   [(self.session_id, json.dumps(i, default=str)) for i in items])
            self._conn.commit()

    async def pop_item(self) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT id, item FROM session_items WHERE session_id = ? ORDER BY id DESC"
                                     " LIMIT 1", (self.session_id,)).fetchone()
            if row is None:
                return None
            self._conn.execute("DELETE FROM session_items WHERE id = ?", (row[0],))
            self._conn.commit()
        return json.loads(row[1])

    async def clear_session(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM session_items WHERE session_id = ?", (self.session_id,))
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()
