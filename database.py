"""
database.py – SQLite persistence layer for the AI Knowledge Assistant.

Tables
------
users         – tracked users (default guest user auto-created).
documents     – every uploaded / ingested PDF.
chat_history  – full Q&A log with mode and sources.

The database file is stored at ``knowledge_assistant.db`` in the project
root.  All public methods are class-level on :pyclass:`Database`; a
module-level singleton ``db`` is provided for convenience.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DB_PATH: str = "knowledge_assistant.db"


class Database:
    """Thin wrapper around a SQLite database.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.
    """

    def __init__(self, db_path: str = _DB_PATH) -> None:
        self.db_path: str = db_path
        self._init_db()

    # ------------------------------------------------------------------
    # Connection helper
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Return a new connection with row-factory enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables if they don't already exist."""
        conn = self._get_conn()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    username    TEXT    UNIQUE NOT NULL,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename     TEXT    NOT NULL,
                    file_size    INTEGER DEFAULT 0,
                    chunks       INTEGER DEFAULT 0,
                    status       TEXT    DEFAULT 'uploaded',
                    uploaded_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ingested_at  TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chat_history (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     INTEGER DEFAULT 1,
                    question    TEXT    NOT NULL,
                    answer      TEXT    NOT NULL,
                    mode        TEXT    NOT NULL,
                    sources     TEXT,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );
                """
            )

            # Ensure a default guest user exists.
            conn.execute(
                "INSERT OR IGNORE INTO users (id, username) VALUES (1, 'guest')"
            )
            conn.commit()
            logger.info("Database initialised at %s", Path(self.db_path).resolve())
        finally:
            conn.close()

    # ==================================================================
    # USERS
    # ==================================================================

    def get_or_create_user(self, username: str = "guest") -> Dict[str, Any]:
        """Return the user row, creating it if necessary."""
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO users (username) VALUES (?)", (username,)
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
            return dict(row)
        finally:
            conn.close()

    # ==================================================================
    # DOCUMENTS
    # ==================================================================

    def add_document(self, filename: str, file_size: int = 0) -> int:
        """Record a newly uploaded document. Returns the row id."""
        conn = self._get_conn()
        try:
            # Avoid duplicates: if the same filename already exists, update it.
            existing = conn.execute(
                "SELECT id FROM documents WHERE filename = ?", (filename,)
            ).fetchone()
            if existing:
                conn.execute(
                    """UPDATE documents
                       SET file_size = ?, status = 'uploaded',
                           uploaded_at = CURRENT_TIMESTAMP, ingested_at = NULL, chunks = 0
                     WHERE id = ?""",
                    (file_size, existing["id"]),
                )
                conn.commit()
                return existing["id"]

            cur = conn.execute(
                "INSERT INTO documents (filename, file_size) VALUES (?, ?)",
                (filename, file_size),
            )
            conn.commit()
            return cur.lastrowid  # type: ignore[return-value]
        finally:
            conn.close()

    def update_document_status(
        self,
        filename: str,
        status: str,
        chunks: int = 0,
    ) -> None:
        """Update a document's status (e.g. ``ingested``)."""
        conn = self._get_conn()
        try:
            ingested_at = datetime.utcnow().isoformat() if status == "ingested" else None
            conn.execute(
                """UPDATE documents
                   SET status = ?, chunks = ?, ingested_at = ?
                 WHERE filename = ?""",
                (status, chunks, ingested_at, filename),
            )
            conn.commit()
        finally:
            conn.close()

    def mark_all_ingested(self, total_chunks: int) -> None:
        """Mark every 'uploaded' document as 'ingested'."""
        conn = self._get_conn()
        try:
            doc_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM documents WHERE status = 'uploaded'"
            ).fetchone()["cnt"]
            chunks_each = total_chunks // max(doc_count, 1)

            conn.execute(
                """UPDATE documents
                   SET status = 'ingested',
                       chunks = ?,
                       ingested_at = CURRENT_TIMESTAMP
                 WHERE status = 'uploaded'""",
                (chunks_each,),
            )
            conn.commit()
        finally:
            conn.close()

    def get_documents(self) -> List[Dict[str, Any]]:
        """Return all documents ordered by most recent first."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY uploaded_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ==================================================================
    # CHAT HISTORY
    # ==================================================================

    def add_chat(
        self,
        question: str,
        answer: str,
        mode: str,
        sources: Optional[List[str]] = None,
        user_id: int = 1,
    ) -> int:
        """Log a Q&A exchange. Returns the row id."""
        conn = self._get_conn()
        try:
            sources_json = json.dumps(sources or [])
            cur = conn.execute(
                """INSERT INTO chat_history (user_id, question, answer, mode, sources)
                   VALUES (?, ?, ?, ?, ?)""",
                (user_id, question, answer, mode, sources_json),
            )
            conn.commit()
            return cur.lastrowid  # type: ignore[return-value]
        finally:
            conn.close()

    def get_chat_history(
        self, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Return chat history (most recent first)."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT ch.*, u.username
                     FROM chat_history ch
                     LEFT JOIN users u ON ch.user_id = u.id
                    ORDER BY ch.created_at DESC
                    LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["sources"] = json.loads(d.get("sources") or "[]")
                result.append(d)
            return result
        finally:
            conn.close()

    # ==================================================================
    # ANALYTICS
    # ==================================================================

    def get_analytics(self) -> Dict[str, Any]:
        """Return aggregated statistics for the dashboard."""
        conn = self._get_conn()
        try:
            total_chats = conn.execute(
                "SELECT COUNT(*) as cnt FROM chat_history"
            ).fetchone()["cnt"]

            total_docs = conn.execute(
                "SELECT COUNT(*) as cnt FROM documents"
            ).fetchone()["cnt"]

            ingested_docs = conn.execute(
                "SELECT COUNT(*) as cnt FROM documents WHERE status = 'ingested'"
            ).fetchone()["cnt"]

            total_chunks = conn.execute(
                "SELECT COALESCE(SUM(chunks), 0) as cnt FROM documents"
            ).fetchone()["cnt"]

            # Chats per mode
            mode_rows = conn.execute(
                """SELECT mode, COUNT(*) as cnt
                     FROM chat_history
                    GROUP BY mode
                    ORDER BY cnt DESC"""
            ).fetchall()
            chats_per_mode = {r["mode"]: r["cnt"] for r in mode_rows}

            # Most asked questions (top 10 by frequency)
            top_questions = conn.execute(
                """SELECT question, COUNT(*) as cnt
                     FROM chat_history
                    GROUP BY question
                    ORDER BY cnt DESC
                    LIMIT 10"""
            ).fetchall()

            # Recent activity (last 7 days)
            daily_rows = conn.execute(
                """SELECT DATE(created_at) as day, COUNT(*) as cnt
                     FROM chat_history
                    WHERE created_at >= DATE('now', '-7 days')
                    GROUP BY DATE(created_at)
                    ORDER BY day"""
            ).fetchall()

            return {
                "total_chats": total_chats,
                "total_documents": total_docs,
                "ingested_documents": ingested_docs,
                "total_chunks": total_chunks,
                "chats_per_mode": chats_per_mode,
                "top_questions": [
                    {"question": r["question"], "count": r["cnt"]}
                    for r in top_questions
                ],
                "daily_activity": [
                    {"date": r["day"], "chats": r["cnt"]}
                    for r in daily_rows
                ],
            }
        finally:
            conn.close()
